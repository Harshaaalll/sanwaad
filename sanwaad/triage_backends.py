"""Which model reads a comment first: triage backends behind one interface.

Triage is the only model call every inbound item pays for, so it is the call
where a cheaper or faster model moves the cost per case the most — and the one
where a worse model does the most damage, because every later step branches on
its labels. That combination is why the choice is measured rather than argued:
`python -m sanwaad.evals.triage_compare` scores each backend here against
labelled complaints, and `SANWAAD_TRIAGE_BACKEND` puts the winner in front of
live traffic.

Three backends:

- **gemini**: the generative triage the pipeline has always used. It writes
  every field, the summary included.
- **laya**: an open-weight *decision* model (Convai, Apache-2.0) that runs
  locally. It does not generate text; it answers typed questions — pick one of
  these categories, place this on a severity scale, how likely is this true —
  with a probability for every option. Cheap and fast, and it says how sure it
  is, which a generative model does not.
- **jev**: TypeSafe AI's hosted decision model, the same idea as Laya. Its API
  is in early access and undocumented publicly, so the adapter is a declared
  gap: it reports itself unavailable, with the reason, until it is written.

A decision model cannot write triage's `summary` or pick the reply `language`,
so on the live path those two come from one small call on the triage tier
("hybrid"). Labels from the decision model, words from the generative one.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from loguru import logger

from .models import Category

BACKENDS = ("gemini", "laya", "jev")


class BackendUnavailable(RuntimeError):
    """This backend cannot run here, and the message says why."""


@dataclass
class TriageLabels:
    """What a backend decided about one comment, in the pipeline's vocabulary."""

    backend: str
    model: str
    is_complaint: bool
    category: str
    severity: int
    sentiment: str
    needs_private_data: bool
    # How sure the backend is of `category`, 0-1. None when it cannot say:
    # a generative model returns an answer, not a distribution.
    confidence: Optional[float] = None
    category_probs: Optional[dict[str, float]] = None
    latency_ms: float = 0.0
    cost: dict = field(default_factory=dict)
    # The complete Triage, when the backend wrote one (gemini does).
    triage: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _cost(model: str, **extra) -> dict:
    return {"stage": "triage", "model": model, "usd": 0.0, "inr": 0.0,
            "prompt_tokens": 0, "output_tokens": 0, "attempts": 1, **extra}


# ---------------------------------------------------------------------------
# The questions a decision model is asked
#
# Written for NimbusPay rather than taken from Laya's generic support preset,
# because the criteria text is the model's only description of each label: a
# category the criteria describe vaguely is a category it confuses. They mirror
# the generative prompt's rubric, so the two backends are answering the same
# question and a disagreement means something.
# ---------------------------------------------------------------------------

CATEGORY_CRITERIA: dict[str, str] = {
    Category.BILLING.value: "an unexpected charge, fee, deduction or cashback question",
    Category.REFUND.value: "money to be returned: a double debit, a failed payment not reversed, "
                           "a refund or reversal not received",
    Category.SERVICE_OUTAGE.value: "the app, UPI or payments failing, down, stuck or pending "
                                   "for many people right now",
    Category.DELIVERY.value: "an order, card or merchant delivery that did not arrive",
    Category.ACCOUNT_ACCESS.value: "cannot log in, wallet frozen or blocked, KYC problem",
    Category.AGENT_BEHAVIOUR.value: "a support agent or staff member was rude, unhelpful "
                                    "or broke a promise",
    Category.DATA_PRIVACY.value: "fraud call, phishing, OTP request, leaked or misused personal data",
    Category.PRAISE.value: "a compliment or thanks, nothing to fix",
    Category.OFF_TOPIC.value: "not about this company's service, spam, or no actionable content",
}

SEVERITY_LEVELS = [
    "mild annoyance, no money at stake",
    "routine issue, small amount, no deadline pressure",
    "real money blocked or lost, customer inconvenienced",
    "large amount, repeated contact, or account fully inaccessible",
    "regulatory, legal, fraud, or safety exposure",
]

SENTIMENTS = {
    "angry": "hostile, furious, or using strong language",
    "frustrated": "annoyed or upset but civil",
    "neutral": "matter of fact",
    "positive": "pleased or thankful",
}


def decision_questions() -> dict:
    """Triage as typed questions. Every instruction names `message`, the state key."""
    return {
        "category": {
            "type": "choice",
            "instructions": "What is the comment in `message` about?",
            "criteria": dict(CATEGORY_CRITERIA),
        },
        "severity": {
            "type": "score",
            "instructions": "How serious is the customer impact described in `message`? "
                            "Judge the impact, not the tone.",
            "criteria": list(SEVERITY_LEVELS),
        },
        "sentiment": {
            "type": "choice",
            "instructions": "How does the author of `message` sound?",
            "criteria": dict(SENTIMENTS),
        },
        "is_complaint": {
            "type": "noul",
            "instructions": "Is `message` a complaint or problem the company should act on?",
        },
        "needs_private_data": {
            "type": "noul",
            "instructions": "Would resolving `message` need the customer's account or "
                            "transaction details, which cannot be discussed in public?",
        },
    }


def labels_from_decision(payload: dict, *, backend: str, model: str,
                         latency_ms: float) -> TriageLabels:
    """Map a decision model's answers onto the pipeline's triage fields.

    Shared by Laya and, once written, Jev: both answer the same typed
    questions, so the mapping lives once.
    """
    answers = payload.get("answers") or {}
    cat = answers["category"]
    sev = answers["severity"]
    # `score` is the expected level, 0-based over the rubric. Rounding the
    # expectation rather than taking the argmax keeps "split between 3 and 4"
    # from snapping to whichever edge happened to be a hair ahead.
    severity = min(5, max(1, int(round(float(sev["score"]))) + 1))
    probs = {k: float(v) for k, v in (cat.get("probabilities") or {}).items()}
    usage = payload.get("usage") or {}
    return TriageLabels(
        backend=backend,
        model=model,
        is_complaint=float(answers["is_complaint"]["noul"]) >= 0.5,
        category=str(cat["choice"]),
        severity=severity,
        sentiment=str(answers["sentiment"]["choice"]),
        needs_private_data=float(answers["needs_private_data"]["noul"]) >= 0.5,
        # `answer_confidence` is the calibrated number; `confidence` is an
        # entropy measure for choice questions. Gate on the calibrated one.
        confidence=float(cat.get("answer_confidence", cat.get("confidence", 0.0))),
        category_probs=probs or None,
        latency_ms=latency_ms,
        cost=_cost(model, prompt_tokens=int(usage.get("input_tokens", 0) or 0)),
    )


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class GeminiBackend:
    name = "gemini"

    def available(self) -> tuple[bool, str]:
        from .llm import is_offline

        if is_offline():
            return True, "offline: no GOOGLE_API_KEY, so this is the keyword stub, not Gemini"
        return True, "live"

    def warm(self) -> None:
        return None

    async def classify(self, text: str, channel: str = "reddit") -> TriageLabels:
        # Imported here, not at the top: the graph imports this module, and the
        # generative triage lives in the graph.
        from .graph.nodes import llm_triage

        started = time.perf_counter()
        result, cost = await llm_triage(text, channel, use_cache=False)
        return TriageLabels(
            backend=self.name,
            model=cost.get("model", "unknown"),
            is_complaint=result.is_complaint,
            category=result.category.value,
            severity=result.severity,
            sentiment=result.sentiment,
            needs_private_data=result.needs_private_data,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            cost=cost,
            triage=result.model_dump(mode="json"),
        )


class LayaBackend:
    """Laya on this machine, CPU by default.

    `SANWAAD_LAYA_MODEL` picks the checkpoint:
      multilingual (default)  mmBERT-base, 322M, 100+ languages
      english                 ModernBERT-large, 421M, English only
      auto                    Laya's router picks by script, and keeps both resident

    Multilingual is the default because this traffic is Hinglish and
    Devanagari as often as English, and Laya's own benchmark shows the English
    checkpoint does not degrade on Hindi, it collapses — to near chance, at
    high confidence. It is also one model in memory instead of two.
    """

    name = "laya"

    def __init__(self) -> None:
        self._router = None
        self._lock = threading.Lock()

    @property
    def checkpoint(self) -> str:
        return os.getenv("SANWAAD_LAYA_MODEL", "multilingual").strip().lower() or "multilingual"

    def available(self) -> tuple[bool, str]:
        missing = [m for m in ("laya", "torch") if importlib.util.find_spec(m) is None]
        if missing:
            return False, (f"not installed ({', '.join(missing)}): "
                           "pip install -r requirements-models.txt")
        return True, f"local, checkpoint {self.checkpoint}"

    def _get_router(self):
        with self._lock:
            if self._router is None:
                ok, why = self.available()
                if not ok:
                    raise BackendUnavailable(why)
                from laya import Router

                auto = self.checkpoint == "auto"
                # One resident model unless routing between two is asked for:
                # this runs next to the embedding model on small hosts.
                self._router = Router(max_loaded=2 if auto else 1,
                                      device=os.getenv("SANWAAD_LAYA_DEVICE") or None)
            return self._router

    def _predict(self, text: str) -> dict:
        router = self._get_router()
        model = None if self.checkpoint == "auto" else self.checkpoint
        return router.predict({"message": text}, decision_questions(), model=model)

    def warm(self) -> None:
        """Download and load the checkpoint, so the first timed call is not a cold start."""
        self._predict("warm up")

    async def classify(self, text: str, channel: str = "reddit") -> TriageLabels:
        started = time.perf_counter()
        # Torch inference blocks; keep it off the event loop.
        payload = await asyncio.to_thread(self._predict, text)
        latency = (time.perf_counter() - started) * 1000.0
        chosen = (payload.get("routing") or {}).get("model") or self.checkpoint
        return labels_from_decision(payload, backend=self.name, model=f"laya-{chosen}",
                                    latency_ms=latency)


class JevBackend:
    """TypeSafe AI's hosted decision model — not yet wired.

    What is missing is the request format, not the idea: Jev takes a state and
    typed questions like Laya does, so once the API is known this sends
    `decision_questions()` and passes the reply through `labels_from_decision`.
    """

    name = "jev"

    def available(self) -> tuple[bool, str]:
        if not os.getenv("TYPESAFE_API_KEY"):
            return False, "no TYPESAFE_API_KEY (early access at console.typesafe.ai)"
        return False, "TYPESAFE_API_KEY is set, but the Jev adapter is not written yet"

    def warm(self) -> None:
        return None

    async def classify(self, text: str, channel: str = "reddit") -> TriageLabels:
        raise BackendUnavailable(self.available()[1])


_INSTANCES: dict[str, object] = {}


def get_backend(name: str):
    name = name.strip().lower()
    if name not in BACKENDS:
        raise ValueError(f"unknown triage backend {name!r}; choose from {', '.join(BACKENDS)}")
    if name not in _INSTANCES:
        _INSTANCES[name] = {"gemini": GeminiBackend, "laya": LayaBackend,
                            "jev": JevBackend}[name]()
    return _INSTANCES[name]


def live_backend_name() -> str:
    """The backend live triage uses. An unknown value is reported, not obeyed."""
    name = os.getenv("SANWAAD_TRIAGE_BACKEND", "gemini").strip().lower() or "gemini"
    if name not in BACKENDS:
        logger.warning(f"SANWAAD_TRIAGE_BACKEND={name!r} is not one of {BACKENDS}; using gemini")
        return "gemini"
    return name


def min_confidence() -> float:
    """Below this, a decision model's category is not trusted and Gemini triages instead."""
    try:
        return float(os.getenv("SANWAAD_TRIAGE_MIN_CONFIDENCE", "0.5"))
    except ValueError:
        return 0.5


def status() -> dict:
    """For /ready and the console: which backend is live, and can each one run."""
    report = {}
    for name in BACKENDS:
        ok, why = get_backend(name).available()
        report[name] = {"available": ok, "detail": why}
    return {"live": live_backend_name(), "min_confidence": min_confidence(),
            "backends": report}
