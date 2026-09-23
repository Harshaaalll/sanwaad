"""Thin LLM layer over google-genai with structured output and token accounting.

Three things this deliberately does that a bare SDK call does not:

1.  Every call returns its cost alongside its result. Cost-per-resolution is a
    headline metric of this project, and metrics that are computed at the call
    site are the only ones that stay correct.
2.  It degrades to a deterministic offline mode when no key is configured, so
    the whole graph runs end to end on a fresh clone. A demo that needs three
    API keys before it prints anything is a demo nobody runs.
3.  It treats the model as an UNRELIABLE UPSTREAM DEPENDENCY, because it is one.
    Every call has a timeout. Output that fails its schema is retried once with
    the validation problem spelled out. A provider error or a second schema
    failure moves to a fallback model. If that fails too, the step's
    deterministic fallback runs and the result is marked `degraded` — or, when
    a step has no safe fallback, a `ModelCallError` is raised rather than
    anything half-formed being passed downstream.

Each of those events — attempts, schema failures, fallbacks, the prompt version
that produced the output — is written onto the cost entry and the trace span,
so "why did this case get a worse reply on Tuesday" has an answer.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from typing import Optional, Type, TypeVar

from loguru import logger
from pydantic import BaseModel

from .config import inr_per_usd, llm_cost
from .obs import TRACER

T = TypeVar("T", bound=BaseModel)

_client = None

# Cost-entry `model` values that did not involve a model call.
NON_CALL_MODELS = frozenset({"", "offline", "rules", "cache", "none"})


class ModelCallError(RuntimeError):
    """Every model and the retry budget were exhausted, and the step had no
    deterministic fallback it could safely use instead."""

    def __init__(self, stage: str, errors: list[str]):
        super().__init__(f"{stage}: all model attempts failed — {'; '.join(errors[-3:])}")
        self.stage = stage
        self.errors = errors


def _get_client():
    global _client
    if _client is not None:
        return _client
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true")
    if not api_key and not use_vertex:
        return None
    from google import genai

    if use_vertex:
        _client = genai.Client(
            vertexai=True,
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "asia-south1"),
        )
    else:
        _client = genai.Client(api_key=api_key)
    return _client


def is_offline() -> bool:
    return _get_client() is None


def prompt_version(system: str) -> str:
    """A version derived from the prompt text itself.

    Hand-maintained version numbers drift from the prompts they describe. A
    hash cannot: change one word of a system prompt and every trace written
    afterwards carries a different version, which is exactly what you need
    when a regression appears and the first question is "what changed?".
    """
    return hashlib.sha256(system.encode("utf-8")).hexdigest()[:8]


async def _generate(client, *, model: str, system: str, user: str, schema: Type[BaseModel],
                    temperature: float, max_output_tokens: Optional[int]):
    """The single place the provider is called. Tests replace this."""
    from google.genai import types

    config = dict(
        system_instruction=system,
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=schema,
    )
    if max_output_tokens:
        config["max_output_tokens"] = max_output_tokens
    return await client.aio.models.generate_content(
        model=model, contents=user, config=types.GenerateContentConfig(**config))


def _parse(resp, schema: Type[T]) -> T:
    parsed = getattr(resp, "parsed", None)
    if parsed is not None:
        payload = parsed.model_dump() if isinstance(parsed, BaseModel) else parsed
        return schema.model_validate(payload)
    # Some models wrap JSON in a fence despite the mime type.
    text = (getattr(resp, "text", "") or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    return schema.model_validate(json.loads(text))


def _entry(stage: str, model: str, *, version: str, usd: float = 0.0, prompt_tokens: int = 0,
           output_tokens: int = 0, attempts: int = 0, schema_failures: int = 0,
           fallback_used: bool = False, degraded: bool = False,
           errors: Optional[list[str]] = None) -> dict:
    return {
        "stage": stage, "model": model, "usd": usd, "inr": usd * inr_per_usd(),
        "prompt_tokens": prompt_tokens, "output_tokens": output_tokens,
        "attempts": attempts, "schema_failures": schema_failures,
        "fallback_used": fallback_used, "degraded": degraded,
        "errors": errors or [], "prompt_version": version,
    }


_SCHEMA_CORRECTION = (
    "\n\nYour previous response did not match the required JSON schema ({problem}). "
    "Respond again with only a JSON object that satisfies the schema exactly."
)


async def structured(
    *,
    model: str,
    system: str,
    user: str,
    schema: Type[T],
    temperature: float = 0.2,
    stage: str = "llm",
    offline_fallback: Optional[dict] = None,
    fallback_model: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    timeout_s: float = 30.0,
    max_latency_ms: Optional[int] = None,
    schema_retries: int = 1,
    trace_id: Optional[str] = None,
) -> tuple[T, dict]:
    """Run a structured-output call. Returns (parsed_model, cost_entry)."""
    version = prompt_version(system)
    client = _get_client()
    if client is None:
        if offline_fallback is None:
            raise RuntimeError(
                f"No GOOGLE_API_KEY set and no offline fallback for stage {stage!r}"
            )
        return schema.model_validate(offline_fallback), _entry(stage, "offline", version=version)

    models = [model] + ([fallback_model] if fallback_model and fallback_model != model else [])
    errors: list[str] = []
    attempts = schema_failures = prompt_tokens = output_tokens = 0
    usd = 0.0

    started = time.perf_counter()
    with TRACER.span(f"llm.{stage}", trace_id=trace_id, prompt_version=version,
                     latency_budget_ms=max_latency_ms) as span:
        for index, current in enumerate(models):
            correction = ""
            for _ in range(1 + schema_retries):
                attempts += 1
                try:
                    resp = await asyncio.wait_for(
                        _generate(client, model=current, system=system, user=user + correction,
                                  schema=schema, temperature=temperature,
                                  max_output_tokens=max_output_tokens),
                        timeout=timeout_s)
                except (asyncio.TimeoutError, TimeoutError):
                    errors.append(f"{current}: timeout after {timeout_s:g}s")
                    break           # a slow provider will not get faster on retry: next model
                except Exception as exc:
                    errors.append(f"{current}: {type(exc).__name__}")
                    break           # rate limit, 5xx, bad request: next model

                usage = getattr(resp, "usage_metadata", None)
                p = getattr(usage, "prompt_token_count", 0) or 0
                o = getattr(usage, "candidates_token_count", 0) or 0
                prompt_tokens += p
                output_tokens += o
                usd += llm_cost(current, p, o)["usd"]

                try:
                    result = _parse(resp, schema)
                except (ValueError, TypeError) as exc:
                    # Valid JSON in the wrong shape is still a failure: the next
                    # step would branch on a field that is not there.
                    schema_failures += 1
                    problem = type(exc).__name__
                    errors.append(f"{current}: output failed schema ({problem})")
                    correction = _SCHEMA_CORRECTION.format(problem=problem)
                    continue

                entry = _entry(stage, current, version=version, usd=usd,
                               prompt_tokens=prompt_tokens, output_tokens=output_tokens,
                               attempts=attempts, schema_failures=schema_failures,
                               fallback_used=index > 0, errors=errors)
                span.set(model=current, attempts=attempts, schema_failures=schema_failures,
                         fallback_used=index > 0, prompt_tokens=prompt_tokens,
                         output_tokens=output_tokens, cost_inr=entry["inr"])
                _check_latency(stage, started, max_latency_ms, span)
                return result, entry

        span.set(degraded=True, attempts=attempts, schema_failures=schema_failures,
                 llm_errors=errors[-3:])
        _check_latency(stage, started, max_latency_ms, span)
        if offline_fallback is None:
            raise ModelCallError(stage, errors)
        return schema.model_validate(offline_fallback), _entry(
            stage, "degraded", version=version, usd=usd, prompt_tokens=prompt_tokens,
            output_tokens=output_tokens, attempts=attempts, schema_failures=schema_failures,
            fallback_used=len(models) > 1, degraded=True, errors=errors)



def _check_latency(stage: str, started: float, budget_ms: Optional[int], span) -> float:
    """Compare what this step actually took against what it was budgeted.

    The budget is not a deadline. `timeout_s` already handles giving up; this
    measures whether a step that succeeded did so acceptably fast, which is a
    different question and one nothing was asking. Recording it on the span is
    what lets `obs.stage_stats` count breaches per stage after the fact, and
    the warning is for the operator watching one slow case in real time.
    """
    elapsed_ms = (time.perf_counter() - started) * 1000
    span.set(latency_ms=round(elapsed_ms, 1))
    if budget_ms and elapsed_ms > budget_ms:
        span.set(over_budget=True)
        logger.warning(f"latency budget: {stage} took {elapsed_ms:.0f}ms "
                       f"against a {budget_ms}ms budget")
    return elapsed_ms


def format_citations(citations: list) -> str:
    """Render retrieved clauses for a prompt.

    The clause id is included in the rendered text on purpose: the drafting
    model is asked to cite ids, and a model cannot cite an identifier it was
    never shown.
    """
    return "\n\n".join(
        f"[{c.clause_id}] {c.heading}\n{c.text}" for c in citations
    )
