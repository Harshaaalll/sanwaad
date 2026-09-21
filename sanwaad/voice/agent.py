"""Browser voice agent over WebRTC.

Transport choice is a cost decision. Telephony providers such as Exotel bill
per minute. A grievance escalation starts from someone already looking at a
web page, so a browser-to-server WebRTC leg carries no per-minute cost at all
— the same Pipecat pipeline, minus the telephony bill. A phone leg stays
possible for customers who want an actual call; see `escalation.channel`.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

from loguru import logger

from ..config import inr_per_usd, llm_cost
from ..models import Citation, VoiceOutcome
from .brief import CitationTracker, build_hotwords, build_voice_prompt
from .numbers import normalise

# Sarvam STT is priced in INR per second of audio; Murf TTS in USD per character.
STT_SARVAM_PER_SEC_INR = 30.0 / 3600.0
TTS_MURF_PER_CHAR_USD = 10.0 / 1_000_000.0  # FALCON


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def _pipecat_language(tag: str):
    from pipecat.transcriptions.language import Language

    return {
        "en-IN": Language.EN_IN, "en-US": Language.EN_US,
        "hi-IN": Language.HI_IN, "hi-Latn": Language.HI_IN,
        "mr-IN": Language.MR_IN, "ta-IN": Language.TA_IN,
        "te-IN": Language.TE_IN, "kn-IN": Language.KN_IN,
        "bn-IN": Language.BN_IN, "gu-IN": Language.GU_IN,
        "ml-IN": Language.ML_IN,
    }.get(tag, Language.EN_IN)


class VoiceMetrics:
    """Per-call token, character and cost accumulation.

    Deliberately the same shape as the text pipeline's cost entries, so a
    voice leg and a text leg roll into one cost-per-resolution figure instead
    of living in two incomparable dashboards.
    """

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cache_read_tokens = 0
        self.tts_characters = 0
        self.started_at = time.time()

    def as_cost_entries(self, model: str) -> list[dict]:
        duration = time.time() - self.started_at
        rate = inr_per_usd()

        stt_inr = duration * STT_SARVAM_PER_SEC_INR
        tts_usd = self.tts_characters * TTS_MURF_PER_CHAR_USD
        llm = llm_cost(model, self.prompt_tokens, self.completion_tokens)

        return [
            {"stage": "voice_stt", "model": "sarvam", "inr": stt_inr,
             "usd": stt_inr / rate, "prompt_tokens": 0, "output_tokens": 0},
            {"stage": "voice_llm", **llm},
            {"stage": "voice_tts", "model": "murf-falcon", "usd": tts_usd,
             "inr": tts_usd * rate, "prompt_tokens": 0, "output_tokens": 0},
        ]


async def run_voice_call(
    webrtc_connection,
    *,
    case_id: str,
    complaint: str,
    public_reply: str,
    summary: str,
    category: str,
    language: str,
    citations: list[Citation],
    max_duration_s: int = 300,
) -> VoiceOutcome:
    """Run one grievance callback over WebRTC. Returns the outcome the graph
    resumes on."""
    import asyncio

    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.frames.frames import (
        Frame,
        LLMRunFrame,
        MetricsFrame,
        TranscriptionFrame,
        TTSSpeakFrame,
        TTSTextFrame,
    )
    from pipecat.metrics.metrics import LLMUsageMetricsData, TTSUsageMetricsData
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from pipecat.services.sarvam.stt import SarvamSTTService
    from pipecat.transports.base_transport import TransportParams
    from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
    from pipecat_murf_tts import MurfTTSService

    pc_lang = _pipecat_language(language)
    metrics = VoiceMetrics()
    tracker = CitationTracker(citations)

    # The vocabulary of this call, from this call's own case. It goes into the
    # prompt so the model reads a near-miss as the word that was meant, and it
    # is what an STT that accepts a bias list would be given.
    hotwords = build_hotwords(complaint=complaint, category=category,
                              citations=citations)
    logger.info(f"[voice:{case_id}] hotwords: {hotwords}")

    system_prompt = build_voice_prompt(
        complaint=complaint, public_reply=public_reply, summary=summary,
        category=category, language=language, citations=citations,
        hotwords=hotwords,
    )

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    stt = SarvamSTTService(
        api_key=_require_env("SARVAM_API_KEY"),
        settings=SarvamSTTService.Settings(
            model=os.getenv("SARVAM_STT_MODEL", "saarika:v2.5"),
            language=pc_lang,
            vad_signals=True,
        ),
        keepalive_timeout=10.0,
    )

    llm = _build_llm(system_prompt)

    tts = MurfTTSService(
        api_key=_require_env("MURF_API_KEY"),
        params=MurfTTSService.InputParams(
            voice_id=os.getenv("MURF_VOICE_ID", "en-IN-aditya"),
            style=os.getenv("MURF_STYLE", "Conversational"),
            rate=1.0, pitch=0, sample_rate=24000, format="PCM",
            model=os.getenv("MURF_MODEL", "FALCON"),
        ),
    )

    heard_amounts: list[float] = []

    class SpokenNumbers(FrameProcessor):
        """Words to digits, before the transcript reaches the model.

        It sits directly after the STT because everything downstream — the
        model's reasoning, any ledger lookup, the consistency receipt — is
        arithmetic on an amount, and "chaar hazaar paanch sau" is not a number
        to any of them. What the customer actually said is logged unchanged
        beside the rewrite, so a wrong conversion is visible rather than
        silently believed.
        """

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if isinstance(frame, TranscriptionFrame) and frame.text:
                found = normalise(frame.text)
                if found.changed:
                    logger.info(f"[voice:{case_id}] heard {frame.text!r} "
                                f"→ {found.text!r}")
                    heard_amounts.extend(found.amounts_inr)
                    frame.text = found.text
            await self.push_frame(frame, direction)

    class Observer(FrameProcessor):
        """Feeds the citation tracker and the cost meter off the live stream.

        Reading TTSTextFrame rather than the raw LLM stream means we track what
        the customer actually heard, not what the model generated before any
        interruption cut it off. On a barge-in those differ, and only the
        former is a claim we made.
        """

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if isinstance(frame, TTSTextFrame):
                tracker.observe(frame.text)
            elif isinstance(frame, MetricsFrame):
                for d in frame.data:
                    if isinstance(d, LLMUsageMetricsData):
                        u = d.value
                        metrics.prompt_tokens += getattr(u, "prompt_tokens", 0) or 0
                        metrics.completion_tokens += getattr(u, "completion_tokens", 0) or 0
                        metrics.cache_read_tokens += getattr(u, "cache_read_input_tokens", 0) or 0
                    elif isinstance(d, TTSUsageMetricsData):
                        metrics.tts_characters += int(d.value or 0)
            await self.push_frame(frame, direction)

    context = LLMContext(messages=[])
    aggregators = LLMContextAggregatorPair(context)

    pipeline = Pipeline([
        transport.input(),
        stt,
        SpokenNumbers(),
        aggregators.user(),
        llm,
        tts,
        Observer(),
        transport.output(),
        aggregators.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )

    @transport.event_handler("on_client_connected")
    async def _on_connected(_transport, _client):
        logger.info(f"[voice:{case_id}] client connected")
        await task.queue_frame(LLMRunFrame())

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnected(_transport, _client):
        logger.info(f"[voice:{case_id}] client disconnected")
        await task.cancel()

    async def _cap():
        await asyncio.sleep(max_duration_s)
        logger.warning(f"[voice:{case_id}] hard cap {max_duration_s}s reached")
        await task.cancel()

    cap = asyncio.create_task(_cap())
    try:
        await PipelineRunner(handle_sigint=False).run(task)
    finally:
        cap.cancel()

    duration = time.time() - metrics.started_at
    used = tracker.used()
    logger.info(f"[voice:{case_id}] {duration:.0f}s, clauses used: {used}, "
                f"amounts heard: {heard_amounts}")

    return VoiceOutcome(
        happened=True,
        channel="webrtc",
        duration_s=duration,
        resolved=False,  # set by the agent's own wrap-up tool, or by review
        summary=f"WebRTC callback, {duration:.0f}s",
        citations_used=used,
    )


def _build_llm(system_prompt: str):
    """Gemini via Vertex when configured, else the Developer API.

    Vertex is preferred for production because it is where explicit context
    caching pays off: the system prompt here is
    the policy clause set, which is identical across every call in a category
    and therefore exactly the thing worth caching at a tenth of input price.
    """
    if os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true"):
        from pipecat.services.google.vertex.llm import GoogleVertexLLMService

        return GoogleVertexLLMService(
            credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            model=os.getenv("SANWAAD_VOICE_MODEL", "gemini-2.5-flash"),
            system_instruction=system_prompt,
            params=GoogleVertexLLMService.InputParams(
                location=os.getenv("GOOGLE_CLOUD_LOCATION", "asia-south1"),
                project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
                temperature=0.4,
                max_tokens=200,
            ),
        )

    from pipecat.services.google.llm import GoogleLLMService

    return GoogleLLMService(
        api_key=_require_env("GOOGLE_API_KEY"),
        model=os.getenv("SANWAAD_VOICE_MODEL", "gemini-2.5-flash"),
        system_instruction=system_prompt,
        params=GoogleLLMService.InputParams(temperature=0.4, max_tokens=200),
    )
