"""vLLM-Omni TTS service for Pipecat 1.4+ — streaming edition.

Integrates vLLM-Omni's OpenAI-compatible ``/v1/audio/speech`` endpoint with
streaming response support for lower time-to-first-audio latency.

Usage (server)::

    vllm serve fishaudio/s2-pro --omni --port 8091
"""

import asyncio
import base64
import io
import wave
from pathlib import Path
from typing import Optional

from loguru import logger
from openai import AsyncOpenAI
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    StartFrame,
    TextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import TTSService

from src.config import TtsConfig

# 20ms audio chunks
CHUNK_SIZE_BYTES = 960  # 24000 Hz * 2 bytes * 0.020s


class VLLMTTSService(TTSService):
    """TTS via vLLM-Omni (OpenAI-compatible speech endpoint) with streaming.

    Uses ``AsyncOpenAI`` + ``with_streaming_response.create()`` for chunked
    audio delivery.  Falls back to batch mode if streaming is not supported
    by the endpoint.
    """

    def __init__(self, config: TtsConfig, **kwargs):
        super().__init__(
            sample_rate=config.sample_rate,
            text_aggregation_mode="sentence",
            **kwargs,
        )
        self._config = config
        self._client: Optional[AsyncOpenAI] = None
        self._ref_audio_base64: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, frame: StartFrame):
        self._client = AsyncOpenAI(
            base_url=str(self._config.base_url),
            api_key=self._config.api_key,
        )
        # Load reference audio
        if self._config.ref_audio_path:
            ref_path = Path(self._config.ref_audio_path)
            if ref_path.exists():
                with open(ref_path, "rb") as f:
                    self._ref_audio_base64 = base64.b64encode(f.read()).decode()
                logger.info(f"Ref audio loaded: {ref_path}")
            else:
                logger.warning(f"Ref audio not found: {ref_path}")
        await self._pre_warm()
        logger.info(
            f"VLLMTTSService ready | {self._config.base_url} | {self._config.model}"
        )

    async def cleanup(self):
        if self._client:
            await self._client.close()
        self._client = None
        self._ref_audio_base64 = None
        await super().cleanup()

    # ------------------------------------------------------------------
    # Core — called for each aggregated text segment
    # ------------------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Intercept TextFrame, synthesize, push AudioRawFrames."""
        if isinstance(frame, TextFrame):
            text = frame.text.strip()
            if text:
                await self._synthesize(text)
            return  # Don't push the text frame downstream
        await self.push_frame(frame, direction)

    async def _synthesize(self, text: str):
        """Call vLLM-Omni and push audio chunks downstream (streaming if possible)."""
        body: dict = {
            "model": self._config.model,
            "input": text,
            "voice": self._config.voice,
            "speed": self._config.speed,
            "response_format": self._config.response_format,
        }
        if self._ref_audio_base64:
            body["ref_audio"] = self._ref_audio_base64
        if self._config.ref_text:
            body["ref_text"] = self._config.ref_text

        try:
            await self._synthesize_streaming(body, text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"Streaming TTS failed ({exc}), falling back to batch")
            try:
                await self._synthesize_batch(body, text)
            except asyncio.CancelledError:
                raise
            except Exception as exc2:
                logger.error(f"TTS failed (batch): {exc2}")

    # ------------------------------------------------------------------
    # Streaming path
    # ------------------------------------------------------------------

    async def _synthesize_streaming(self, body: dict, text: str):
        """Stream audio chunks from vLLM-Omni via ``with_streaming_response``."""
        total_bytes = 0

        async with self._client.audio.speech.with_streaming_response.create(
            **body
        ) as response:
            async for chunk_bytes in response.iter_bytes(CHUNK_SIZE_BYTES):
                if chunk_bytes:
                    await self.push_frame(
                        AudioRawFrame(
                            audio=chunk_bytes,
                            sample_rate=self._config.sample_rate,
                            num_channels=1,
                        )
                    )
                    total_bytes += len(chunk_bytes)

        duration = total_bytes / (self._config.sample_rate * 2)  # 16-bit mono
        logger.debug(
            f"TTS(stream) → {duration:.1f}s ({total_bytes}B) \"{text[:50]}\""
        )

    # ------------------------------------------------------------------
    # Batch fallback (WAV decode with early chunking)
    # ------------------------------------------------------------------

    async def _synthesize_batch(self, body: dict, text: str):
        """Batch TTS with WAV decoding and immediate chunk output."""
        resp = await self._client.audio.speech.create(**body)

        if not resp.content:
            return

        # Decode WAV → int16 PCM → AudioRawFrame chunks
        try:
            with wave.open(io.BytesIO(resp.content), "rb") as wf:
                pcm = wf.readframes(wf.getnframes())
        except Exception as exc:
            logger.error(f"TTS WAV decode failed: {exc}")
            return

        # Push 20ms chunks
        for i in range(0, len(pcm), CHUNK_SIZE_BYTES):
            chunk = pcm[i : i + CHUNK_SIZE_BYTES]
            if chunk:
                await self.push_frame(
                    AudioRawFrame(
                        audio=chunk,
                        sample_rate=self._config.sample_rate,
                        num_channels=1,
                    )
                )
                if (i // CHUNK_SIZE_BYTES) % 10 == 0:
                    await asyncio.sleep(0)

        duration = len(pcm) / (self._config.sample_rate * 2)
        logger.debug(
            f"TTS(batch) → {duration:.1f}s \"{text[:50]}\""
        )

    # ------------------------------------------------------------------
    # Pre-warm
    # ------------------------------------------------------------------

    async def _pre_warm(self):
        """Hide first-inference latency."""
        try:
            body: dict = {
                "model": self._config.model,
                "input": " ",
                "voice": self._config.voice,
                "speed": self._config.speed,
                "response_format": self._config.response_format,
            }
            if self._ref_audio_base64:
                body["ref_audio"] = self._ref_audio_base64
            if self._config.ref_text:
                body["ref_text"] = self._config.ref_text
            resp = await self._client.audio.speech.create(**body)
            logger.info("TTS pre-warm complete")
        except Exception as exc:
            logger.warning(f"TTS pre-warm failed (non-fatal): {exc}")
