"""vLLM-Omni TTS service for Pipecat 1.4+.

This service calls an OpenAI-compatible ``/v1/audio/speech`` endpoint and
emits Pipecat ``TTSAudioRawFrame`` objects. It intentionally avoids Pipecat's
``TTSService`` base class because that import path requires NLTK sentence data
at import time in Pipecat 1.4.0, which makes startup fragile when NLTK data is
missing or corrupt.
"""

from __future__ import annotations

import asyncio
import base64
import io
import wave
from collections.abc import AsyncGenerator, Iterable
from pathlib import Path
from typing import Optional

from loguru import logger
from openai import AsyncOpenAI
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from src.config import TtsConfig

CHUNK_DURATION_SECS = 0.020


class VLLMTTSService(FrameProcessor):
    """TTS via vLLM-Omni's OpenAI-compatible speech endpoint."""

    def __init__(self, config: TtsConfig, **kwargs):
        super().__init__(**kwargs)
        self._config = config
        self._client: Optional[AsyncOpenAI] = None
        self._ref_audio_base64: Optional[str] = None

    @property
    def _chunk_size_bytes(self) -> int:
        return int(self._config.sample_rate * 2 * CHUNK_DURATION_SECS)

    async def start(self, frame: StartFrame):
        if self._client:
            return

        self._client = AsyncOpenAI(
            base_url=str(self._config.base_url),
            api_key=self._config.api_key,
        )

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

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        async for frame in self._generate_audio_frames(text, context_id=context_id):
            yield frame

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            await self.start(frame)
            return

        if isinstance(frame, EndFrame):
            await self.cleanup()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, CancelFrame):
            await self.cleanup()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TextFrame):
            text = frame.text.strip()
            if text:
                await self._synthesize(text)
            return

        await self.push_frame(frame, direction)

    async def _synthesize(self, text: str):
        if not self._client:
            logger.warning("TTS client is not initialized; dropping text")
            return

        total_bytes = 0
        async for frame in self._generate_audio_frames(text, context_id=None):
            if isinstance(frame, TTSAudioRawFrame):
                total_bytes += len(frame.audio)
            await self.push_frame(frame)

        duration = total_bytes / (self._config.sample_rate * 2)
        logger.debug(f"TTS -> {duration:.1f}s ({total_bytes}B) \"{text[:50]}\"")

    async def _generate_audio_frames(
        self,
        text: str,
        *,
        context_id: str | None,
    ) -> AsyncGenerator[TTSAudioRawFrame, None]:
        body = self._build_body(text)

        try:
            async for frame in self._stream_audio_frames(body, context_id=context_id):
                yield frame
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"Streaming TTS failed ({exc}), falling back to batch")

        try:
            async for frame in self._batch_audio_frames(body, context_id=context_id):
                yield frame
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(f"TTS failed (batch): {exc}")

    async def _stream_audio_frames(
        self,
        body: dict,
        *,
        context_id: str | None,
    ) -> AsyncGenerator[TTSAudioRawFrame, None]:
        assert self._client is not None

        async with self._client.audio.speech.with_streaming_response.create(**body) as response:
            if self._config.response_format.lower() == "wav":
                wav_bytes = bytearray()
                async for chunk in response.iter_bytes(self._chunk_size_bytes):
                    if chunk:
                        wav_bytes.extend(chunk)
                for pcm_chunk in self._pcm_chunks_from_response(bytes(wav_bytes)):
                    yield self._audio_frame(pcm_chunk, context_id=context_id)
                return

            async for chunk in response.iter_bytes(self._chunk_size_bytes):
                if chunk:
                    yield self._audio_frame(chunk, context_id=context_id)

    async def _batch_audio_frames(
        self,
        body: dict,
        *,
        context_id: str | None,
    ) -> AsyncGenerator[TTSAudioRawFrame, None]:
        assert self._client is not None

        response = await self._client.audio.speech.create(**body)
        if not response.content:
            return

        for chunk in self._pcm_chunks_from_response(response.content):
            yield self._audio_frame(chunk, context_id=context_id)

    def _build_body(self, text: str) -> dict:
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
        return body

    def _audio_frame(self, audio: bytes, *, context_id: str | None) -> TTSAudioRawFrame:
        return TTSAudioRawFrame(
            audio=audio,
            sample_rate=self._config.sample_rate,
            num_channels=1,
            context_id=context_id,
        )

    def _response_to_pcm(self, content: bytes) -> bytes:
        fmt = self._config.response_format.lower()
        if fmt != "wav" and not content.startswith(b"RIFF"):
            return content

        try:
            with wave.open(io.BytesIO(content), "rb") as wf:
                return wf.readframes(wf.getnframes())
        except Exception as exc:
            logger.error(f"TTS WAV decode failed: {exc}")
            return b""

    def _pcm_chunks_from_response(self, content: bytes) -> Iterable[bytes]:
        return self._chunks(self._response_to_pcm(content))

    def _chunks(self, pcm: bytes) -> Iterable[bytes]:
        for i in range(0, len(pcm), self._chunk_size_bytes):
            chunk = pcm[i : i + self._chunk_size_bytes]
            if chunk:
                yield chunk

    async def _pre_warm(self):
        if not self._client:
            return

        try:
            await self._client.audio.speech.create(**self._build_body(" "))
            logger.info("TTS pre-warm complete")
        except Exception as exc:
            logger.warning(f"TTS pre-warm failed (non-fatal): {exc}")
