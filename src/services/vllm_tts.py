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

import httpx
from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from src.config import TtsConfig

CHUNK_DURATION_SECS = 0.020
SENTENCE_END_CHARS = ".!?;\n\u3002\uff01\uff1f\uff1b"


class VLLMTTSService(FrameProcessor):
    """TTS via vLLM-Omni's OpenAI-compatible speech endpoint."""

    def __init__(self, config: TtsConfig, **kwargs):
        super().__init__(**kwargs)
        self._config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._ref_audio_bytes: Optional[bytes] = None
        self._text_buffer = ""

    @property
    def _chunk_size_bytes(self) -> int:
        return int(self._config.sample_rate * 2 * CHUNK_DURATION_SECS)

    async def start(self, frame: StartFrame):
        if self._client:
            return

        self._client = httpx.AsyncClient(timeout=None)

        if self._config.ref_audio_path:
            ref_path = Path(self._config.ref_audio_path)
            if ref_path.exists():
                with open(ref_path, "rb") as f:
                    self._ref_audio_bytes = f.read()
                logger.info(f"Ref audio loaded: {ref_path}")
            else:
                logger.warning(f"Ref audio not found: {ref_path}")

        await self._pre_warm()
        logger.info(
            f"VLLMTTSService ready | {self._config.base_url} | model configured"
        )

    async def cleanup(self):
        if self._client:
            await self._client.close()
        self._client = None
        self._ref_audio_bytes = None
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
            await self._flush_text_buffer()
            await self.cleanup()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, CancelFrame):
            await self._flush_text_buffer()
            await self.cleanup()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseStartFrame):
            self._text_buffer = ""
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            await self._flush_text_buffer()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMTextFrame):
            self._text_buffer += frame.text
            for sentence in self._pop_complete_sentences():
                await self._synthesize(sentence)
            return

        await self.push_frame(frame, direction)

    def _pop_complete_sentences(self) -> list[str]:
        sentences = []
        start = 0
        for index, char in enumerate(self._text_buffer):
            if char in SENTENCE_END_CHARS:
                sentence = self._text_buffer[start:index + 1].strip()
                if sentence:
                    sentences.append(sentence)
                start = index + 1

        self._text_buffer = self._text_buffer[start:].lstrip()
        return sentences

    async def _flush_text_buffer(self):
        text = self._text_buffer.strip()
        self._text_buffer = ""
        if text:
            await self._synthesize(text)

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

        async with self._client.stream(
            "POST",
            self._speech_url,
            headers=self._headers,
            json=body,
        ) as response:
            response.raise_for_status()
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

        response = await self._client.post(
            self._speech_url,
            headers=self._headers,
            json=body,
        )
        response.raise_for_status()
        if not response.content:
            return

        for chunk in self._pcm_chunks_from_response(response.content):
            yield self._audio_frame(chunk, context_id=context_id)

    @property
    def _speech_url(self) -> str:
        base_url = str(self._config.base_url).rstrip("/")
        if base_url.endswith("/v1"):
            return base_url + "/audio/speech"
        return base_url + "/v1/audio/speech"

    @property
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json; charset=utf-8"}
        api_key = self._config.api_key.strip()
        if api_key and api_key != "not-needed":
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _build_body(self, text: str) -> dict:
        body: dict = {
            "model": self._config.model,
            "input": text,
            "voice": self._config.voice,
            "speed": self._config.speed,
        }
        if self._ref_audio_bytes:
            body["ref_audio"] = base64.b64encode(self._ref_audio_bytes).decode()
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
            response = await self._client.post(
                self._speech_url,
                headers=self._headers,
                json=self._build_body("Hello."),
            )
            response.raise_for_status()
            logger.info("TTS pre-warm complete")
        except Exception as exc:
            logger.warning(f"TTS pre-warm failed (non-fatal): {exc}")
