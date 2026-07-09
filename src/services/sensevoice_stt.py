"""SenseVoiceSmall STT service for Pipecat 1.4+ — with interim results.

Extends ``SegmentedSTTService`` which handles audio buffering and VAD
segmentation automatically.  We only need to override ``run_stt()`` to
run FunASR inference and yield transcription + AED event frames.

Optional interim results: while the user is speaking, the service
periodically (every ``interim_interval_ms``) runs ASR on the current
audio buffer and emits ``InterimTranscriptionFrame`` for real-time
display without affecting the final LLM context.
"""

import asyncio
import re
import time
from typing import AsyncGenerator, Optional

from loguru import logger
from pipecat.frames.frames import (
    DataFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.services.stt_service import STTSettings, SegmentedSTTService

from src.config import AsrConfig


# ---------------------------------------------------------------------------
# SenseVoice rich-output parser
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<\|([^|>]+)\|>")
_AED_EVENTS = frozenset({"BGM", "Applause", "Laughter", "Coughing", "Sneeze", "Crying"})
_LANG_CODES = frozenset({"zh", "en", "ja", "ko", "yue"})
_EMOTIONS = frozenset({"HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEAR", "SURPRISE"})

# Minimum audio duration for interim ASR (0.5s @ 16kHz mono 16-bit)
_MIN_INTERIM_AUDIO_BYTES = 16000  # 0.5s * 16000 * 2


def parse_sensevoice_output(raw_text: str) -> dict:
    """Parse SenseVoiceSmall rich output into structured fields."""
    tags = _TAG_RE.findall(raw_text)
    language: Optional[str] = None
    emotion: Optional[str] = None
    aed_events: list[str] = []
    for tag in tags:
        if tag in _LANG_CODES:
            language = tag
        elif tag.upper() in _EMOTIONS:
            emotion = tag.upper()
        elif tag in _AED_EVENTS:
            aed_events.append(tag)
    clean_text = _TAG_RE.sub("", raw_text).strip()
    return {
        "text": clean_text,
        "language": language,
        "emotion": emotion,
        "aed_events": aed_events,
    }


# ---------------------------------------------------------------------------
# Custom Frame
# ---------------------------------------------------------------------------


class EnvironmentalSoundFrame(DataFrame):
    """Emitted for each AED event detected alongside speech."""

    def __init__(self, sound_type: str, timestamp_ns: int, confidence: float = 1.0):
        super().__init__()
        self.sound_type = sound_type
        self.confidence = confidence


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class SenseVoiceSTTService(SegmentedSTTService):
    """Segmented ASR backed by FunASR SenseVoiceSmall.

    ``SegmentedSTTService`` handles VAD-boundary detection, audio buffering,
    and WAV encoding.  We override ``run_stt()`` to call FunASR and yield
    ``TranscriptionFrame`` + ``EnvironmentalSoundFrame`` objects.

    When ``enable_interim`` is True, periodic ASR snapshots are taken during
    speech to emit ``InterimTranscriptionFrame`` for real-time display.
    """

    def __init__(self, config: AsrConfig, **kwargs):
        super().__init__(
            sample_rate=config.sample_rate,
            settings=STTSettings(model=config.model, language=None),
            ttfs_p99_latency=1.0,
            **kwargs,
        )
        self._config = config
        self._model = None

        # Interim results state
        self._interim_task: Optional[asyncio.Task] = None
        self._last_interim_text: str = ""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, frame):
        from funasr import AutoModel

        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(
            None,
            lambda: AutoModel(
                model=self._config.model,
                vad_model=self._config.vad_model,
                device=self._config.device,
                disable_progress_bar=True,
                disable_log=True,
            ),
        )
        logger.info(
            f"SenseVoiceSmall loaded | device={self._config.device} "
            f"model={self._config.model}"
        )
        if self._config.enable_interim:
            logger.info(
                f"SenseVoiceSmall interim results enabled | "
                f"interval={self._config.interim_interval_ms}ms"
            )
        await super().start(frame)

    async def cleanup(self):
        await self._cancel_interim_task()
        self._model = None
        await super().cleanup()

    # ------------------------------------------------------------------
    # VAD event handlers — extended for interim results
    # ------------------------------------------------------------------

    async def _handle_user_started_speaking(self, frame: VADUserStartedSpeakingFrame):
        await super()._handle_user_started_speaking(frame)
        self._last_interim_text = ""
        if self._config.enable_interim:
            self._interim_task = asyncio.create_task(self._run_interim_loop())

    async def _handle_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        await self._cancel_interim_task()
        await super()._handle_user_stopped_speaking(frame)

    # ------------------------------------------------------------------
    # Core — called by SegmentedSTTService with complete WAV bytes
    # ------------------------------------------------------------------

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe one complete speech segment."""
        if self._model is None:
            logger.warning("ASR model not loaded")
            return

        import io
        import wave

        import numpy as np

        # Decode WAV bytes → float32 numpy array
        try:
            with wave.open(io.BytesIO(audio), "rb") as wf:
                pcm = wf.readframes(wf.getnframes())
                arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception as exc:
            logger.error(f"Failed to decode audio for ASR: {exc}")
            return

        if len(arr) < 160:  # < 10ms — too short
            return

        # Run FunASR
        result = await self._transcribe_array(arr)
        if result is None:
            return

        raw_text = result[0].get("text", "")
        if not raw_text:
            return

        parsed = parse_sensevoice_output(raw_text)

        logger.info(
            f"ASR → \"{parsed['text'][:80]}\" "
            f"| lang={parsed['language']} "
            f"| emotion={parsed['emotion']} "
            f"| events={parsed['aed_events']}"
        )

        # Emit transcription
        if parsed["text"]:
            yield TranscriptionFrame(
                text=parsed["text"],
                user_id="user",
                timestamp=str(time.time_ns()),
            )

        # Emit AED events
        for event_type in parsed["aed_events"]:
            yield EnvironmentalSoundFrame(
                sound_type=event_type,
                timestamp_ns=time.time_ns(),
            )

    # ------------------------------------------------------------------
    # Interim results loop
    # ------------------------------------------------------------------

    async def _run_interim_loop(self):
        """Periodically transcribe buffered audio for real-time display."""
        interval = self._config.interim_interval_ms / 1000.0
        try:
            while True:
                await asyncio.sleep(interval)
                if len(self._audio_buffer) < _MIN_INTERIM_AUDIO_BYTES:
                    continue

                # Build WAV from current buffer
                result = await self._transcribe_buffer(self._audio_buffer)
                if result is None:
                    continue

                raw_text = result[0].get("text", "")
                if not raw_text:
                    continue

                parsed = parse_sensevoice_output(raw_text)
                text = parsed["text"]
                if not text or text == self._last_interim_text:
                    continue

                self._last_interim_text = text
                await self.push_frame(
                    InterimTranscriptionFrame(
                        text=text,
                        user_id="user",
                        timestamp=str(time.time_ns()),
                    )
                )
                logger.debug(f"ASR(interim) → \"{text[:60]}\"")
        except asyncio.CancelledError:
            pass

    async def _cancel_interim_task(self):
        if self._interim_task:
            self._interim_task.cancel()
            try:
                await self._interim_task
            except asyncio.CancelledError:
                pass
            self._interim_task = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _transcribe_array(self, arr) -> Optional[list]:
        """Run FunASR on a numpy float32 array (CPU-bound, runs in executor)."""
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                lambda: self._model.generate(
                    input=arr,
                    language="auto",
                    use_itn=self._config.use_itn,
                ),
            )
        except Exception as exc:
            logger.error(f"FunASR inference failed: {exc}")
            return None

    async def _transcribe_buffer(self, buffer: bytearray) -> Optional[list]:
        """Run FunASR on a raw PCM bytearray (WAV encoding + inference)."""
        import io
        import wave

        import numpy as np

        try:
            content = io.BytesIO()
            with wave.open(content, "wb") as wf:
                wf.setsampwidth(2)
                wf.setnchannels(1)
                wf.setframerate(self.sample_rate)
                wf.writeframes(bytes(buffer))
            content.seek(0)

            with wave.open(content, "rb") as wf:
                pcm = wf.readframes(wf.getnframes())
                arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

            if len(arr) < 160:
                return None

            return await self._transcribe_array(arr)
        except Exception as exc:
            logger.error(f"Interim ASR failed: {exc}")
            return None
