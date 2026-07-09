"""FsmnVadGate — first-stage VAD pre-filter using FunASR fsmn-vad.

Fast, lightweight speech/non-speech classifier that runs on small audio
chunks (~100 ms).  Acts as a gate: marks ``AudioRawFrame`` with a
``speech_likelihood`` attribute so downstream VAD can skip clearly
non-speech frames.

Reduces false triggers from keyboard clicks, door slams, etc.
"""

from loguru import logger
from pipecat.frames.frames import (
    AudioRawFrame,
    DataFrame,
    Frame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from src.vad.adaptive_noise import NoiseFloorFrame


class SpeechGateFrame(DataFrame):
    """Attached to AudioRawFrame metadata by FsmnVadGate."""

    def __init__(self, is_speech: bool, score: float):
        super().__init__()
        self.is_speech = is_speech
        self.score = score  # 0.0 (noise) … 1.0 (certain speech)


class FsmnVadGate(FrameProcessor):
    """First-stage VAD gate powered by FunASR fsmn-vad.

    Runs on ~100 ms audio chunks.  If fsmn-vad says "no speech", the
    chunk is still passed through but tagged with ``is_speech=False``
    so the Silero VAD stage can skip it.

    Parameters:
        min_speech_score: Score threshold for "likely speech" (0–1).
        chunk_samples: Samples per inference chunk (at 16 kHz).
        device: ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        min_speech_score: float = 0.5,
        chunk_samples: int = 1600,  # 100 ms @ 16 kHz
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._min_score = min_speech_score
        self._chunk_samples = chunk_samples
        self._device = device
        self._model = None
        self._sample_rate: int = 0
        # Rolling flag: how many consecutive chunks were non-speech
        self._consecutive_non_speech: int = 0
        # Noise-floor adaptive threshold (populated by NoiseFloorFrame)
        self._noise_floor_rms: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, frame: StartFrame):
        self._sample_rate = frame.audio_in_sample_rate
        await self._load_model()

    async def cleanup(self):
        self._model = None

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            self._sample_rate = frame.audio_in_sample_rate
            if self._model is None:
                await self._load_model()

        # Track noise floor
        if isinstance(frame, NoiseFloorFrame):
            self._noise_floor_rms = frame.rms_noise_floor

        elif isinstance(frame, AudioRawFrame):
            is_speech = await self._classify_chunk(frame)
            # Tag the frame: downstream VAD can read this metadata
            await self.push_frame(SpeechGateFrame(is_speech=is_speech, score=1.0 if is_speech else 0.0))

        await self.push_frame(frame, direction)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _classify_chunk(self, frame: AudioRawFrame) -> bool:
        """Classify one chunk — speech or not."""
        if self._model is None:
            return True  # No model → let everything through

        try:
            import numpy as np
            arr = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0
        except (ValueError, AttributeError):
            return True

        # Adaptive energy gate: if very quiet relative to noise floor, skip inference
        rms = float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))
        if self._noise_floor_rms > 0 and rms < self._noise_floor_rms * 1.5:
            self._consecutive_non_speech += 1
            return False

        # Too short
        if len(arr) < 160:  # < 10 ms
            return False

        try:
            import asyncio
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._model.generate(input=arr, disable_pbar=True),
            )
            if result and len(result) > 0 and "text" in result[0]:
                # fsmn-vad returns speech segments; empty means no speech detected
                text = result[0]["text"]
                is_speech = bool(text and text.strip())
                if is_speech:
                    self._consecutive_non_speech = 0
                else:
                    self._consecutive_non_speech += 1
                return is_speech
        except Exception as exc:
            logger.debug(f"fsmn-vad inference error: {exc}")

        return True  # On error, let through

    async def _load_model(self) -> None:
        try:
            import asyncio

            from funasr import AutoModel

            loop = asyncio.get_running_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: AutoModel(
                    model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                    device=self._device,
                    disable_pbar=True,
                    disable_progress_bar=True,
                    disable_log=True,
                ),
            )
            logger.info(f"FsmnVadGate loaded | device={self._device}")
        except Exception as exc:
            self._model = None
            logger.warning(f"FsmnVadGate disabled; model load failed: {exc}")

    @property
    def is_in_non_speech_burst(self) -> bool:
        """True when several consecutive chunks have been non-speech."""
        return self._consecutive_non_speech >= 5  # ~500 ms
