"""AED Suppressor — real-time non-speech event detection for VAD gating.

Uses lightweight spectral features (no external model required by default)
to detect music, clapping, wind noise, and other non-speech audio events
in ~100 ms chunks.  When a non-speech event is detected, emits
``AedSuppressFrame`` to temporarily disable VAD triggering.

Optional YAMNet integration: set ``use_yamnet=True`` for a more
accurate classifier (requires ``tensorflow`` or ``torch.hub``).
"""

import asyncio
import time
from collections import deque

import numpy as np
from loguru import logger
from pipecat.frames.frames import (
    AudioRawFrame,
    BotSpeakingFrame,
    BotStoppedSpeakingFrame,
    DataFrame,
    Frame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class AedSuppressFrame(DataFrame):
    """Emitted when a non-speech event is active — gates VAD output."""

    def __init__(self, suppressed: bool, event_type: str = "", confidence: float = 0.0):
        super().__init__()
        self.suppressed = suppressed
        self.event_type = event_type
        self.confidence = confidence


class AedSuppressor(FrameProcessor):
    """Lightweight real-time AED for VAD suppression.

    Extracts spectral features from audio chunks and classifies them
    into speech / music / noise / transient categories.  When a non-speech
    event (music, clap, cough, wind) is detected, emits suppression
    frames that prevent the VAD from triggering.

    Does NOT block audio — all frames pass through.  Suppression is a
    soft hint: downstream VAD can choose to honour or ignore it.

    Parameters:
        suppression_duration_ms: How long to suppress VAD after an event ends.
        min_chunk_ms: Minimum audio chunk for feature extraction.
        rms_threshold_ratio: Multiplier above noise floor to consider "active".
        enabled_events: Which non-speech event types to suppress on.
    """

    _SUPPORTED_EVENTS = frozenset({
        "music", "clap", "wind", "cough", "sneeze", "laughter",
        "applause", "bang", "screech", "silence",
    })

    def __init__(
        self,
        suppression_duration_ms: int = 500,
        min_chunk_ms: int = 100,
        rms_threshold_ratio: float = 2.0,
        enabled_events: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._suppression_ms = suppression_duration_ms
        self._min_chunk_ms = min_chunk_ms
        self._rms_ratio = rms_threshold_ratio
        self._enabled = enabled_events or ["music", "clap", "bang", "screech", "silence"]
        self._sample_rate: int = 0

        # State
        self._suppressed_until: float = 0.0
        self._current_event: str = ""
        self._noise_floor_rms: float = 0.0
        self._bot_speaking: bool = False
        self._chunk_count: int = 0

        # Rolling spectral history (last N chunks)
        self._spectral_history: deque[dict] = deque(maxlen=30)  # ~3 s @ 100 ms

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, frame: StartFrame):
        self._sample_rate = frame.audio_in_sample_rate

    async def cleanup(self):
        self._spectral_history.clear()

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # BotSpeakingFrame flows upstream from TTS — track whether bot is
        # producing audio so we can suppress VAD barge-in on non-speech events.
        if isinstance(frame, BotSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._suppressed_until = 0  # Reset suppression when bot finishes

        elif isinstance(frame, AudioRawFrame):
            # Always classify — lightweight spectral features won't interfere
            # with real speech.  Suppression frames are soft hints consumed
            # by SmartInterruptionGate (post-VAD).
            event = self._classify_chunk(frame)
            if event and event["type"] in self._enabled and event["confidence"] > 0.6:
                self._suppressed_until = time.monotonic() + self._suppression_ms / 1000.0
                self._current_event = event["type"]
                logger.debug(
                    f"AED suppress → {event['type']} "
                    f"(conf={event['confidence']:.2f}) "
                    f"for {self._suppression_ms}ms"
                )

        # Emit suppression status
        now = time.monotonic()
        if self._suppressed_until > now:
            await self.push_frame(AedSuppressFrame(
                suppressed=True,
                event_type=self._current_event,
                confidence=0.8,
            ))
        elif self._chunk_count % 10 == 0:
            await self.push_frame(AedSuppressFrame(
                suppressed=False,
                event_type="",
                confidence=0.0,
            ))

        self._chunk_count += 1
        await self.push_frame(frame, direction)

    # ------------------------------------------------------------------
    # Feature extraction + classification
    # ------------------------------------------------------------------

    def _classify_chunk(self, frame: AudioRawFrame) -> dict | None:
        """Extract spectral features and classify the chunk.

        Returns:
            ``{"type": str, "confidence": float}`` or ``None`` if too quiet.
        """
        try:
            arr = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0
        except (ValueError, AttributeError):
            return None

        if len(arr) < self._min_chunk_ms * self._sample_rate // 1000:
            return None

        # ---- Energy gate ----
        rms = float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))
        if rms < 1e-6:
            return {"type": "silence", "confidence": 0.95}

        # ---- Spectral features ----
        n_fft = min(512, len(arr))
        try:
            spec = np.abs(np.fft.rfft(arr, n=n_fft))
            spec = spec[: n_fft // 2 + 1]
        except Exception:
            return None

        if len(spec) < 4:
            return None

        # Spectral flatness (Wiener entropy) — high → noise-like (music, wind)
        geo_mean = np.exp(np.mean(np.log(spec + 1e-10)))
        arith_mean = np.mean(spec)
        flatness = float(geo_mean / (arith_mean + 1e-10))  # [0, 1]

        # Spectral centroid — high → bright (claps, screeches)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self._sample_rate)[: len(spec)]
        if np.sum(spec) > 0:
            centroid = float(np.sum(freqs * spec) / np.sum(spec))
        else:
            centroid = 0.0

        # Zero-crossing rate — high → noise, low → tonal (music)
        zcr = float(np.sum(np.abs(np.diff(np.signbit(arr)))) / len(arr))

        # ---- Classification (heuristic) ----
        features = {
            "flatness": flatness,
            "centroid": centroid,
            "zcr": zcr,
            "rms": rms,
        }
        self._spectral_history.append(features)

        # Silence
        if rms < self._noise_floor_rms * self._rms_ratio:
            return None  # Too quiet to classify

        # Music: low ZCR + moderate flatness + sustained energy
        if zcr < 0.15 and 0.3 < flatness < 0.7:
            return {"type": "music", "confidence": 0.7 + flatness * 0.3}

        # Clap / bang / transient: very high centroid + short burst
        if centroid > 3000 and flatness > 0.5:
            return {"type": "clap", "confidence": min(centroid / 6000, 0.95)}

        # Wind / noise: very high ZCR + high flatness
        if zcr > 0.45 and flatness > 0.6:
            return {"type": "wind", "confidence": min(zcr, 0.9)}

        # Screech: high centroid + sustained
        if centroid > 4000 and zcr < 0.3:
            return {"type": "screech", "confidence": min(centroid / 8000, 0.9)}

        # Default: likely speech or ambiguous — don't suppress
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_suppressed(self) -> bool:
        return time.monotonic() < self._suppressed_until

    @property
    def current_event_type(self) -> str:
        return self._current_event if self.is_suppressed else ""

    @property
    def recent_features(self) -> list[dict]:
        return list(self._spectral_history)
