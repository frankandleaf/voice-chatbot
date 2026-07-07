"""Adaptive noise-floor tracker with rolling RMS window.

Maintains a 30-second rolling buffer of audio energy, computes the
85th-percentile noise floor, and broadcasts ``NoiseFloorFrame``
periodically so other processors can adapt their thresholds.
"""

import asyncio
import time
from collections import deque

import numpy as np
from loguru import logger
from pipecat.frames.frames import (
    AudioRawFrame,
    DataFrame,
    Frame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class NoiseFloorFrame(DataFrame):
    """Broadcast periodically with the current noise-floor estimate."""

    def __init__(self, rms_noise_floor: float, rms_current: float, db_snr: float):
        super().__init__()
        self.rms_noise_floor = rms_noise_floor
        self.rms_current = rms_current
        self.db_snr = db_snr


class AdaptiveNoiseTracker(FrameProcessor):
    """Track ambient noise level with a rolling RMS window.

    Passes audio through unchanged while maintaining a noise-floor
    estimate.  Emits ``NoiseFloorFrame`` downstream every second so
    the VAD gate can adapt its thresholds.

    Parameters:
        window_secs: Rolling window duration (default 30 s).
        percentile: Percentile for noise-floor calculation (default 85).
        update_interval_s: How often to emit ``NoiseFloorFrame``.
    """

    def __init__(
        self,
        window_secs: float = 30.0,
        percentile: float = 85.0,
        update_interval_s: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._window_secs = window_secs
        self._percentile = percentile
        self._update_interval_s = update_interval_s
        self._sample_rate: int = 0
        self._rms_buffer: deque[float] = deque()
        self._max_samples: int = 0
        self._last_update: float = 0
        # Warm-up: track 85th percentile of ALL samples until window fills
        self._all_rms: list[float] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, frame: StartFrame):
        self._sample_rate = frame.audio_in_sample_rate
        # Each RMS measurement covers ~100 ms
        self._max_samples = int(self._window_secs / 0.1)
        await super().start(frame)

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, AudioRawFrame):
            rms = self._compute_rms(frame)
            if rms is not None:
                self._rms_buffer.append(rms)
                self._all_rms.append(rms)
                if len(self._rms_buffer) > self._max_samples:
                    self._rms_buffer.popleft()
                if len(self._all_rms) > self._max_samples * 3:
                    self._all_rms = self._all_rms[-self._max_samples:]
            await self._maybe_emit_noise_floor()

        await self.push_frame(frame, direction)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_rms(self, frame: AudioRawFrame) -> float | None:
        try:
            arr = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32)
            if len(arr) == 0:
                return None
            return float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))
        except (ValueError, AttributeError):
            return None

    async def _maybe_emit_noise_floor(self):
        now = time.monotonic()
        if now - self._last_update < self._update_interval_s:
            return
        self._last_update = now

        if len(self._rms_buffer) < 5:
            return  # Not enough data yet

        values = np.array(list(self._rms_buffer), dtype=np.float64)
        noise_floor = float(np.percentile(values, self._percentile))
        current = values[-1]

        db_snr = 0.0
        if noise_floor > 0:
            ratio = current / noise_floor
            if ratio > 0:
                db_snr = float(20.0 * np.log10(ratio))

        await self.push_frame(NoiseFloorFrame(
            rms_noise_floor=noise_floor,
            rms_current=current,
            db_snr=db_snr,
        ))

    @property
    def noise_floor_rms(self) -> float:
        """Current noise-floor RMS estimate."""
        if len(self._rms_buffer) < 5:
            return 0.0
        return float(np.percentile(list(self._rms_buffer), self._percentile))

    @property
    def current_rms(self) -> float:
        return self._rms_buffer[-1] if self._rms_buffer else 0.0

    @property
    def snr_db(self) -> float:
        nf = self.noise_floor_rms
        cur = self.current_rms
        if nf > 0 and cur > 0:
            return float(20.0 * np.log10(cur / nf))
        return 0.0
