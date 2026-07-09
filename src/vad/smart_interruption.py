"""Smart interruption gate — multi-signal VAD event filter (post-STT).

Sits between STT and ContentConcatenator.  When the bot is speaking and the
user tries to barge in, this gate holds VAD frames and transcriptions until
enough words have been transcribed (or a timeout fires).  This prevents
coughs, door slams, and single-word interjections from interrupting the bot.

Once the word-count threshold is met, an ``InterruptionFrame`` is issued
followed by the buffered VAD + transcription frames, so downstream
processors see a clean turn start.

Pipeline position::

    VAD → STT → SmartInterruptionGate → ContentConcatenator → LLM → TTS

Requires ``allow_interruptions=False`` on the ``PipelineParams`` — this
gate is the sole source of ``InterruptionFrame``.
"""

import time as _time

from loguru import logger
from pipecat.frames.frames import (
    BotSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterruptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from src.vad.aed_suppressor import AedSuppressFrame


class SmartInterruptionGate(FrameProcessor):
    """Gate VAD events through min-words + AED-suppression + bot-state.

    Parameters:
        min_words: Minimum transcribed words to allow interruption.
        max_wait_secs: Maximum wait for min_words before forwarding anyway.
        aed_suppression: Whether to gate on AED non-speech events.
    """

    def __init__(
        self,
        min_words: int = 2,
        max_wait_secs: float = 2.0,
        aed_suppression: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._min_words = min_words
        self._max_wait = max_wait_secs
        self._aed_suppression = aed_suppression

        # Bot state (BotSpeakingFrame flows upstream from TTS)
        self._bot_speaking: bool = False

        # Per-barge-in gating state
        self._gating: bool = False
        self._word_count: int = 0
        self._wait_start: float = 0.0
        self._aed_suppressed: bool = False

        # Buffered frames held during gating
        self._held_start: VADUserStartedSpeakingFrame | None = None
        self._held_transcriptions: list[TranscriptionFrame] = []
        self._held_stop: VADUserStoppedSpeakingFrame | None = None

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        now = _time.monotonic()

        # ---- Bot speaking state (upstream frames from TTS) ----
        if isinstance(frame, BotSpeakingFrame):
            self._bot_speaking = True
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._reset_gating()
            await self.push_frame(frame, direction)
            return

        # ---- AED suppression ----
        if isinstance(frame, AedSuppressFrame):
            self._aed_suppressed = frame.suppressed and self._aed_suppression
            if self._aed_suppressed and self._gating:
                logger.debug(
                    f"SmartInterruption: AED suppressed ({frame.event_type}) — "
                    f"cancelling barge-in"
                )
                self._reset_gating()
            await self.push_frame(frame, direction)
            return

        # ---- Explicit interruption — reset gating ----
        if isinstance(frame, InterruptionFrame):
            self._reset_gating()
            await self.push_frame(frame, direction)
            return

        # ---- VAD: user started speaking ----
        if isinstance(frame, VADUserStartedSpeakingFrame):
            if self._bot_speaking and self._min_words > 0 and not self._aed_suppressed:
                # Barge-in: hold the frame, wait for enough words
                self._start_gating(frame)
                logger.debug(
                    f"SmartInterruption: gating barge-in "
                    f"(need {self._min_words} words, timeout {self._max_wait}s)"
                )
                # Do NOT forward — STT already received it (we're post-STT)
                return
            else:
                # New turn or gating disabled — forward immediately
                self._reset_gating()
                await self.push_frame(frame, direction)
                return

        # ---- Transcription: count words while gating ----
        if isinstance(frame, TranscriptionFrame) and self._gating:
            if frame.text:
                self._held_transcriptions.append(frame)
                self._word_count += len(frame.text.strip().split())
                logger.debug(
                    f"SmartInterruption: {self._word_count}/{self._min_words} words"
                )

            # Release on word-count threshold
            if self._word_count >= self._min_words:
                logger.info(
                    f"SmartInterruption: releasing barge-in "
                    f"({self._word_count} words)"
                )
                await self._release()
                return

            # Release on timeout
            if now - self._wait_start > self._max_wait:
                logger.debug(
                    f"SmartInterruption: timeout — releasing barge-in "
                    f"({self._word_count} words)"
                )
                await self._release()
                return

            # Still gating — don't forward yet
            return

        # ---- VAD: user stopped while gating ----
        if isinstance(frame, VADUserStoppedSpeakingFrame) and self._gating:
            # STT has finished transcribing.  If we have enough words by now,
            # release; otherwise discard (not enough words).
            if self._word_count >= self._min_words:
                logger.info(
                    f"SmartInterruption: releasing on VAD stop "
                    f"({self._word_count} words)"
                )
                await self._release()
            else:
                logger.debug(
                    f"SmartInterruption: discarding barge-in — "
                    f"only {self._word_count}/{self._min_words} words"
                )
                self._reset_gating()
            return

        # ---- Everything else — pass through ----
        await self.push_frame(frame, direction)

    # ------------------------------------------------------------------
    # Gating state machine
    # ------------------------------------------------------------------

    def _start_gating(self, start_frame: VADUserStartedSpeakingFrame) -> None:
        """Begin a gating interval."""
        self._gating = True
        self._word_count = 0
        self._wait_start = _time.monotonic()
        self._held_start = start_frame
        self._held_transcriptions = []
        self._held_stop = None

    async def _release(self) -> None:
        """Release all buffered frames: interruption → VAD start → transcriptions."""
        # 1. Interrupt the bot first
        await self.push_frame(InterruptionFrame())

        # 2. VAD start — signals new turn to ContentConcatenator
        if self._held_start is not None:
            await self.push_frame(self._held_start)

        # 3. Accumulated transcription(s)
        for tf in self._held_transcriptions:
            await self.push_frame(tf)

        # 4. VAD stop (if we have it yet)
        if self._held_stop is not None:
            await self.push_frame(self._held_stop)

        self._reset_gating()

    def _reset_gating(self) -> None:
        """Clear all gating state."""
        self._gating = False
        self._word_count = 0
        self._held_start = None
        self._held_transcriptions = []
        self._held_stop = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def is_gating(self) -> bool:
        return self._gating
