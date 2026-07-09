"""Content concatenation + conversation history for Pipecat 1.4+.

Collects ``TranscriptionFrame`` results across a turn, concatenates on
``VADUserStoppedSpeakingFrame``, and emits ``LLMContextFrame`` for the LLM.

Supports multimodal content via ``InputImageRawFrame`` — when an image is
present, the user message is built as an OpenAI multimodal content array
with both text and image_url parts.
"""

from typing import Optional

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InputImageRawFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMContextFrame,
    TextFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from src.config import AedConfig, LlmConfig
from src.services.sensevoice_stt import EnvironmentalSoundFrame


class ContentConcatenator(FrameProcessor):
    """Aggregate partial transcripts → complete utterance → LLM context.

    Sits between ASR and LLM.  On end-of-speech it builds the full user
    message (optionally annotated with AED events and image), manages
    conversation history, and emits ``LLMContextFrame``.

    When ``LlmConfig.supports_vision`` is True and an image frame is
    available, the user message is built as a multimodal OpenAI content
    array using ``LLMContext.create_image_message()``.
    """

    def __init__(self, llm_config: LlmConfig, aed_config: AedConfig | None = None):
        super().__init__()
        self._llm = llm_config
        self._aed = aed_config or AedConfig()

        # Per-turn state
        self._turn_text: str = ""
        self._turn_aed: list[dict] = []
        self._speaking: bool = False

        # Multimodal state
        self._latest_image_frame: Optional[InputImageRawFrame] = None
        self._image_sent: bool = False

        # No local conversation history: the upstream LLM service is stateful.
        # Each request must contain exactly the latest user turn.
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputImageRawFrame):
            self._latest_image_frame = frame
            # Don't push downstream — consumed in _finalize_turn()
            return

        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._speaking = True
            self._turn_text = ""
            self._turn_aed = []
            self._image_sent = False
            await self.push_frame(frame, direction)

        elif isinstance(frame, TranscriptionFrame):
            if frame.text:
                self._turn_text = frame.text
                await self._finalize_turn()
            # Consume ASR text here. TranscriptionFrame is a TextFrame subclass;
            # forwarding it would let downstream TTS speak the user's words.
            return

        elif isinstance(frame, InterimTranscriptionFrame):
            # Interim ASR is only for local diagnostics/display, not LLM/TTS input.
            return

        elif isinstance(frame, EnvironmentalSoundFrame):
            if self._speaking and self._aed.enabled:
                self._turn_aed.append({
                    "type": frame.sound_type,
                    "confidence": frame.confidence,
                })
            return

        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._speaking = False
            if self._turn_text.strip():
                await self._finalize_turn()
            await self.push_frame(frame, direction)

        elif isinstance(frame, TextFrame):
            await self.push_frame(frame, direction)

        elif isinstance(frame, InterruptionFrame):
            self._speaking = False
            self._turn_text = ""
            self._turn_aed = []
            self._latest_image_frame = None
            self._image_sent = False
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _finalize_turn(self):
        text = self._turn_text.strip()
        has_image = (
            self._llm.supports_vision
            and self._latest_image_frame is not None
            and not self._image_sent
        )

        if not text and not has_image:
            return

        # Annotate text with AED event descriptions if enabled
        if text and self._aed.enabled and self._aed.include_in_context and self._turn_aed:
            seen = {e["type"] for e in self._turn_aed}
            descs = [self._aed.event_mappings.get(t, f"[{t}]") for t in seen]
            text = f"{text} {' '.join(descs)}"

        # Build message — multimodal if image is available
        if has_image:
            message = await LLMContext.create_image_message(
                role="user",
                format=self._latest_image_frame.format or "image/jpeg",
                size=self._latest_image_frame.size,
                image=self._latest_image_frame.image,
                text=text or "",
            )
            self._image_sent = True
        else:
            message = {"role": "user", "content": text}

        context = LLMContext(messages=[message])
        logger.info(f"LLM context <- \"{text[:80]}\"")
        await self.push_frame(LLMContextFrame(context=context))
        self._turn_text = ""
        self._turn_aed = []
