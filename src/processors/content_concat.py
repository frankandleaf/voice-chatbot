"""Content concatenation + conversation history for Pipecat 1.4+.

Collects ``TranscriptionFrame`` results across a turn, concatenates on
``VADUserStoppedSpeakingFrame``, and emits ``LLMContextFrame`` for the LLM.

Supports multimodal content via ``InputImageRawFrame`` — when an image is
present, the user message is built as an OpenAI multimodal content array
with both text and image_url parts.
"""

from typing import Optional

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

        # Conversation history (list of OpenAI-format message dicts)
        self._history: list[dict] = []
        self._system = {"role": "system", "content": llm_config.system_prompt}

        # Assistant tracking — accumulates TextFrames from TTS/LLM output.
        # Saved to history in _finalize_turn() BEFORE being reset.
        self._assistant_text: str = ""

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
            if self._speaking and frame.text:
                self._turn_text = frame.text
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
            await self._finalize_turn()
            await self.push_frame(frame, direction)

        elif isinstance(frame, TextFrame):
            self._assistant_text += frame.text
            await self.push_frame(frame, direction)

        elif isinstance(frame, InterruptionFrame):
            self._speaking = False
            self._turn_text = ""
            self._turn_aed = []
            self._assistant_text = ""
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

        self._history.append(message)

        # Save assistant text from previous response
        if self._assistant_text.strip():
            self._history.append(
                {"role": "assistant", "content": self._assistant_text.strip()}
            )
        self._assistant_text = ""

        # Trim history to max_history_rounds
        max_msgs = self._llm.max_history_rounds * 2
        if len(self._history) > max_msgs:
            self._history = self._history[-max_msgs:]

        messages = [self._system] + list(self._history)
        context = LLMContext(messages=messages)
        await self.push_frame(LLMContextFrame(context=context))
