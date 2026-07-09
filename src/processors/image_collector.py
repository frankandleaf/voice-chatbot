"""Image collector frame processor for Pipecat.

Caches the latest image frame from the transport and injects it into the
pipeline when the user stops speaking (VADUserStoppedSpeakingFrame).
"""

from typing import Optional

from pipecat.frames.frames import (
    Frame,
    InputImageRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class ImageCollector(FrameProcessor):
    """Caches the most recent image frame and pushes it on end-of-speech.

    Only the latest image is kept — older images are discarded. This ensures
    the LLM always sees the most current visual context when the user
    finishes speaking.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._latest_image: Optional[InputImageRawFrame] = None
        self._user_speaking: bool = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputImageRawFrame):
            self._latest_image = frame
            # Don't push downstream yet — wait for end-of-speech
            return

        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._user_speaking = True
            await self.push_frame(frame, direction)

        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._user_speaking = False
            await self.push_frame(frame, direction)
            # Inject latest image after VAD stop so ContentConcatenator
            # can include it in the multimodal LLM context
            if self._latest_image:
                await self.push_frame(self._latest_image, direction)

        else:
            await self.push_frame(frame, direction)
