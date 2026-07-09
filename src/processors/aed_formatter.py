"""AED formatter — log environmental sounds, pass through all frames."""

from collections import deque

from loguru import logger
from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from src.config import AedConfig
from src.services.sensevoice_stt import EnvironmentalSoundFrame


class AedFormatter(FrameProcessor):
    """Translate AED events → human-readable log lines.

    Pass-through for all non-AED frames.  The actual LLM context injection
    is done by ``ContentConcatenator``.
    """

    def __init__(self, config: AedConfig):
        super().__init__()
        self._config = config
        self._recent: deque[EnvironmentalSoundFrame] = deque(maxlen=20)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, EnvironmentalSoundFrame) and self._config.enabled:
            self._recent.append(frame)
            desc = self._config.event_mappings.get(
                frame.sound_type, f"[{frame.sound_type}]"
            )
            logger.info(f"AED → {desc}")
        await self.push_frame(frame, direction)

    @property
    def recent_events(self) -> list[EnvironmentalSoundFrame]:
        return list(self._recent)
