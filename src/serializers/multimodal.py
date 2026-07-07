"""Custom multimodal frame serializer for Pipecat WebSocket transport.

Extends ProtobufFrameSerializer with JPEG image frame support.

Wire format::

    byte 0: frame type indicator
      0x00 → Protobuf frame (Audio/Text/Transcription/Interruption/Message)
      0x01 → JPEG image → InputImageRawFrame
      0x02 → JSON text message (reserved for control messages)
"""

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InputImageRawFrame,
    OutputImageRawFrame,
)
from pipecat.serializers.protobuf import ProtobufFrameSerializer


class MultimodalFrameSerializer(ProtobufFrameSerializer):
    """Frame serializer that adds JPEG image support to Protobuf.

    Delegates all non-image frames to the parent ProtobufFrameSerializer.
    Image frames are serialized as raw JPEG bytes with a 0x01 prefix.
    """

    # Frame type indicators
    TYPE_PROTOBUF = 0x00
    TYPE_JPEG = 0x01
    TYPE_JSON = 0x02

    # Default image dimensions for Android camera capture
    DEFAULT_IMAGE_SIZE = (1280, 720)
    DEFAULT_IMAGE_FORMAT = "image/jpeg"

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serialize a frame to wire format.

        Image frames → 0x01 + JPEG bytes.
        All others → 0x00 + Protobuf.
        """
        if isinstance(frame, OutputImageRawFrame):
            # Serialize as 0x01 prefix + raw JPEG bytes
            image_bytes = frame.image
            return bytes([self.TYPE_JPEG]) + image_bytes

        # Delegate to Protobuf for all other frame types
        result = await super().serialize(frame)
        if result is None:
            return None
        # Prefix with protobuf type indicator
        if isinstance(result, bytes):
            return bytes([self.TYPE_PROTOBUF]) + result
        return result

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserialize wire format back to a frame.

        0x00 → Protobuf frame.
        0x01 → JPEG image → InputImageRawFrame.
        0x02 → JSON text (reserved).
        """
        if not data or (isinstance(data, bytes) and len(data) == 0):
            return None

        if isinstance(data, str):
            # String data — treat as protobuf (text messages from WebSocket)
            return await super().deserialize(data)

        # Binary data — check first byte for type
        frame_type = data[0]
        payload = data[1:]

        if frame_type == self.TYPE_PROTOBUF:
            return await super().deserialize(payload)

        elif frame_type == self.TYPE_JPEG:
            if len(payload) == 0:
                logger.warning("Received empty JPEG frame")
                return None
            return InputImageRawFrame(
                image=bytes(payload),
                size=self.DEFAULT_IMAGE_SIZE,
                format=self.DEFAULT_IMAGE_FORMAT,
            )

        elif frame_type == self.TYPE_JSON:
            # Reserved for future control messages
            logger.debug(f"Received JSON control frame: {payload[:100]}")
            return None

        else:
            logger.warning(f"Unknown frame type indicator: 0x{frame_type:02x}")
            return None
