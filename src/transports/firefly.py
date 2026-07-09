"""WebSocket transport adapter for the Firefly Android client.

The Firefly app opens separate WebSocket paths and sends raw payloads instead
of Pipecat protobuf frames:

* /microphone: raw PCM16 microphone chunks at 32 kHz mono
* /postImage: raw JPEG camera frames
* /audio: raw PCM16 assistant audio at 44.1 kHz mono
* /emotion: text commands for Unity avatar emotion changes

This adapter keeps that client protocol unchanged and translates only at the
server boundary into Pipecat frames.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlparse

from loguru import logger
from pipecat.audio.resamplers.soxr_stream_resampler import SOXRStreamAudioResampler
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InputImageRawFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pydantic import Field
from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed


MICROPHONE_PATH: Final[str] = "/microphone"
IMAGE_PATH: Final[str] = "/postImage"
AUDIO_PATH: Final[str] = "/audio"
EMOTION_PATH: Final[str] = "/emotion"
_SUPPORTED_PATHS: Final[set[str]] = {
    MICROPHONE_PATH,
    IMAGE_PATH,
    AUDIO_PATH,
    EMOTION_PATH,
}


class FireflyTransportParams(TransportParams):
    """Transport parameters for the Firefly client wire protocol."""

    client_audio_in_sample_rate: int = 32000
    client_audio_out_sample_rate: int = 44100
    image_width: int = 640
    image_height: int = 480
    image_format: str = "image/jpeg"
    max_message_size_bytes: int = 8 * 1024 * 1024
    output_queue_size: int = Field(default=256, ge=1)
    ping_interval_secs: float | None = 20.0
    ping_timeout_secs: float | None = 20.0
    close_timeout_secs: float = 1.0


@dataclass(slots=True)
class _OutputSlot:
    path: str
    websocket: ServerConnection
    queue: asyncio.Queue[str | bytes]
    sender_task: asyncio.Task


def _request_path(websocket: ServerConnection) -> str:
    request = websocket.request
    if request is None:
        return "/"
    return urlparse(request.path).path


def _trim_jpeg_payload(payload: bytes) -> bytes:
    """Trim ByteBuffer padding around a JPEG payload when present."""
    start = payload.find(b"\xff\xd8")
    if start < 0:
        return payload

    end = payload.rfind(b"\xff\xd9")
    if end >= start:
        return payload[start : end + 2]
    return payload[start:]


class _FireflyOutputRegistry:
    """Tracks Firefly output WebSockets and decouples sends from the pipeline."""

    def __init__(self, queue_size: int):
        self._queue_size = queue_size
        self._slots: dict[str, _OutputSlot] = {}
        self._lock = asyncio.Lock()

    async def hold(self, path: str, websocket: ServerConnection) -> None:
        queue: asyncio.Queue[str | bytes] = asyncio.Queue(maxsize=self._queue_size)
        sender_task = asyncio.create_task(self._sender_loop(path, websocket, queue))
        slot = _OutputSlot(path=path, websocket=websocket, queue=queue, sender_task=sender_task)

        async with self._lock:
            old = self._slots.get(path)
            self._slots[path] = slot

        if old:
            await self._close_slot(old, code=1012, reason="Replaced by a newer Firefly connection")

        logger.info(f"Firefly output connected: {path} {websocket.remote_address}")
        try:
            await websocket.wait_closed()
        finally:
            await self._remove_if_current(slot)
            await self._close_slot(slot)
            logger.info(f"Firefly output disconnected: {path} {websocket.remote_address}")

    async def send_audio(self, audio: bytes) -> bool:
        return await self._send(AUDIO_PATH, audio)

    async def send_emotion(self, command: str) -> bool:
        return await self._send(EMOTION_PATH, command)

    async def close_all(self) -> None:
        async with self._lock:
            slots = list(self._slots.values())
            self._slots.clear()

        for slot in slots:
            await self._close_slot(slot, code=1001, reason="Firefly transport stopped")

    async def _send(self, path: str, message: str | bytes) -> bool:
        async with self._lock:
            slot = self._slots.get(path)

        if slot is None or slot.websocket.state.name != "OPEN":
            return False

        if slot.queue.full():
            try:
                slot.queue.get_nowait()
                slot.queue.task_done()
                logger.warning(f"Firefly output queue full for {path}; dropped oldest chunk")
            except asyncio.QueueEmpty:
                pass

        try:
            slot.queue.put_nowait(message)
            return True
        except asyncio.QueueFull:
            logger.warning(f"Firefly output queue full for {path}; dropped current chunk")
            return False

    async def _remove_if_current(self, slot: _OutputSlot) -> None:
        async with self._lock:
            if self._slots.get(slot.path) is slot:
                del self._slots[slot.path]

    async def _close_slot(
        self,
        slot: _OutputSlot,
        *,
        code: int = 1000,
        reason: str = "",
    ) -> None:
        if not slot.sender_task.done():
            slot.sender_task.cancel()
            try:
                await slot.sender_task
            except asyncio.CancelledError:
                pass
        if slot.websocket.state.name == "OPEN":
            await slot.websocket.close(code=code, reason=reason)

    @staticmethod
    async def _sender_loop(
        path: str,
        websocket: ServerConnection,
        queue: asyncio.Queue[str | bytes],
    ) -> None:
        try:
            while True:
                message = await queue.get()
                try:
                    await websocket.send(message)
                finally:
                    queue.task_done()
        except asyncio.CancelledError:
            raise
        except ConnectionClosed:
            pass
        except Exception as exc:
            logger.warning(f"Firefly sender failed for {path}: {exc}")


class FireflyInputTransport(BaseInputTransport):
    """Receives Firefly microphone and camera WebSockets."""

    _params: FireflyTransportParams

    def __init__(
        self,
        *,
        host: str,
        port: int,
        params: FireflyTransportParams,
        output_registry: _FireflyOutputRegistry,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self._host = host
        self._port = port
        self._output_registry = output_registry
        self._server: Server | None = None
        self._input_connections: dict[str, ServerConnection] = {}
        self._input_lock = asyncio.Lock()

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self.set_transport_ready(frame)

        if self._server is not None:
            return

        self._server = await serve(
            self._handle_connection,
            self._host,
            self._port,
            compression=None,
            max_size=self._params.max_message_size_bytes,
            ping_interval=self._params.ping_interval_secs,
            ping_timeout=self._params.ping_timeout_secs,
            close_timeout=self._params.close_timeout_secs,
        )
        logger.info(
            f"Firefly WebSocket server ready: ws://{self._host}:{self._port} "
            f"paths={sorted(_SUPPORTED_PATHS)}"
        )

    async def stop(self, frame: EndFrame):
        await self._close_server()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        await self._close_server()
        await super().cancel(frame)

    async def cleanup(self):
        await self._close_server()
        await super().cleanup()

    async def _close_server(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        await self._output_registry.close_all()

        async with self._input_lock:
            connections = list(self._input_connections.values())
            self._input_connections.clear()

        for websocket in connections:
            if websocket.state.name == "OPEN":
                await websocket.close(code=1001, reason="Firefly transport stopped")

    async def _handle_connection(self, websocket: ServerConnection) -> None:
        path = _request_path(websocket)
        if path not in _SUPPORTED_PATHS:
            logger.warning(f"Rejecting unsupported Firefly WebSocket path: {path}")
            await websocket.close(code=1008, reason="Unsupported Firefly endpoint")
            return

        if path == MICROPHONE_PATH:
            await self._handle_microphone(websocket)
        elif path == IMAGE_PATH:
            await self._handle_images(websocket)
        elif path in {AUDIO_PATH, EMOTION_PATH}:
            await self._output_registry.hold(path, websocket)

    async def _replace_input_connection(self, path: str, websocket: ServerConnection) -> None:
        async with self._input_lock:
            old = self._input_connections.get(path)
            self._input_connections[path] = websocket

        if old and old is not websocket and old.state.name == "OPEN":
            await old.close(code=1012, reason="Replaced by a newer Firefly connection")

    async def _remove_input_connection(self, path: str, websocket: ServerConnection) -> None:
        async with self._input_lock:
            if self._input_connections.get(path) is websocket:
                del self._input_connections[path]

    async def _handle_microphone(self, websocket: ServerConnection) -> None:
        await self._replace_input_connection(MICROPHONE_PATH, websocket)
        resampler = SOXRStreamAudioResampler(quality="QQ")
        logger.info(f"Firefly microphone connected: {websocket.remote_address}")

        try:
            async for message in websocket:
                if not isinstance(message, bytes):
                    logger.debug("Ignoring non-binary message on /microphone")
                    continue
                if not message:
                    continue

                audio = await resampler.resample(
                    message,
                    self._params.client_audio_in_sample_rate,
                    self.sample_rate,
                )
                if audio:
                    await self.push_audio_frame(
                        InputAudioRawFrame(
                            audio=audio,
                            sample_rate=self.sample_rate,
                            num_channels=self._params.audio_in_channels,
                        )
                    )
        except ConnectionClosed:
            pass
        finally:
            await self._remove_input_connection(MICROPHONE_PATH, websocket)
            logger.info(f"Firefly microphone disconnected: {websocket.remote_address}")

    async def _handle_images(self, websocket: ServerConnection) -> None:
        await self._replace_input_connection(IMAGE_PATH, websocket)
        logger.info(f"Firefly camera connected: {websocket.remote_address}")

        try:
            async for message in websocket:
                if not isinstance(message, bytes):
                    logger.debug("Ignoring non-binary message on /postImage")
                    continue
                image = _trim_jpeg_payload(message)
                if not image:
                    continue
                await self.push_video_frame(
                    InputImageRawFrame(
                        image=image,
                        size=(self._params.image_width, self._params.image_height),
                        format=self._params.image_format,
                    )
                )
        except ConnectionClosed:
            pass
        finally:
            await self._remove_input_connection(IMAGE_PATH, websocket)
            logger.info(f"Firefly camera disconnected: {websocket.remote_address}")


class FireflyOutputTransport(BaseOutputTransport):
    """Sends assistant audio to the Firefly /audio WebSocket."""

    _params: FireflyTransportParams

    def __init__(self, params: FireflyTransportParams, registry: _FireflyOutputRegistry, **kwargs):
        super().__init__(params, **kwargs)
        self._registry = registry

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        if not frame.audio:
            return True
        sent = await self._registry.send_audio(frame.audio)
        if not sent:
            logger.debug("Dropping Firefly audio chunk because /audio is not connected")
        return sent

    async def send_message(
        self,
        frame: OutputTransportMessageFrame | OutputTransportMessageUrgentFrame,
    ):
        message = frame.message
        if isinstance(message, str):
            await self._registry.send_emotion(message)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)


class FireflyTransport(BaseTransport):
    """Pipecat transport that adapts the existing Firefly client protocol."""

    def __init__(
        self,
        *,
        params: FireflyTransportParams,
        host: str = "0.0.0.0",
        port: int = 8765,
        input_name: str | None = None,
        output_name: str | None = None,
    ):
        super().__init__(input_name=input_name, output_name=output_name)
        self._params = params
        self._host = host
        self._port = port
        self._registry = _FireflyOutputRegistry(params.output_queue_size)
        self._input: FireflyInputTransport | None = None
        self._output: FireflyOutputTransport | None = None

    def input(self) -> FrameProcessor:
        if self._input is None:
            self._input = FireflyInputTransport(
                host=self._host,
                port=self._port,
                params=self._params,
                output_registry=self._registry,
                name=self._input_name,
            )
        return self._input

    def output(self) -> FrameProcessor:
        if self._output is None:
            self._output = FireflyOutputTransport(
                params=self._params,
                registry=self._registry,
                name=self._output_name,
            )
        return self._output
