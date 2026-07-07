#!/usr/bin/env python
"""WebSocket test client for the voice-chatbot full-duplex audio pipeline.

Connects to a running WS server, sends a WAV file as input audio, and records
the synthesized TTS response to an output WAV file.

Usage::

    # Start the server first:
    python -m src.main --transport websocket

    # Run the test client:
    python tests/test_ws_client.py <input.wav> [output.wav] [--host HOST] [--port PORT]

The wire protocol is Protobuf (pipecat.frames.protobufs.frames_pb2).
Client sends AudioRawFrame messages (int16 PCM, 16kHz, mono).
Server responds with AudioRawFrame (TTS output) and TranscriptionFrame (ASR text).
"""

import argparse
import asyncio
import sys
import wave
from pathlib import Path

import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipecat.frames.protobufs.frames_pb2 as pb


def load_wav(path: str) -> tuple[bytes, int, int]:
    """Load a WAV file and return (pcm_bytes, sample_rate, num_channels)."""
    with wave.open(path, "rb") as wf:
        assert wf.getsampwidth() == 2, f"Must be 16-bit PCM, got {wf.getsampwidth() * 8}-bit"
        pcm = wf.readframes(wf.getnframes())
        return pcm, wf.getframerate(), wf.getnchannels()


def save_wav(path: str, pcm: bytes, sample_rate: int, num_channels: int = 1):
    """Save raw int16 PCM bytes as a WAV file."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def make_audio_frame(pcm: bytes, sample_rate: int, num_channels: int = 1) -> pb.Frame:
    """Build a Protobuf Frame containing an AudioRawFrame."""
    frame = pb.Frame()
    frame.audio.audio = pcm
    frame.audio.sample_rate = sample_rate
    frame.audio.num_channels = num_channels
    return frame


async def run_client(
    ws_url: str,
    input_wav: str,
    output_wav: str,
    chunk_ms: int = 100,
) -> None:
    """Connect, stream input audio, collect output audio + transcriptions."""
    pcm, sample_rate, num_channels = load_wav(input_wav)
    sample_rate = sample_rate

    # Calculate chunk size for the configured interval
    bytes_per_sample = 2  # int16
    chunk_samples = sample_rate * chunk_ms // 1000
    chunk_bytes = chunk_samples * bytes_per_sample * num_channels

    output_chunks: list[bytes] = []
    transcriptions: list[str] = []

    print(f"Connecting to {ws_url} ...")
    async with websockets.connect(ws_url) as ws:
        print("Connected. Streaming audio ...")

        # Stream input audio in chunks
        total_sent = 0
        for offset in range(0, len(pcm), chunk_bytes):
            chunk = pcm[offset : offset + chunk_bytes]
            if not chunk:
                break
            proto = make_audio_frame(chunk, sample_rate, num_channels)
            await ws.send(proto.SerializeToString())
            total_sent += len(chunk)

            # Small delay to simulate real-time streaming
            await asyncio.sleep(chunk_ms / 1000)

        print(f"Sent {total_sent / (sample_rate * bytes_per_sample):.1f}s of audio. "
              f"Waiting for response ...")

        # Receive frames from server
        try:
            async for message in ws:
                proto = pb.Frame.FromString(message)
                which = proto.WhichOneof("frame")

                if which == "audio":
                    output_chunks.append(proto.audio.audio)
                elif which == "transcription":
                    text = proto.transcription.text
                    if text:
                        transcriptions.append(text)
                        print(f"  ASR → \"{text}\"")
                elif which == "text":
                    text = proto.text.text
                    if text:
                        print(f"  Text → \"{text}\"")
                elif which == "interruption":
                    print("  ⚡ Interruption")
        except websockets.ConnectionClosed:
            print("Connection closed by server.")

    # Save output
    if output_chunks:
        combined = b"".join(output_chunks)
        save_wav(output_wav, combined, sample_rate)
        duration = len(combined) / (sample_rate * bytes_per_sample)
        print(f"Output: {output_wav} ({duration:.1f}s, {len(combined)} bytes)")
    else:
        print("No audio output received.")

    if transcriptions:
        print(f"Transcriptions: {' | '.join(transcriptions)}")


def main():
    p = argparse.ArgumentParser(
        description="WebSocket test client for voice-chatbot"
    )
    p.add_argument("input", help="Input WAV file (int16 PCM)")
    p.add_argument("output", nargs="?", default="output.wav",
                   help="Output WAV file (default: output.wav)")
    p.add_argument("--host", default="localhost", help="Server host")
    p.add_argument("--port", type=int, default=8765, help="Server port")
    p.add_argument("--chunk-ms", type=int, default=100,
                   help="Audio chunk duration in ms (default: 100)")
    args = p.parse_args()

    ws_url = f"ws://{args.host}:{args.port}"
    asyncio.run(run_client(ws_url, args.input, args.output, args.chunk_ms))


if __name__ == "__main__":
    main()
