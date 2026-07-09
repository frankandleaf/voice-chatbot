"""Tests for the vLLM TTS frame conversion helpers."""

import io
import wave

import pytest
from pipecat.frames.frames import LLMFullResponseEndFrame, LLMTextFrame, TextFrame

from src.config import TtsConfig
from src.services.vllm_tts import VLLMTTSService


def _wav_bytes(pcm: bytes, sample_rate: int = 24000) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return out.getvalue()


def test_wav_response_is_decoded_to_pcm():
    pcm = b"\x01\x00\x02\x00" * 16
    service = VLLMTTSService(TtsConfig(response_format="wav", sample_rate=24000))

    assert service._response_to_pcm(_wav_bytes(pcm)) == pcm


def test_pcm_response_passes_through():
    pcm = b"\x01\x00\x02\x00" * 16
    service = VLLMTTSService(TtsConfig(response_format="pcm", sample_rate=24000))

    assert service._response_to_pcm(pcm) == pcm


def test_tts_chunks_use_configured_sample_rate():
    pcm = b"\x00" * 2000
    service = VLLMTTSService(TtsConfig(response_format="pcm", sample_rate=24000))

    chunks = list(service._chunks(pcm))

    assert [len(chunk) for chunk in chunks] == [960, 960, 80]


def test_tts_body_matches_qwen_speech_api():
    service = VLLMTTSService(TtsConfig(response_format="wav", speed=1.0))

    body = service._build_body("hello")

    assert body == {
        "model": "fishaudio/s2-pro",
        "input": "hello",
        "voice": "default",
        "speed": 1.0,
    }
    assert "response_format" not in body


def test_tts_uses_configured_speech_url_and_auth_header():
    service = VLLMTTSService(TtsConfig(
        base_url="http://localhost:8091/v1/",
        api_key="secret-token",
    ))

    assert service._speech_url == "http://localhost:8091/v1/audio/speech"
    assert service._headers["Authorization"] == "Bearer secret-token"


def test_tts_adds_v1_when_base_url_is_server_root():
    service = VLLMTTSService(TtsConfig(base_url="http://localhost:8091"))

    assert service._speech_url == "http://localhost:8091/v1/audio/speech"


@pytest.mark.anyio
async def test_tts_synthesizes_complete_sentences_only():
    service = VLLMTTSService(TtsConfig())
    synthesized = []

    async def synthesize(text):
        synthesized.append(text)

    service._synthesize = synthesize

    await service.process_frame(LLMTextFrame("Hello"), None)
    assert synthesized == []

    await service.process_frame(LLMTextFrame(" world. Next"), None)
    assert synthesized == ["Hello world."]

    await service.process_frame(LLMFullResponseEndFrame(), None)
    assert synthesized == ["Hello world.", "Next"]


@pytest.mark.anyio
async def test_tts_ignores_non_llm_text_frames():
    service = VLLMTTSService(TtsConfig())
    synthesized = []

    async def synthesize(text):
        synthesized.append(text)

    service._synthesize = synthesize

    async def capture(frame, direction=None):
        pass

    service.push_frame = capture

    await service.process_frame(TextFrame("user text"), None)

    assert synthesized == []
