"""Tests for the vLLM TTS frame conversion helpers."""

import io
import wave

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
