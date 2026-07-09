"""Tests for the Firefly Android client transport adapter."""

from src.config import AppConfig, load_config
from src.transports.firefly import FireflyTransportParams, _trim_jpeg_payload


def test_app_config_accepts_firefly_transport():
    config = AppConfig(pipeline={"transport": "firefly"})

    assert config.pipeline.transport == "firefly"
    assert config.firefly.input_sample_rate == 32000
    assert config.firefly.output_sample_rate == 44100


def test_checked_in_config_loads_firefly_settings():
    config = load_config("config.yaml")

    assert config.pipeline.transport == "firefly"
    assert config.firefly.input_sample_rate == 32000
    assert config.firefly.output_sample_rate == 44100


def test_firefly_transport_params_map_client_rates():
    params = FireflyTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_in_enabled=True,
        audio_in_sample_rate=16000,
        audio_out_sample_rate=44100,
        client_audio_in_sample_rate=32000,
        client_audio_out_sample_rate=44100,
    )

    assert params.audio_in_sample_rate == 16000
    assert params.audio_out_sample_rate == 44100
    assert params.client_audio_in_sample_rate == 32000
    assert params.client_audio_out_sample_rate == 44100


def test_trim_jpeg_payload_removes_bytebuffer_padding():
    jpeg = b"\xff\xd8jpeg-bytes\xff\xd9"
    payload = b"prefix" + jpeg + b"\x00\x00padding"

    assert _trim_jpeg_payload(payload) == jpeg


def test_trim_jpeg_payload_keeps_unknown_payload():
    payload = b"not-a-jpeg"

    assert _trim_jpeg_payload(payload) == payload
