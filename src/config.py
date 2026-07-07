"""Configuration schema and loader.

Uses Pydantic v2 for validation and PyYAML for parsing.
Environment variables are interpolated via ${VAR_NAME} syntax.
"""

import os
import re
from pathlib import Path
from typing import Literal, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Individual config models
# ---------------------------------------------------------------------------


class PipelineConfig(BaseModel):
    """Top-level pipeline / transport settings."""

    sample_rate: int = 16000
    channels: int = 1
    transport: Literal["local", "websocket"] = "local"
    host: str = "0.0.0.0"
    port: int = 8765
    session_timeout: int | None = Field(
        default=None,
        description="WebSocket session timeout in seconds (None = unlimited)",
    )


class VadConfig(BaseModel):
    """Voice Activity Detection parameters."""

    provider: Literal["silero"] = "silero"
    start_secs: float = Field(default=0.2, description="Seconds of speech before VAD activates")
    stop_secs: float = Field(
        default=0.5, description="Seconds of silence before VAD deactivates (hangover)"
    )
    sample_rate: int = 16000


class FsmnGateConfig(BaseModel):
    """fsmn-vad first-stage gate configuration."""

    enabled: bool = True
    min_speech_score: float = Field(default=0.5, description="Score threshold (0-1)")
    chunk_ms: int = Field(default=100, description="Inference chunk duration")
    device: str = "cpu"


class AedSuppressorConfig(BaseModel):
    """Real-time AED for VAD suppression."""

    enabled: bool = True
    suppression_duration_ms: int = Field(default=500, description="Suppress VAD for this long")
    rms_threshold_ratio: float = Field(default=2.0, description="Above noise floor")
    enabled_events: list[str] = Field(
        default=["music", "clap", "bang", "screech", "silence"]
    )


class SmartInterruptionConfig(BaseModel):
    """Min-words gating for interruption."""

    enabled: bool = True
    min_words: int = Field(default=2, description="Minimum words to allow interruption")
    max_wait_secs: float = Field(default=2.0, description="Max wait before releasing anyway")


class AdaptiveNoiseConfig(BaseModel):
    """Rolling noise-floor tracker."""

    enabled: bool = True
    window_secs: float = Field(default=30.0, description="Rolling window duration")
    percentile: float = Field(default=85.0, description="Noise floor percentile")
    update_interval_s: float = Field(default=1.0, description="Emit interval")


class VadAdvancedConfig(BaseModel):
    """All advanced VAD/AED features."""

    fsmn_gate: FsmnGateConfig = FsmnGateConfig()
    aed_suppressor: AedSuppressorConfig = AedSuppressorConfig()
    smart_interruption: SmartInterruptionConfig = SmartInterruptionConfig()
    adaptive_noise: AdaptiveNoiseConfig = AdaptiveNoiseConfig()


class AsrConfig(BaseModel):
    """ASR configuration for SenseVoiceSmall."""

    provider: Literal["sensevoice"] = "sensevoice"
    model: str = "iic/SenseVoiceSmall"
    vad_model: str = "fsmn-vad"
    device: str = "cuda"
    use_itn: bool = True
    sample_rate: int = 16000
    # Enable interim (streaming) results
    enable_interim: bool = False
    interim_interval_ms: int = Field(
        default=320, description="Interval for interim ASR results in milliseconds"
    )


class LlmConfig(BaseModel):
    """OpenAI-compatible LLM configuration.

    The user fills in base_url, api_key, and model.
    """

    provider: Literal["openai"] = "openai"
    base_url: str = Field(default="https://api.openai.com/v1")
    api_key: str = Field(default="")
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 1.0
    system_prompt: str = "You are a helpful voice assistant. Keep responses concise and conversational."
    max_history_rounds: int = Field(
        default=20, description="Maximum number of conversation turns to keep"
    )
    supports_vision: bool = Field(
        default=False,
        description="Whether the model supports vision (multimodal) input. "
                    "When True, images from transport are included in LLM context.",
    )


class TtsConfig(BaseModel):
    """vLLM-Omni TTS configuration with voice cloning support."""

    provider: Literal["vllm-omni"] = "vllm-omni"
    base_url: str = "http://localhost:8091/v1"
    api_key: str = "not-needed"
    model: str = "fishaudio/s2-pro"
    voice: str = "default"
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Speech speed multiplier")
    ref_audio_path: Optional[str] = Field(
        default=None, description="Path to reference audio for voice cloning"
    )
    ref_text: Optional[str] = Field(
        default=None, description="Transcript of the reference audio"
    )
    sample_rate: int = 24000
    response_format: str = "wav"


class AedConfig(BaseModel):
    """Audio Event Detection configuration."""

    enabled: bool = True
    include_in_context: bool = Field(
        default=True,
        description="Whether to inject AED events into LLM context",
    )
    # Event types SenseVoiceSmall can detect
    event_mappings: dict[str, str] = Field(
        default={
            "BGM": "[Background music playing]",
            "Applause": "[Applause detected]",
            "Laughter": "[Laughter heard]",
            "Coughing": "[Coughing detected]",
            "Sneeze": "[Sneeze heard]",
            "Crying": "[Crying detected]",
        }
    )


class AudioConfig(BaseModel):
    """Audio device and I/O settings."""

    input_device: Optional[str] = None
    output_device: Optional[str] = None
    input_sample_rate: int = 16000
    output_sample_rate: int = 24000
    frames_per_buffer: int = Field(
        default=1600, description="Audio chunk size in samples for local transport"
    )


class AppConfig(BaseModel):
    """Root configuration aggregating all sub-sections."""

    pipeline: PipelineConfig = PipelineConfig()
    vad: VadConfig = VadConfig()
    vad_advanced: VadAdvancedConfig = VadAdvancedConfig()
    asr: AsrConfig = AsrConfig()
    llm: LlmConfig = LlmConfig()
    tts: TtsConfig = TtsConfig()
    aed: AedConfig = AedConfig()
    audio: AudioConfig = AudioConfig()


# ---------------------------------------------------------------------------
# YAML loader with env-var interpolation
# ---------------------------------------------------------------------------

_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")


def _interpolate_env(value: str) -> str:
    """Replace ${VAR_NAME} with environment variable values."""

    def _replace(match: re.Match) -> str:
        var = match.group(1)
        env_val = os.environ.get(var, "")
        if env_val == "":
            # If we have a .env file loaded value, try that
            pass
        return env_val

    return _ENV_VAR_RE.sub(_replace, value)


def _walk_and_interpolate(obj: dict) -> dict:
    """Recursively interpolate env vars in all string values of a dict."""
    for key, value in obj.items():
        if isinstance(value, str):
            obj[key] = _interpolate_env(value)
        elif isinstance(value, dict):
            obj[key] = _walk_and_interpolate(value)
        elif isinstance(value, list):
            obj[key] = [
                _interpolate_env(v) if isinstance(v, str) else v for v in value
            ]
    return obj


def load_config(config_path: str | Path) -> AppConfig:
    """Load and validate configuration from a YAML file.

    Environment variables in the form ``${VAR_NAME}`` are interpolated
    automatically.  ``.env`` files in the current directory are loaded
    as a fallback.

    Args:
        config_path: Path to a YAML configuration file.

    Returns:
        A validated ``AppConfig`` instance.
    """
    # Load .env if present (doesn't override existing env vars)
    load_dotenv(override=False)

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    raw = _walk_and_interpolate(raw)

    return AppConfig(**raw)


def save_default_config(path: str | Path) -> None:
    """Write a default config.yaml for the user to customise."""
    config = AppConfig()
    # Use model_dump with json-friendly serialisation
    data = config.model_dump()

    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
