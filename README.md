# Voice Chatbot — Pipecat ASR-LLM-TTS Pipeline

Real-time voice chatbot built on [Pipecat](https://github.com/pipecat-ai/pipecat) with:

- **ASR**: SenseVoiceSmall (FunAudioLLM/Alibaba) — multilingual + environmental sound detection
- **LLM**: OpenAI-compatible (configurable — any OpenAI-compatible endpoint)
- **TTS**: vLLM-Omni with reference-audio voice cloning

Features VAD-based interruption (barge-in), content concatenation, and AED (Audio Event Detection) for environmental sounds like music, laughter, and coughing.

## Architecture

```
Microphone → VAD (Silero) → SenseVoiceSmall ASR → AED Formatter
    → Content Concatenator → OpenAI LLM → vLLM TTS → Speaker
```

## Quick Start

### 1. Prerequisites

- Python 3.10+
- CUDA-compatible GPU (or CPU mode)
- [vLLM-Omni](https://github.com/vllm-project/vllm-omni) server running for TTS

### 2. Setup

```bash
cd voice-chatbot
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env with your LLM endpoint and API key
```

Then edit `config.yaml`:
- Set `tts.ref_audio_path` to your voice reference WAV
- Set `tts.ref_text` to the transcript of that reference
- Adjust `asr.device` to `cuda` or `cpu`
- Tune `vad.start_secs` and `vad.stop_secs` for your environment

### 4. Start TTS Server (separate terminal)

```bash
vllm serve fishaudio/s2-pro --omni --port 8091
```

Or use any model that vLLM-Omni supports (CosyVoice3, Qwen3-TTS, etc.).

### 5. Run

```bash
python -m src.main
```

With the checked-in config this starts the Firefly WebSocket adapter. Use
`--transport local` for local microphone/speaker testing. Press `Ctrl+C` to stop.

## Configuration Reference

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `pipeline` | `transport` | `firefly` | `local` for mic/speaker, `websocket` for Pipecat protobuf, `firefly` for the Android client |
| `vad` | `start_secs` | `0.2` | Speech duration before activation |
| `vad` | `stop_secs` | `0.3` | Silence before end-of-speech |
| `asr` | `device` | `cuda` | `cuda` or `cpu` |
| `asr` | `enable_interim` | `false` | Enable streaming interim ASR results |
| `llm` | `base_url` | `${LLM_BASE_URL}` | OpenAI-compatible API URL |
| `llm` | `model` | `${LLM_MODEL}` | Model name |
| `llm` | `max_history_rounds` | `20` | Conversation turns to remember |
| `tts` | `base_url` | `http://localhost:8091/v1` | vLLM-Omni endpoint |
| `tts` | `ref_audio_path` | `null` | Path to reference WAV for voice cloning |
| `tts` | `ref_text` | `null` | Transcript of reference audio |
| `aed` | `enabled` | `true` | Enable environmental sound detection |
| `aed` | `include_in_context` | `true` | Inject events into LLM context |
| `firefly` | `input_sample_rate` | `32000` | Android `/microphone` raw PCM16 mono input rate |
| `firefly` | `output_sample_rate` | `44100` | Android `/audio` raw PCM16 mono playback rate |

## Environmental Sound Detection

SenseVoiceSmall detects 6 event types:
- **BGM** — Background music
- **Applause** — Clapping
- **Laughter** — Laughing
- **Coughing** — Cough sounds
- **Sneeze** — Sneezing
- **Crying** — Crying sounds

Detected events are appended to the user message as annotations:
```
User: What's the weather like? [Background music playing]
```

## TTS Voice Cloning

To use a custom voice:
1. Record a short (3-15 second) clean audio clip of the target voice
2. Transcribe what's said in the clip
3. Set `tts.ref_audio_path` and `tts.ref_text` in `config.yaml`

## License

This project code is MIT. Individual components:
- Pipecat: BSD 2-Clause
- SenseVoiceSmall: ModelScope License (verify for commercial use)
- Silero VAD: MIT

## Project Structure

```
voice-chatbot/
├── config.yaml          # User configuration
├── pyproject.toml       # Dependencies
├── src/
│   ├── main.py          # Entry point, pipeline assembly
│   ├── config.py        # Pydantic config schema + loader
│   ├── services/
│   │   ├── sensevoice_stt.py  # SenseVoiceSmall STT service
│   │   └── vllm_tts.py        # vLLM-Omni TTS service
│   ├── processors/
│   │   ├── content_concat.py  # Content concatenation + history
│   │   └── aed_formatter.py   # AED event formatter
│   └── utils/
│       ├── audio.py      # Audio utilities
│       └── logging.py    # Logging setup
└── tests/                # Unit tests
```
