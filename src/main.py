"""Voice Chatbot — ASR-LLM-TTS pipeline on Pipecat 1.4+.

Advanced VAD pipeline:
  1. AdaptiveNoiseTracker — rolling 30s noise floor
  2. AedSuppressor — spectral-feature real-time non-speech detection
  3. FsmnVadGate — FunASR fsmn-vad first-stage speech/non-speech gate
  4. VADProcessor — Silero neural VAD (second stage)
  5. SenseVoiceSTTService — ASR with AED (6 event types)
  6. SmartInterruptionGate — post-STT word-count barge-in gating
  7. AedFormatter — log environmental sound events
  8. ContentConcatenator — partial concat + AED context injection
  9. OpenAILLMService — configurable LLM
  10. VLLMTTSService — TTS with reference-audio voice cloning

Usage::

    python -m src.main                     # default config.yaml
    python -m src.main -c prod.yaml        # custom config
    python -m src.main --dump-config cfg.yaml
"""

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from src.config import AppConfig, load_config, save_default_config
from src.utils.logging import setup_logging

load_dotenv()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Voice Chatbot — Pipecat ASR-LLM-TTS")
    p.add_argument("-c", "--config", default="config.yaml", help="YAML config path")
    p.add_argument("--transport", choices=["local", "websocket", "firefly"],
                   help="Override transport mode from config")
    p.add_argument("--dump-config", metavar="PATH", help="Write default config and exit")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


async def run_pipeline(config: AppConfig) -> None:
    """Build and run the full voice chatbot pipeline."""

    # ------------------------------------------------------------------
    # Pipecat + custom imports
    # ------------------------------------------------------------------
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.audio.vad_processor import VADProcessor
    from pipecat.services.openai.llm import OpenAILLMService
    from pipecat.transports.local.audio import (
        LocalAudioTransport,
        LocalAudioTransportParams,
    )

    from src.processors.aed_formatter import AedFormatter
    from src.processors.content_concat import ContentConcatenator
    from src.processors.image_collector import ImageCollector
    from src.services.sensevoice_stt import SenseVoiceSTTService
    from src.services.vllm_tts import VLLMTTSService
    from src.vad.adaptive_noise import AdaptiveNoiseTracker
    from src.vad.aed_suppressor import AedSuppressor
    from src.vad.fsmn_gate import FsmnVadGate
    from src.vad.smart_interruption import SmartInterruptionGate

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------
    if config.pipeline.transport == "firefly":
        from src.transports.firefly import FireflyTransport, FireflyTransportParams

        transport_params = FireflyTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_in_enabled=True,
            audio_in_sample_rate=config.audio.input_sample_rate,
            audio_out_sample_rate=config.firefly.output_sample_rate,
            audio_in_channels=config.firefly.input_channels,
            audio_out_channels=config.firefly.output_channels,
            client_audio_in_sample_rate=config.firefly.input_sample_rate,
            client_audio_out_sample_rate=config.firefly.output_sample_rate,
            image_width=config.firefly.image_width,
            image_height=config.firefly.image_height,
            image_format=config.firefly.image_format,
            max_message_size_bytes=config.firefly.max_message_size_bytes,
            output_queue_size=config.firefly.output_queue_size,
            ping_interval_secs=config.firefly.ping_interval_secs,
            ping_timeout_secs=config.firefly.ping_timeout_secs,
            close_timeout_secs=config.firefly.close_timeout_secs,
        )
        transport = FireflyTransport(
            params=transport_params,
            host=config.pipeline.host,
            port=config.pipeline.port,
        )

    elif config.pipeline.transport == "websocket":
        from src.serializers.multimodal import MultimodalFrameSerializer
        from pipecat.transports.websocket.server import (
            SingleClientWebsocketServerTransport,
            SingleClientWebsocketServerParams,
        )

        transport_params = SingleClientWebsocketServerParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=config.audio.input_sample_rate,
            audio_out_sample_rate=config.audio.output_sample_rate,
            audio_in_channels=config.pipeline.channels,
            audio_out_channels=config.pipeline.channels,
            serializer=MultimodalFrameSerializer(),
            session_timeout=config.pipeline.session_timeout,
        )
        transport = SingleClientWebsocketServerTransport(
            params=transport_params,
            host=config.pipeline.host,
            port=config.pipeline.port,
        )

        @transport.event_handler("on_client_connected")
        async def _on_connected(_transport, ws):
            logger.info(f"WebSocket client connected: {ws.remote_address}")

        @transport.event_handler("on_client_disconnected")
        async def _on_disconnected(_transport, ws):
            logger.info(f"WebSocket client disconnected: {ws.remote_address}")

    else:
        transport_params = LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=config.audio.input_sample_rate,
            audio_out_sample_rate=config.audio.output_sample_rate,
            audio_in_channels=config.pipeline.channels,
            audio_out_channels=config.pipeline.channels,
        )
        transport = LocalAudioTransport(params=transport_params)

# -------------- Pre-VAD: Adaptive noise tracker --------------
    adaptive_noise = AdaptiveNoiseTracker(
        window_secs=config.vad_advanced.adaptive_noise.window_secs,
        percentile=config.vad_advanced.adaptive_noise.percentile,
        update_interval_s=config.vad_advanced.adaptive_noise.update_interval_s,
    ) if config.vad_advanced.adaptive_noise.enabled else None

    # -------------- Pre-VAD: AED suppressor (spectral features) --------------
    aed_suppressor = AedSuppressor(
        suppression_duration_ms=config.vad_advanced.aed_suppressor.suppression_duration_ms,
        rms_threshold_ratio=config.vad_advanced.aed_suppressor.rms_threshold_ratio,
        enabled_events=config.vad_advanced.aed_suppressor.enabled_events,
    ) if config.vad_advanced.aed_suppressor.enabled else None

    # -------------- Pre-VAD: fsmn-vad gate (first stage) --------------
    fsmn_gate = FsmnVadGate(
        min_speech_score=config.vad_advanced.fsmn_gate.min_speech_score,
        chunk_samples=config.vad_advanced.fsmn_gate.chunk_ms * config.audio.input_sample_rate // 1000,
        device=config.vad_advanced.fsmn_gate.device,
    ) if config.vad_advanced.fsmn_gate.enabled else None

    # -------------- Silero VAD (second stage) --------------
    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(
                start_secs=config.vad.start_secs,
                stop_secs=config.vad.stop_secs,
            ),
        ),
    )

    # ------------------------------------------------------------------
    # ASR: SenseVoiceSmall
    # ------------------------------------------------------------------
    stt = SenseVoiceSTTService(config=config.asr)

    # -------------- Post-STT: Smart interruption gate --------------
    # Moved AFTER STT so it can count transcribed words before deciding
    # whether to allow a barge-in.  Requires allow_interruptions=False
    # on the PipelineTask — the gate is the sole source of InterruptionFrame.
    smart_interrupt = SmartInterruptionGate(
        min_words=config.vad_advanced.smart_interruption.min_words,
        max_wait_secs=config.vad_advanced.smart_interruption.max_wait_secs,
    ) if config.vad_advanced.smart_interruption.enabled else None

    # ------------------------------------------------------------------
    # AED formatter (log environmental sounds)
    # ------------------------------------------------------------------
    aed_fmt = AedFormatter(config=config.aed) if config.aed.enabled else None

    # ------------------------------------------------------------------
    # Content concatenation + conversation history
    # ------------------------------------------------------------------
    concat = ContentConcatenator(llm_config=config.llm, aed_config=config.aed)

    # ------------------------------------------------------------------
    # LLM: OpenAI-compatible
    # ------------------------------------------------------------------
    llm = OpenAILLMService(
        api_key=config.llm.api_key,
        base_url=config.llm.base_url,
        settings=OpenAILLMService.Settings(
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            top_p=config.llm.top_p,
        ),
    )

    # ------------------------------------------------------------------
    # TTS: vLLM-Omni
    # ------------------------------------------------------------------
    tts = VLLMTTSService(config=config.tts)

    # ------------------------------------------------------------------
    # Assemble processors list
    # ------------------------------------------------------------------
    processors: list = [transport.input()]

    # Image collector — caches latest camera frame from Android
    image_collector = ImageCollector()
    processors.append(image_collector)

    # Pre-VAD chain
    if adaptive_noise:
        processors.append(adaptive_noise)
    if aed_suppressor:
        processors.append(aed_suppressor)
    if fsmn_gate:
        processors.append(fsmn_gate)

    # Core VAD
    processors.append(vad)

    # Main pipeline
    processors.append(stt)

    # Post-STT gate: counts words from transcription before allowing barge-in
    if smart_interrupt:
        processors.append(smart_interrupt)

    if aed_fmt:
        processors.append(aed_fmt)
    processors.append(concat)
    processors.append(llm)
    processors.append(tts)
    processors.append(transport.output())

    # ------------------------------------------------------------------
    # Pipeline lifecycle
    # ------------------------------------------------------------------
    pipeline = Pipeline(processors)
    task = PipelineTask(pipeline, params=PipelineParams(
        allow_interruptions=False,  # SmartInterruptionGate handles interruptions
    ))
    runner = PipelineRunner()

    enabled_features = []
    if fsmn_gate:
        enabled_features.append("fsmn-gate")
    if aed_suppressor:
        enabled_features.append("aed-suppress")
    if smart_interrupt:
        enabled_features.append("smart-interrupt")
    if adaptive_noise:
        enabled_features.append("adaptive-noise")
    if config.llm.supports_vision:
        enabled_features.append("vision")
    if config.asr.enable_interim:
        enabled_features.append("interim-asr")

    logger.info(
        f"Pipeline | transport={config.pipeline.transport} | asr={config.asr.device} | "
        f"llm={config.llm.model} | tts={config.tts.model}"
    )
    if config.pipeline.transport == "websocket":
        logger.info(
            f"WebSocket server → ws://{config.pipeline.host}:{config.pipeline.port}"
        )
    elif config.pipeline.transport == "firefly":
        logger.info(
            f"Firefly WebSocket server → ws://{config.pipeline.host}:{config.pipeline.port}"
        )
    if enabled_features:
        logger.info(f"VAD features | {', '.join(enabled_features)} | "
                     f"hangover={config.vad.stop_secs}s")
    if config.pipeline.transport != "websocket":
        logger.info("🎤 Speak into the mic — Ctrl+C to stop")

    try:
        await runner.run(task)
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Pipeline stopped")


def main() -> None:
    args = parse_args()

    if args.dump_config:
        save_default_config(args.dump_config)
        print(f"✓ Default config → {args.dump_config}")
        return

    setup_logging(level=args.log_level)

    config_path = Path(args.config)
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        logger.error(f"Config not found: {config_path}")
        logger.info("Run with --dump-config config.yaml to generate one")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"Config error: {exc}")
        sys.exit(1)

    # CLI transport override
    if args.transport:
        config.pipeline.transport = args.transport

    logger.info(f"Config: {config_path} | transport={config.pipeline.transport}")
    asyncio.run(run_pipeline(config))


if __name__ == "__main__":
    main()
