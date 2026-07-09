"""Unit tests for ContentConcatenator."""

import pytest

from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    LLMContextFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)

from src.config import AedConfig, LlmConfig
from src.processors.content_concat import ContentConcatenator


class TestContentConcatenator:
    """Tests for the ContentConcatenator processor."""

    def test_initial_state(self):
        """Verify initial state is clean."""
        llm_config = LlmConfig()
        cc = ContentConcatenator(llm_config)
        assert cc._turn_text == ""
        assert cc._turn_aed == []
        assert cc._speaking is False

    def test_no_local_history_is_kept(self):
        """History lives in the upstream LLM service, not this processor."""
        cc = ContentConcatenator(LlmConfig())
        assert not hasattr(cc, "_history")

    def test_aed_disabled(self):
        """When AED is disabled, events should not be collected."""
        llm_config = LlmConfig()
        aed_config = AedConfig(enabled=False)
        cc = ContentConcatenator(llm_config, aed_config)
        assert cc._aed.enabled is False

    @pytest.mark.anyio
    async def test_transcription_frames_are_consumed_before_tts(self):
        """ASR text frames are TextFrame subclasses but must not reach TTS."""
        cc = ContentConcatenator(LlmConfig())
        pushed = []

        async def capture(frame, direction=None):
            pushed.append(frame)

        cc.push_frame = capture

        await cc.process_frame(VADUserStartedSpeakingFrame(), None)
        await cc.process_frame(
            TranscriptionFrame(text="hello", user_id="user", timestamp="1"),
            None,
        )
        await cc.process_frame(
            InterimTranscriptionFrame(text="hello", user_id="user", timestamp="1"),
            None,
        )

        assert not any(isinstance(frame, TranscriptionFrame) for frame in pushed)
        assert not any(isinstance(frame, InterimTranscriptionFrame) for frame in pushed)
        assert any(isinstance(frame, LLMContextFrame) for frame in pushed)
        assert cc._turn_text == ""

    @pytest.mark.anyio
    async def test_transcription_after_vad_stop_still_triggers_llm(self):
        """Segmented STT can emit the final transcription after VAD stop."""
        cc = ContentConcatenator(LlmConfig())
        pushed = []

        async def capture(frame, direction=None):
            pushed.append(frame)

        cc.push_frame = capture

        await cc.process_frame(VADUserStartedSpeakingFrame(), None)
        await cc.process_frame(VADUserStoppedSpeakingFrame(), None)
        await cc.process_frame(
            TranscriptionFrame(text="after stop", user_id="user", timestamp="1"),
            None,
        )

        contexts = [frame for frame in pushed if isinstance(frame, LLMContextFrame)]
        assert len(contexts) == 1
        assert contexts[0].context.messages[-1]["content"] == "after stop"
        assert [message["role"] for message in contexts[0].context.messages] == ["user"]

    @pytest.mark.anyio
    async def test_every_turn_sends_only_latest_user_message(self):
        cc = ContentConcatenator(LlmConfig())
        pushed = []

        async def capture(frame, direction=None):
            pushed.append(frame)

        cc.push_frame = capture

        await cc.process_frame(
            TranscriptionFrame(text="first", user_id="user", timestamp="1"),
            None,
        )
        await cc.process_frame(
            TranscriptionFrame(text="second", user_id="user", timestamp="1"),
            None,
        )

        contexts = [frame for frame in pushed if isinstance(frame, LLMContextFrame)]
        assert [context.context.messages for context in contexts] == [
            [{"role": "user", "content": "first"}],
            [{"role": "user", "content": "second"}],
        ]
        assert not hasattr(cc, "_history")
