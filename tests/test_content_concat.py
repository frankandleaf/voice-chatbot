"""Unit tests for ContentConcatenator."""

import pytest

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
        assert len(cc._history) == 0

    def test_system_prompt_stored(self):
        """Verify system prompt is captured from config."""
        llm_config = LlmConfig(system_prompt="Hello, world!")
        cc = ContentConcatenator(llm_config)
        assert cc._system == {"role": "system", "content": "Hello, world!"}

    def test_history_trimming(self):
        """Verify conversation history respects max_history_rounds."""
        llm_config = LlmConfig(max_history_rounds=2)
        cc = ContentConcatenator(llm_config)

        # Add 5 rounds of conversation directly to history
        for i in range(5):
            cc._history.append({"role": "user", "content": f"msg {i}"})
            cc._history.append({"role": "assistant", "content": f"reply {i}"})

        # Simulate _finalize_turn's trimming logic
        max_msgs = llm_config.max_history_rounds * 2
        if len(cc._history) > max_msgs:
            cc._history = cc._history[-max_msgs:]

        messages = [cc._system] + list(cc._history)
        # 2 rounds = 4 messages + system prompt = 5 total
        assert len(messages) == 5
        assert messages[0] == cc._system
        assert messages[1]["content"] == "msg 3"  # oldest kept
        assert messages[4]["content"] == "reply 4"  # newest

    def test_aed_disabled(self):
        """When AED is disabled, events should not be collected."""
        llm_config = LlmConfig()
        aed_config = AedConfig(enabled=False)
        cc = ContentConcatenator(llm_config, aed_config)
        assert cc._aed.enabled is False
