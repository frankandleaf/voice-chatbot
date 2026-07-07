"""Unit tests for SenseVoice output parsing."""

import pytest

from src.services.sensevoice_stt import parse_sensevoice_output


class TestSenseVoiceParsing:
    """Tests for the SenseVoiceSmall output parser."""

    def test_simple_chinese(self):
        result = parse_sensevoice_output(
            "<|zh|><|NEUTRAL|><|Speech|>今天天气怎么样"
        )
        assert result["text"] == "今天天气怎么样"
        assert result["language"] == "zh"
        assert result["emotion"] == "NEUTRAL"
        assert result["aed_events"] == []

    def test_with_emotion(self):
        result = parse_sensevoice_output(
            "<|en|><|HAPPY|><|Speech|>That's wonderful news"
        )
        assert result["text"] == "That's wonderful news"
        assert result["language"] == "en"
        assert result["emotion"] == "HAPPY"

    def test_with_aed_events(self):
        result = parse_sensevoice_output(
            "<|zh|><|NEUTRAL|><|Speech|>大家好<|BGM|><|Applause|>"
        )
        assert "大家好" in result["text"]
        assert result["aed_events"] == ["BGM", "Applause"]

    def test_laughter_detection(self):
        result = parse_sensevoice_output(
            "<|en|><|HAPPY|><|Speech|>That's hilarious<|Laughter|>"
        )
        assert result["emotion"] == "HAPPY"
        assert "Laughter" in result["aed_events"]

    def test_empty_text(self):
        result = parse_sensevoice_output("")
        assert result["text"] == ""
        assert result["language"] is None
        assert result["emotion"] is None
        assert result["aed_events"] == []

    def test_mixed_tags(self):
        result = parse_sensevoice_output(
            "<|zh|><|SAD|><|Speech|>我很难过<|Crying|><|BGM|>"
        )
        assert "我很难过" in result["text"]
        assert result["emotion"] == "SAD"
        assert set(result["aed_events"]) == {"Crying", "BGM"}
