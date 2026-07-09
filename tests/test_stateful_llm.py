from types import SimpleNamespace

import pytest
from pipecat.frames.frames import LLMTextFrame
from pipecat.processors.aggregators.llm_context import LLMContext

from src.config import LlmConfig
from src.services.stateful_llm import StatefulOpenAILLMService, app_identity


def test_app_identity_prefixes_and_sanitizes_device_name():
    assert app_identity("Pixel 8 Pro") == "app_Pixel_8_Pro"
    assert app_identity("app_Frank-Phone") == "app_Frank-Phone"
    assert app_identity("") == "app_firefly"


def test_build_params_adds_stateful_fields_and_disables_streaming():
    service = StatefulOpenAILLMService(
        stateful_config=LlmConfig(device_name="Pixel 8 Pro", headless=False, stream=False),
        api_key="test-key",
        base_url="http://llm.test/v1",
        settings=StatefulOpenAILLMService.Settings(model="test-model"),
    )

    params = service.build_chat_completion_params({"messages": [{"role": "user", "content": "hi"}]})

    assert params["user_id"] == "app_Pixel_8_Pro"
    assert params["session_id"] == "app_Pixel_8_Pro"
    assert params["headless"] is False
    assert params["stream"] is False
    assert "stream_options" not in params


@pytest.mark.anyio
async def test_non_stream_response_pushes_single_llm_text(monkeypatch):
    service = StatefulOpenAILLMService(
        stateful_config=LlmConfig(device_name="phone", headless=False, stream=False),
        api_key="test-key",
        base_url="http://llm.test/v1",
        settings=StatefulOpenAILLMService.Settings(model="test-model"),
    )
    pushed = []

    async def capture(frame, direction=None):
        pushed.append(frame)

    async def noop(*args, **kwargs):
        return None

    class FakeCompletions:
        async def create(self, **params):
            assert params["user_id"] == "app_phone"
            assert params["session_id"] == "app_phone"
            assert params["headless"] is False
            assert params["stream"] is False
            return SimpleNamespace(
                model="fake-model",
                usage=None,
                choices=[SimpleNamespace(message=SimpleNamespace(content="hello there."))],
            )

    service._client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    monkeypatch.setattr(service, "push_frame", capture)
    monkeypatch.setattr(service, "start_ttfb_metrics", noop)
    monkeypatch.setattr(service, "stop_ttfb_metrics", noop)

    await service._process_context(LLMContext(messages=[{"role": "user", "content": "hi"}]))

    llm_text = [frame for frame in pushed if isinstance(frame, LLMTextFrame)]
    assert len(llm_text) == 1
    assert llm_text[0].text == "hello there."
