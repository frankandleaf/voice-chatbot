"""OpenAI-compatible LLM adapter for the stateful Firefly FastAPI service."""

from __future__ import annotations

import re
from typing import Any

from loguru import logger
from pipecat.frames.frames import LLMTextFrame
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.settings import assert_given
from pipecat.utils.tracing.service_decorators import traced_llm

from src.config import LlmConfig


_ID_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


def app_identity(device_name: str) -> str:
    """Build the app_<device> id expected by the stateful LLM API."""
    device = _ID_UNSAFE_CHARS.sub("_", device_name.strip()).strip("_") or "firefly"
    if device.startswith("app_"):
        return device
    return f"app_{device}"


class StatefulOpenAILLMService(OpenAILLMService):
    """OpenAI service variant that sends Firefly state/session fields."""

    def __init__(self, *, stateful_config: LlmConfig, **kwargs: Any):
        self._stateful_identity = app_identity(stateful_config.device_name)
        self._stateful_headless = stateful_config.headless
        self._stateful_stream = stateful_config.stream
        super().__init__(**kwargs)
        logger.info(
            "Stateful LLM identity | "
            f"user_id=session_id={self._stateful_identity} | "
            f"headless={self._stateful_headless} | stream={self._stateful_stream}"
        )

    def build_chat_completion_params(self, params_from_context: Any) -> dict:
        params = super().build_chat_completion_params(params_from_context)
        params.update({
            "user_id": self._stateful_identity,
            "session_id": self._stateful_identity,
            "headless": self._stateful_headless,
            "stream": self._stateful_stream,
        })
        if not self._stateful_stream:
            params.pop("stream_options", None)
        return params

    @traced_llm
    async def _process_context(self, context: LLMContext):
        if self._stateful_stream:
            await super()._process_context(context)
            return

        await self.start_ttfb_metrics()
        adapter = self.get_llm_adapter()
        params_from_context = adapter.get_llm_invocation_params(
            context,
            system_instruction=assert_given(self._settings.system_instruction),
            convert_developer_to_user=not self.supports_developer_role,
        )
        params = self.build_chat_completion_params(params_from_context)
        response = await self._client.chat.completions.create(**params)
        await self.stop_ttfb_metrics()

        if response.model and self.get_full_model_name() != response.model:
            self.set_full_model_name(response.model)

        if response.usage:
            await self.start_llm_usage_metrics(
                LLMTokenUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )
            )

        if not response.choices:
            return

        content = response.choices[0].message.content
        if content:
            await self.push_frame(LLMTextFrame(text=content))
