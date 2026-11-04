"""Route OpenEvolve's mutation LLM through Inspect AI.

OpenEvolve talks to OpenAI-compatible endpoints by default. The paper drove the mutation
operator with Claude Opus 4.6 at xhigh reasoning effort, so we give OpenEvolve an
Inspect-backed backend instead.

Registered under the provider name "inspect", so a config just says:

    llm:
      models:
        - name: "anthropic/claude-opus-4-6"
          provider: "inspect"
"""

import logging

from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)

PROVIDER_NAME = "inspect"


class InspectLLM(LLMInterface):
    """OpenEvolve LLM backend that calls an Inspect AI model."""

    def __init__(self, model_cfg=None):
        self.model_name = model_cfg.name
        self.system_message = getattr(model_cfg, "system_message", None)
        self.max_tokens = getattr(model_cfg, "max_tokens", None)
        self.temperature = getattr(model_cfg, "temperature", None)
        self.timeout = getattr(model_cfg, "timeout", None)
        self.reasoning_effort = getattr(model_cfg, "reasoning_effort", None)
        self.weight = getattr(model_cfg, "weight", 1.0)
        self._model = None

    def _get_model(self):
        if self._model is None:
            from inspect_ai.model import GenerateConfig, get_model
            cfg = {}
            if self.max_tokens:
                cfg["max_tokens"] = self.max_tokens
            if self.temperature is not None:
                cfg["temperature"] = self.temperature
            if self.reasoning_effort:
                cfg["reasoning_effort"] = self.reasoning_effort
            if self.timeout:
                cfg["timeout"] = int(self.timeout)
            # 1M-token context beta, matching the eval harness. The optimizer prompt
            # carries several full generator prompts plus a score table.
            headers = None
            if "opus-4-6" in self.model_name or "sonnet-4-6" in self.model_name:
                headers = {"anthropic_beta": "context-1m-2025-08-07"}
            if headers:
                cfg["extra_headers"] = headers
            self._model = get_model(self.model_name, config=GenerateConfig(**cfg))
        return self._model

    async def generate(self, prompt: str, **kwargs) -> str:
        return await self.generate_with_context(
            system_message=self.system_message,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(self, system_message, messages, **kwargs) -> str:
        from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser

        chat = []
        if system_message:
            chat.append(ChatMessageSystem(content=system_message))
        for m in messages:
            role, content = m.get("role"), m.get("content", "")
            if role == "system":
                chat.append(ChatMessageSystem(content=content))
            elif role == "assistant":
                chat.append(ChatMessageAssistant(content=content))
            else:
                chat.append(ChatMessageUser(content=content))

        result = await self._get_model().generate(chat)
        return result.completion or ""


def register() -> None:
    """Add the "inspect" provider to OpenEvolve's registry. Idempotent."""
    from openevolve.llm.ensemble import _PROVIDER_REGISTRY
    if PROVIDER_NAME not in _PROVIDER_REGISTRY:
        _PROVIDER_REGISTRY[PROVIDER_NAME] = lambda cfg: InspectLLM(cfg)
        logger.info("registered OpenEvolve LLM provider %r", PROVIDER_NAME)
