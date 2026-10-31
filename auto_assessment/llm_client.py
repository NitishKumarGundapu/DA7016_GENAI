"""Provider-specific LangChain clients behind one structured-output interface."""
from __future__ import annotations

import os
from dataclasses import replace

from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .config import Settings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class LLMClient:
    """Shared wrapper around a LangChain chat model.

    Callers never mutate the underlying model: temperature and reasoning overrides
    are applied to a copy for each call.
    """

    model: BaseChatModel

    def __init__(self, settings: Settings, model: BaseChatModel):
        self.settings = settings
        self.model = model

    def chat(
        self,
        temperature: float | None = None,
        *,
        reasoning: bool | None = None,
    ) -> BaseChatModel:
        """Return a model copy configured for one call."""
        temperature = self.settings.grader_temperature if temperature is None else temperature
        model = self.model.model_copy(update={"temperature": temperature})
        if reasoning is None:
            return model
        if self.settings.provider == "ollama":
            return model.model_copy(update={"reasoning": reasoning})
        body = dict(model.extra_body or {})
        body["reasoning"] = {"enabled": reasoning}
        return model.model_copy(update={"extra_body": body})

    def invoke(
        self,
        prompt: LanguageModelInput,
        *,
        temperature: float | None = None,
        reasoning: bool | None = None,
    ) -> BaseMessage:
        """Send a prompt and return the provider's message."""
        return self.chat(temperature=temperature, reasoning=reasoning).invoke(prompt)

    def structured(
        self,
        schema: type[BaseModel],
        *,
        temperature: float | None = None,
        reasoning: bool | None = None,
    ) -> Runnable:
        """Return a runnable that produces validated instances of ``schema``."""
        return self.chat(temperature=temperature, reasoning=reasoning).with_structured_output(schema)

    def describe(self) -> dict[str, str]:
        """Summarize the active models for LangSmith run metadata."""
        return {
            "provider": self.settings.provider,
            "text_model": (
                self.settings.ollama_model
                if self.settings.provider == "ollama"
                else self.settings.openrouter_model
            ),
            "vision_model": self.settings.openrouter_vision_model,
            "ocr_provider": "openrouter",
            "pdf_engine": "mistral-ocr",
        }


class OllamaClient(LLMClient):
    """Local Ollama chat model."""

    def __init__(self, settings: Settings | None = None):
        settings = replace(settings or Settings.from_env(), provider="ollama")
        super().__init__(
            settings,
            ChatOllama(
                model=settings.ollama_model,
                base_url=settings.ollama_base_url,
                temperature=settings.grader_temperature,
                num_ctx=settings.ollama_num_ctx,
            ),
        )


class OpenRouterClient(LLMClient):
    """OpenRouter-hosted chat model (OpenAI-compatible API)."""

    def __init__(self, settings: Settings | None = None):
        settings = replace(settings or Settings.from_env(), provider="openrouter")
        super().__init__(
            settings,
            ChatOpenAI(
                model=settings.openrouter_model,
                base_url=OPENROUTER_BASE_URL,
                api_key=os.getenv("OPENROUTER_API_KEY"),
                temperature=settings.grader_temperature,
                timeout=120,
                max_retries=2,
            ),
        )


def create_llm(settings: Settings | None = None) -> LLMClient:
    """Return the client selected by ``settings.provider``."""
    settings = settings or Settings.from_env()
    if settings.provider == "ollama":
        return OllamaClient(settings)
    if settings.provider == "openrouter":
        return OpenRouterClient(settings)
    raise ValueError("provider must be either 'ollama' or 'openrouter'")
