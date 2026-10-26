"""HybridLLM: one custom class that wraps BOTH OpenAI and Ollama.

Every node in the graph asks this class for a runnable instead of importing a
provider directly, so switching backends (or running with automatic fallback)
is a config change, not a code change.

    llm = HybridLLM(settings)
    llm.chat()                          # plain chat model (text)
    llm.chat(vision=True)               # vision-capable model
    llm.structured(MySchema)            # returns a validated MySchema object
    llm.structured(MySchema, vision=True)

provider modes
    openai               OpenAI only
    ollama               Ollama only (fully local)
    openai_then_ollama   try OpenAI, fall back to Ollama on any error
    ollama_then_openai   try Ollama, fall back to OpenAI on any error
"""
from __future__ import annotations

import logging
import os
from typing import Type

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from .config import Settings

log = logging.getLogger(__name__)

_ORDERS = {
    "openai": ["openai"],
    "ollama": ["ollama"],
    "openai_then_ollama": ["openai", "ollama"],
    "ollama_then_openai": ["ollama", "openai"],
}


class HybridLLM:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or Settings.from_env()
        if self.s.provider not in _ORDERS:
            raise ValueError(f"Unknown provider '{self.s.provider}'. Choose from {list(_ORDERS)}")

    # ------------------------------------------------------------------ builders
    def _build_openai(self, vision: bool, temperature: float) -> BaseChatModel:
        from langchain_openai import ChatOpenAI

        model = self.s.openai_vision_model if vision else self.s.openai_model
        return ChatOpenAI(model=model, temperature=temperature, timeout=120, max_retries=2)

    def _build_ollama(self, vision: bool, temperature: float) -> BaseChatModel:
        from langchain_ollama import ChatOllama

        model = self.s.ollama_vision_model if vision else self.s.ollama_model
        return ChatOllama(
            model=model,
            base_url=self.s.ollama_base_url,
            temperature=temperature,
            num_ctx=self.s.ollama_num_ctx,
        )

    def _available_order(self) -> list[str]:
        order = list(_ORDERS[self.s.provider])
        if "openai" in order and not os.getenv("OPENAI_API_KEY"):
            log.warning("OPENAI_API_KEY not set - skipping OpenAI backend.")
            order.remove("openai")
        if not order:
            raise RuntimeError(
                "No usable LLM backend. Set OPENAI_API_KEY or choose provider='ollama'."
            )
        return order

    def _build(self, name: str, vision: bool, temperature: float) -> BaseChatModel:
        return (self._build_openai if name == "openai" else self._build_ollama)(vision, temperature)

    # ------------------------------------------------------------------ public API
    def chat(self, *, vision: bool = False, temperature: float | None = None) -> Runnable:
        t = self.s.grader_temperature if temperature is None else temperature
        models = [self._build(n, vision, t) for n in self._available_order()]
        return models[0].with_fallbacks(models[1:]) if len(models) > 1 else models[0]

    def structured(
        self,
        schema: Type[BaseModel],
        *,
        vision: bool = False,
        temperature: float | None = None,
    ) -> Runnable:
        """Chat model that returns a validated `schema` instance (with fallback)."""
        t = self.s.grader_temperature if temperature is None else temperature
        runnables: list[Runnable] = []
        for name in self._available_order():
            model = self._build(name, vision, t)
            method = "function_calling" if name == "openai" else "json_schema"
            runnables.append(model.with_structured_output(schema, method=method))
        return runnables[0].with_fallbacks(runnables[1:]) if len(runnables) > 1 else runnables[0]

    def describe(self) -> dict:
        return {
            "provider": self.s.provider,
            "order": self._available_order(),
            "openai": [self.s.openai_model, self.s.openai_vision_model],
            "ollama": [self.s.ollama_model, self.s.ollama_vision_model],
        }
