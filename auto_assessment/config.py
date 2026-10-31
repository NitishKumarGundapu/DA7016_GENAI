"""Central runtime settings, loaded from environment variables or a local ``.env`` file."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean flag, accepting the usual truthy spellings."""
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    """Configuration shared by the web UI, CLI, and evaluation harness."""

    # Provider used for text work (reference alignment, grading, verification, report).
    provider: str = "ollama"

    # OpenRouter models. The vision model is also used for every PDF/image OCR call,
    # regardless of which provider grades the text.
    openrouter_model: str = "openai/gpt-4o"
    openrouter_vision_model: str = "mistralai/mistral-small-3.1-24b-instruct"

    # Local Ollama endpoint used when ``provider == "ollama"``.
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:14b"
    ollama_num_ctx: int = 8192

    grader_temperature: float = 0.0
    use_critic: bool = True
    max_grade_retries: int = 2
    extraction_reasoning: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables, falling back to the defaults above."""
        defaults = cls()
        return cls(
            provider=os.getenv("LLM_PROVIDER", defaults.provider),
            openrouter_model=os.getenv("OPENROUTER_MODEL", defaults.openrouter_model),
            openrouter_vision_model=os.getenv(
                "OPENROUTER_VISION_MODEL", defaults.openrouter_vision_model
            ),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", defaults.ollama_base_url),
            ollama_model=os.getenv("OLLAMA_MODEL", defaults.ollama_model),
            ollama_num_ctx=int(os.getenv("OLLAMA_NUM_CTX", defaults.ollama_num_ctx)),
            use_critic=_env_bool("USE_CRITIC", defaults.use_critic),
            max_grade_retries=int(os.getenv("MAX_GRADE_RETRIES", defaults.max_grade_retries)),
            extraction_reasoning=_env_bool(
                "EXTRACTION_REASONING", defaults.extraction_reasoning
            ),
        )
