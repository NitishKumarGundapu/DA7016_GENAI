"""Central settings, loaded from environment variables / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # openai | ollama | openai_then_ollama | ollama_then_openai
    provider: str = "openai_then_ollama"

    openai_model: str = "gpt-4o"
    openai_vision_model: str = "gpt-4o"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:14b"
    ollama_vision_model: str = "qwen2.5vl:7b"
    ollama_num_ctx: int = 8192

    grader_temperature: float = 0.0
    use_critic: bool = True
    max_grade_retries: int = 2

    pdf_dpi: int = 200
    max_image_side: int = 1800
    transcribe_concurrency: int = 4

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        return cls(
            provider=os.getenv("LLM_PROVIDER", d.provider),
            openai_model=os.getenv("OPENAI_MODEL", d.openai_model),
            openai_vision_model=os.getenv("OPENAI_VISION_MODEL", d.openai_vision_model),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", d.ollama_base_url),
            ollama_model=os.getenv("OLLAMA_MODEL", d.ollama_model),
            ollama_vision_model=os.getenv("OLLAMA_VISION_MODEL", d.ollama_vision_model),
            ollama_num_ctx=int(os.getenv("OLLAMA_NUM_CTX", d.ollama_num_ctx)),
            use_critic=_env_bool("USE_CRITIC", d.use_critic),
            max_grade_retries=int(os.getenv("MAX_GRADE_RETRIES", d.max_grade_retries)),
            pdf_dpi=int(os.getenv("PDF_DPI", d.pdf_dpi)),
            max_image_side=int(os.getenv("MAX_IMAGE_SIDE", d.max_image_side)),
        )
