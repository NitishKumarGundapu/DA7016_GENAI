"""Public package surface for the Auto-Assessment grading pipeline.

The web UI, CLI, and evaluation scripts import from here rather than reaching
into individual modules.
"""
from .config import Settings
from .llm import HybridLLM
from .llm_client import OllamaClient, OpenRouterClient, create_llm
from .pipeline import run_assessment, stream_assessment

__all__ = [
    "Settings",
    "HybridLLM",
    "OllamaClient",
    "OpenRouterClient",
    "create_llm",
    "run_assessment",
    "stream_assessment",
]
