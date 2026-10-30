"""Compatibility import; new code should use llm_client.create_llm."""
from .llm_client import create_llm as HybridLLM

__all__ = ["HybridLLM"]
