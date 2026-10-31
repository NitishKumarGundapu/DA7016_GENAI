"""Offline client configuration, model identity, and compatibility checks."""
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from pydantic import ValidationError

from auto_assessment.config import Settings
from auto_assessment.llm import HybridLLM
from auto_assessment.llm_client import OllamaClient, OpenRouterClient, create_llm
from auto_assessment.models import CriticVerdict, QuestionGrade
from auto_assessment.schemas import CriticVerdict as LegacyCriticVerdict
from auto_assessment.state import GraphState as LegacyGraphState
from auto_assessment.models import GraphState


@pytest.mark.parametrize("provider,client_type", [
    ("ollama", OllamaClient),
    ("openrouter", OpenRouterClient),
])
def test_provider_selection_and_per_call_temperature(provider, client_type):
    settings = Settings(
        provider=provider, grader_temperature=0.4,
        ollama_model="local-test", ollama_base_url="http://localhost:11435",
        ollama_num_ctx=4096, openrouter_model="remote-test",
    )
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        client = create_llm(settings)
        legacy = HybridLLM(settings)
    assert isinstance(client, client_type)
    assert isinstance(legacy, client_type)
    assert client.describe()["provider"] == provider
    assert client.describe()["text_model"] == (
        "local-test" if provider == "ollama" else "remote-test"
    )
    assert client.chat().temperature == 0.4
    assert client.chat(temperature=0).temperature == 0
    assert client.chat(temperature=0.2).temperature == 0.2
    assert client.model.temperature == 0.4
    assert isinstance(client.structured(CriticVerdict, temperature=0), Runnable)
    if provider == "ollama":
        assert client.model.base_url == "http://localhost:11435"
        assert client.model.num_ctx == 4096
    else:
        assert client.model.openai_api_base == "https://openrouter.ai/api/v1"
        assert client.model.request_timeout == 120
        assert client.model.max_retries == 2


@pytest.mark.parametrize("provider", ["ollama", "openrouter"])
def test_reasoning_can_be_disabled_per_call_without_mutating_the_client(provider):
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        client = create_llm(Settings(provider=provider))
    disabled = client.chat(temperature=0, reasoning=False)
    enabled = client.chat(reasoning=True)
    if provider == "ollama":
        assert disabled.reasoning is False and enabled.reasoning is True
        assert client.model.reasoning is None
    else:
        assert disabled.extra_body["reasoning"] == {"enabled": False}
        assert enabled.extra_body["reasoning"] == {"enabled": True}
        assert client.model.extra_body is None
        assert client.model.reasoning is None


@pytest.mark.parametrize("client_type,provider", [
    (OllamaClient, "ollama"),
    (OpenRouterClient, "openrouter"),
])
def test_explicit_client_has_correct_provider_without_mutating_settings(client_type, provider):
    settings = Settings(provider="openrouter" if provider == "ollama" else "ollama")
    original_provider = settings.provider
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        client = client_type(settings)
    assert client.describe()["provider"] == provider
    assert settings.provider == original_provider


def test_extraction_reasoning_env_flag_defaults_to_disabled():
    assert Settings().extraction_reasoning is False
    with patch.dict("os.environ", {"EXTRACTION_REASONING": "true"}):
        assert Settings.from_env().extraction_reasoning is True
    with patch.dict("os.environ", {"EXTRACTION_REASONING": "false"}):
        assert Settings.from_env().extraction_reasoning is False


def test_invalid_provider_rejected():
    with pytest.raises(ValueError, match="provider must"):
        create_llm(Settings(provider="unsupported"))


def test_invoke_preserves_message_usage_and_propagates_errors():
    client = OllamaClient(Settings())
    message = AIMessage(
        content="response",
        usage_metadata={"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
    )
    with patch.object(client, "chat") as chat:
        chat.return_value.invoke.return_value = message
        assert client.invoke("prompt", temperature=0) is message
        chat.assert_called_once_with(temperature=0, reasoning=None)
        chat.return_value.invoke.assert_called_once_with("prompt")
        chat.return_value.invoke.side_effect = RuntimeError("Provider unavailable")
        with pytest.raises(RuntimeError, match="Provider unavailable"):
            client.invoke("prompt")


def test_consolidated_models_preserve_validation_identity_and_independent_lists():
    assert LegacyCriticVerdict is CriticVerdict
    assert LegacyGraphState is GraphState
    with pytest.raises(ValidationError):
        CriticVerdict.model_validate({"ok": True})
    first = QuestionGrade(question_id="1", max_marks=2, marks_awarded=0, attempted=False)
    second = QuestionGrade(question_id="2", max_marks=2, marks_awarded=0, attempted=False)
    first.verification_notes.append("Review required")
    assert second.verification_notes == []
