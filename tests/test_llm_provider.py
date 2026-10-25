"""Tests for bounded Gemini requests and timeout reporting."""

import sys
from types import ModuleType, SimpleNamespace

import google
from src import llm_provider


def _fake_genai(monkeypatch, create_interaction):
    captured = {}

    class FakeInteractions:
        def create(self, **kwargs):
            captured["interaction_request"] = kwargs
            return create_interaction(**kwargs)

    class FakeClient:
        def __init__(self, *, api_key):
            captured["api_key"] = api_key
            self.interactions = FakeInteractions()

    fake_module = ModuleType("google.genai")
    fake_module.Client = FakeClient
    monkeypatch.setitem(sys.modules, "google.genai", fake_module)
    monkeypatch.setattr(google, "genai", fake_module, raising=False)
    monkeypatch.setattr(llm_provider, "get_api_key", lambda: "test-key")
    return captured


def test_generate_answer_uses_sdk_default_timeout(monkeypatch):
    def create_interaction(**kwargs):
        assert kwargs["model"] == "gemini-3.8-flash"
        assert kwargs["input"] == "Question and evidence"
        return SimpleNamespace(output_text="A concise answer.")

    captured = _fake_genai(monkeypatch, create_interaction)

    result = llm_provider.generate_answer("Question and evidence")

    assert result["success"] is True
    assert result["answer"] == "A concise answer."
    assert result["model"] == "gemini-3.8-flash"
    assert captured["api_key"] == "test-key"
    assert captured["interaction_request"] == {
        "model": "gemini-3.8-flash",
        "input": "Question and evidence",
        "generation_config": {
            "temperature": 0.2,
                "max_output_tokens": 2048,
        },
        "store": False,
    }


def test_generate_answer_uses_requested_output_token_limit(monkeypatch):
    captured = _fake_genai(
        monkeypatch,
        lambda **kwargs: SimpleNamespace(output_text='{"questions": []}'),
    )

    result = llm_provider.generate_answer("Create a quiz", max_output_tokens=8192)

    assert result["success"] is True
    assert captured["interaction_request"]["generation_config"]["max_output_tokens"] == 8192


def test_generate_answer_reports_timeout_actionably(monkeypatch):
    def create_interaction(**kwargs):
        assert kwargs["input"] == "Question and evidence"
        raise RuntimeError("504 Deadline expired before operation could complete.")

    _fake_genai(monkeypatch, create_interaction)

    result = llm_provider.generate_answer("Question and evidence")

    assert result["success"] is False
    assert result["error"] == (
        "Gemini returned a timeout/deadline error (504). The app is showing "
        "the retrieved course passages instead. This may be temporary; try "
        "again, and check your connection if it keeps happening."
    )


def test_network_unreachable_error_is_actionable():
    message = llm_provider.format_generation_error(
        "[WinError 10051] A socket operation was attempted to an unreachable network"
    )

    assert "could not connect to Gemini" in message
    assert "VPN, proxy, or firewall" in message


def test_retrieval_fallback_shows_one_full_cited_passage():
    answer = llm_provider.format_retrieval_only_response(
        {
            "chunks": [
                {
                    "text": "Machine learning learns patterns from data.\nIt is a branch of AI.",
                    "source": "machine_learning_notes.txt",
                    "page": 2,
                },
                {
                    "text": "A less relevant passage.",
                    "source": "other.txt",
                },
            ]
        },
        generation_error="Could not connect",
    )

    assert "AI-generated explanation unavailable." in answer
    assert "no Gemini API key" not in answer
    assert "Machine learning learns patterns from data." in answer
    assert "It is a branch of AI." in answer
    assert "Source: machine_learning_notes.txt, page 2" in answer
    assert "A less relevant passage." not in answer
