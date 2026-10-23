"""Tests for embedding model error reporting."""

import sys
from types import ModuleType

from src import embeddings


def test_model_load_error_is_available_to_the_ui(monkeypatch):
    class BrokenSentenceTransformer:
        def __init__(self, model_name):
            raise OSError("missing test dependency")

    fake_module = ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = BrokenSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.setattr(embeddings, "_model", None)

    assert embeddings.get_embedding_model() is None
    assert embeddings.get_embedding_error() == "OSError: missing test dependency"
