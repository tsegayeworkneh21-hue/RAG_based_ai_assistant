"""Unit tests for PipelineConfig validation."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ConfigError, DEFAULT_CONFIG, PipelineConfig


def test_default_config_matches_decision_log():
    assert DEFAULT_CONFIG.chunk_size == 800
    assert DEFAULT_CONFIG.overlap == 150
    assert DEFAULT_CONFIG.min_score == 0.25
    assert DEFAULT_CONFIG.top_k == 5
    assert "all-MiniLM-L6-v2" in DEFAULT_CONFIG.embedding_model


def test_rejects_overlap_equal_to_chunk_size():
    with pytest.raises(ConfigError, match="overlap"):
        PipelineConfig(chunk_size=100, overlap=100)


def test_rejects_min_score_out_of_range():
    with pytest.raises(ConfigError, match="min_score"):
        PipelineConfig(min_score=1.5)


def test_rejects_empty_model_name():
    with pytest.raises(ConfigError, match="embedding_model"):
        PipelineConfig(embedding_model="   ")


def test_rejects_non_positive_top_k():
    with pytest.raises(ConfigError, match="top_k"):
        PipelineConfig(top_k=0)


def test_as_dict_round_trip_keys():
    payload = DEFAULT_CONFIG.as_dict()
    assert set(payload) == {
        "embedding_model",
        "chunk_size",
        "overlap",
        "top_k",
        "min_score",
    }
