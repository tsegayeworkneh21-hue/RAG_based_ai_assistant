"""
config.py
=========
Validated pipeline settings used by ingestion, retrieval, and evaluation.

WHY A SINGLE CONFIG OBJECT?
---------------------------
Chunk size, overlap, similarity threshold, and top-k were originally scattered
as magic numbers (800 / 150 / 0.25 / 5) after informal tests on the verified
QA set. Centralising them makes inputs explicit, invalid combinations fail
fast, and evaluation can re-run against the same defaults the UI uses.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


class ConfigError(ValueError):
    """Raised when pipeline settings are missing or out of range."""


@dataclass(frozen=True)
class PipelineConfig:
    """
    Inputs that control document chunking and retrieval.

    Parameters
    ----------
    embedding_model : HuggingFace model id for sentence-transformers.
    chunk_size      : Target characters per chunk (must be > overlap).
    overlap         : Characters repeated between consecutive chunks.
    top_k           : Maximum chunks returned for a query.
    min_score       : Minimum cosine similarity for a chunk to count as evidence.
    """

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 800
    overlap: int = 150
    top_k: int = 5
    min_score: float = 0.25

    def __post_init__(self) -> None:
        validate_pipeline_config(self)

    def as_dict(self) -> dict:
        """JSON-serializable copy of the settings."""
        return asdict(self)


def validate_pipeline_config(config: PipelineConfig) -> None:
    """
    Reject settings that cannot produce a well-defined pipeline.

    Raises
    ------
    ConfigError
        If any field is empty, non-positive, or internally inconsistent.
    """
    if not isinstance(config.embedding_model, str) or not config.embedding_model.strip():
        raise ConfigError("embedding_model must be a non-empty string.")

    if not isinstance(config.chunk_size, int) or config.chunk_size <= 0:
        raise ConfigError("chunk_size must be a positive integer.")

    if not isinstance(config.overlap, int) or config.overlap < 0:
        raise ConfigError("overlap must be a non-negative integer.")

    if config.overlap >= config.chunk_size:
        raise ConfigError(
            f"overlap ({config.overlap}) must be smaller than chunk_size ({config.chunk_size})."
        )

    if not isinstance(config.top_k, int) or config.top_k < 1:
        raise ConfigError("top_k must be an integer >= 1.")

    if not isinstance(config.min_score, (int, float)) or isinstance(config.min_score, bool):
        raise ConfigError("min_score must be a number between 0 and 1.")

    if not 0.0 <= float(config.min_score) <= 1.0:
        raise ConfigError("min_score must be between 0.0 and 1.0 inclusive.")


DEFAULT_CONFIG = PipelineConfig()
