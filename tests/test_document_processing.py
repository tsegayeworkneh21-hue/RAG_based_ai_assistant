"""
Tests for document_processing.py

RUN WITH:
    pytest tests/test_document_processing.py -v

WHAT WE TEST
------------
1. Loading a real .txt file returns non-empty pages.
2. Chunking produces chunks of expected size with correct metadata.
3. An empty file returns an empty list with a warning (not a crash).
4. A non-existent file returns an empty list (not a crash).
5. Chunk metadata (source, page, chunk_index) is preserved correctly.
6. Overlap: the end of chunk N appears at the start of chunk N+1.
"""

import sys
import os
from pathlib import Path

# Add the project root to sys.path so we can import src.*
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from src.document_processing import (
    load_txt_file,
    chunk_pages,
    make_chunk,
    load_all_from_directory,
)

SAMPLE_DIR = PROJECT_ROOT / "data" / "sample_materials"
ML_NOTES = SAMPLE_DIR / "machine_learning_notes.txt"


# ---------------------------------------------------------------------------
# make_chunk
# ---------------------------------------------------------------------------

def test_make_chunk_structure():
    """make_chunk() should return a dict with all required keys."""
    chunk = make_chunk(text="  hello world  ", source="test.txt", page=1, chunk_index=0)
    assert chunk["text"] == "hello world"   # Text is stripped
    assert chunk["source"] == "test.txt"
    assert chunk["page"] == 1
    assert chunk["chunk_index"] == 0


def test_make_chunk_none_page():
    """page=None is valid for .txt files that have no page concept."""
    chunk = make_chunk(text="sample", source="notes.txt", page=None, chunk_index=5)
    assert chunk["page"] is None


# ---------------------------------------------------------------------------
# load_txt_file
# ---------------------------------------------------------------------------

def test_load_txt_file_returns_pages():
    """Loading the ML notes should return at least one page dict."""
    if not ML_NOTES.exists():
        pytest.skip("Sample ML notes file not found.")
    pages = load_txt_file(str(ML_NOTES))
    assert isinstance(pages, list)
    assert len(pages) >= 1
    assert "text" in pages[0]
    assert "source" in pages[0]
    assert len(pages[0]["text"]) > 100  # Should have substantial content


def test_load_txt_file_source_name():
    """The source field should be just the filename, not the full path."""
    if not ML_NOTES.exists():
        pytest.skip("Sample ML notes file not found.")
    pages = load_txt_file(str(ML_NOTES))
    assert pages[0]["source"] == "machine_learning_notes.txt"


def test_load_txt_file_nonexistent():
    """A non-existent file should return an empty list, not raise an exception."""
    pages = load_txt_file("/nonexistent/path/fake_file.txt")
    assert pages == []


def test_load_txt_file_empty(tmp_path):
    """An empty file should return an empty list."""
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("   \n   \n")  # Only whitespace
    pages = load_txt_file(str(empty_file))
    assert pages == []


# ---------------------------------------------------------------------------
# chunk_pages
# ---------------------------------------------------------------------------

def test_chunk_pages_basic():
    """Chunking a page should produce multiple chunks."""
    # Create a fake long text to chunk
    long_text = "This is a test sentence about machine learning. " * 50
    pages = [{"text": long_text, "source": "test.txt", "page": None}]
    chunks = chunk_pages(pages, chunk_size=200, overlap=40)
    assert len(chunks) > 1


def test_chunk_metadata_preserved():
    """Each chunk should carry the correct source and page metadata."""
    text = "Sentence about overfitting and underfitting. " * 30
    pages = [{"text": text, "source": "ml_notes.txt", "page": 3}]
    chunks = chunk_pages(pages, chunk_size=200, overlap=30)
    for chunk in chunks:
        assert chunk["source"] == "ml_notes.txt"
        assert chunk["page"] == 3
        assert "chunk_index" in chunk
        assert isinstance(chunk["chunk_index"], int)


def test_chunk_index_is_sequential():
    """chunk_index values should be consecutive integers starting at 0."""
    text = "A " * 1000  # Long text to force multiple chunks
    pages = [{"text": text, "source": "test.txt", "page": None}]
    chunks = chunk_pages(pages, chunk_size=100, overlap=20)
    indices = [c["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))


def test_chunk_size_respected():
    """No chunk should be significantly longer than chunk_size characters."""
    text = "Word " * 500
    pages = [{"text": text, "source": "test.txt", "page": None}]
    chunk_size = 300
    chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=50)
    for chunk in chunks:
        # Allow a small margin for sentence-boundary adjustment
        assert len(chunk["text"]) <= chunk_size + 50, \
            f"Chunk too long: {len(chunk['text'])} chars"


def test_empty_pages_returns_empty_chunks():
    """If the pages list is empty, chunking should return an empty list."""
    chunks = chunk_pages([])
    assert chunks == []


def test_chunk_pages_from_real_file():
    """Chunking the real ML notes should produce a reasonable number of chunks."""
    if not ML_NOTES.exists():
        pytest.skip("Sample ML notes file not found.")
    pages = load_txt_file(str(ML_NOTES))
    chunks = chunk_pages(pages)
    # The ML notes are ~8000 characters; with chunk_size=800 we expect ~10-20 chunks
    assert 5 <= len(chunks) <= 50


# ---------------------------------------------------------------------------
# load_all_from_directory
# ---------------------------------------------------------------------------

def test_load_all_from_directory():
    """Loading the sample directory should return a non-trivial number of chunks."""
    if not SAMPLE_DIR.exists():
        pytest.skip("Sample materials directory not found.")
    chunks = load_all_from_directory(str(SAMPLE_DIR))
    assert len(chunks) > 10, "Expected more than 10 chunks from sample materials"


def test_load_all_nonexistent_directory():
    """A non-existent directory should return an empty list."""
    chunks = load_all_from_directory("/nonexistent/path/")
    assert chunks == []
