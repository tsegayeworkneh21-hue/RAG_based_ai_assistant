"""
document_processing.py
======================
STEP 1 and STEP 2 of the RAG pipeline: Document Ingestion and Text Extraction.
STEP 3: Text Chunking.

WHAT THIS MODULE DOES
---------------------
- Loads .txt and .pdf files from disk or from uploaded bytes.
- Extracts clean text from each page/section.
- Splits the text into overlapping chunks with source metadata attached.

WHY CHUNKING MATTERS
--------------------
Language models and embedding models have a maximum input size (context window).
A 50-page textbook cannot be fed to an embedding model all at once.
We split documents into small, manageable pieces called "chunks".

Each chunk carries metadata (source file name, page number, chunk index) so we
can cite it later.

OVERLAP
-------
We overlap consecutive chunks by a few sentences. This prevents important context
from being split across a chunk boundary. If a key concept spans two pages, the
overlapping chunk will still contain it.
"""

import os
import re
from pathlib import Path
from typing import Optional

from src.config import DEFAULT_CONFIG

# We try to import PyMuPDF (fitz). If it is not installed, PDF loading is disabled
# and we warn the user gracefully instead of crashing.
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data structure: a single chunk of text with its metadata
# ---------------------------------------------------------------------------

def make_chunk(text: str, source: str, page: Optional[int], chunk_index: int) -> dict:
    """
    Create a chunk dictionary — the fundamental unit of our knowledge base.

    Parameters
    ----------
    text        : The actual text content of this chunk.
    source      : The filename or label this chunk came from (e.g. "ml_notes.txt").
    page        : The page number within the source document, or None for .txt files.
    chunk_index : A sequential number identifying this chunk within the document.

    Returns
    -------
    A plain Python dict. We use dicts (not classes) to keep the code simple
    and easy to serialize if we ever need to save the index to disk.
    """
    return {
        "text": text.strip(),          # Remove leading/trailing whitespace
        "source": source,              # File name — used for citations
        "page": page,                  # Page number — used for citations
        "chunk_index": chunk_index,    # Order within the source document
    }


# ---------------------------------------------------------------------------
# Text extraction: .txt files
# ---------------------------------------------------------------------------

def load_txt_file(file_path: str) -> list[dict]:
    """
    Read a plain-text file and return a list with one dict per section.
    We treat the whole file as a single logical document at "page 1".

    Parameters
    ----------
    file_path : Absolute or relative path to a .txt file.

    Returns
    -------
    A list containing one dict: {"text": <full text>, "page": 1, "source": <filename>}
    Returns an empty list if the file cannot be read.
    """
    path = Path(file_path)
    source_name = path.name  # e.g. "machine_learning_notes.txt"

    try:
        # We read with UTF-8 encoding and fall back to latin-1 for older files.
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="latin-1")
        except Exception as e:
            print(f"[WARNING] Could not read {source_name}: {e}")
            return []
    except FileNotFoundError:
        print(f"[WARNING] File not found: {file_path}")
        return []

    if not text.strip():
        print(f"[WARNING] {source_name} is empty or contains no readable text.")
        return []

    # Return the text as a single page-level entry.
    # We will split it into chunks in the next step.
    return [{"text": text, "page": None, "source": source_name}]


# ---------------------------------------------------------------------------
# Text extraction: .pdf files
# ---------------------------------------------------------------------------

def load_pdf_file(file_path: str) -> list[dict]:
    """
    Extract text from a PDF file, one dict per page.

    Parameters
    ----------
    file_path : Path to a .pdf file.

    Returns
    -------
    A list of dicts, each with {"text": <page text>, "page": <page number>, "source": <filename>}
    Returns an empty list with a warning if PyMuPDF is not installed or the file
    contains no extractable text (e.g., a scanned image PDF).

    NOTE ON SCANNED PDFs
    --------------------
    PyMuPDF extracts digital text embedded in the PDF. If the PDF is a scanned image
    (photos of pages), there is no embedded text — the result will be empty strings.
    Extracting text from scanned PDFs requires OCR software (Optical Character Recognition)
    such as Tesseract. We detect and warn about this case rather than silently returning nothing.
    """
    if not PYMUPDF_AVAILABLE:
        print("[ERROR] PyMuPDF is not installed. Cannot process PDF files.")
        print("        Run:  pip install pymupdf")
        return []

    path = Path(file_path)
    source_name = path.name
    pages = []

    try:
        doc = fitz.open(str(file_path))  # Open the PDF document
    except Exception as e:
        print(f"[ERROR] Could not open PDF {source_name}: {e}")
        return []

    total_text_length = 0

    for page_num in range(len(doc)):  # Iterate over each page (0-indexed internally)
        page = doc[page_num]
        page_text = page.get_text()  # Extract all text on this page as a string

        total_text_length += len(page_text.strip())

        if page_text.strip():  # Only keep pages that have text
            pages.append({
                "text": page_text,
                "page": page_num + 1,   # Convert to 1-indexed for user display
                "source": source_name,
            })

    doc.close()

    if total_text_length == 0:
        print(f"[WARNING] {source_name} contains no extractable text. It may be a scanned PDF.")
        print("          OCR (Optical Character Recognition) would be required to extract text from scanned PDFs.")

    return pages


# ---------------------------------------------------------------------------
# Load from uploaded bytes (for Streamlit file uploader)
# ---------------------------------------------------------------------------

def load_uploaded_file(file_bytes: bytes, file_name: str) -> list[dict]:
    """
    Process a file uploaded through the Streamlit interface.
    Streamlit gives us the file as raw bytes — we save it to a temporary
    location, process it with the appropriate function, then return the pages.

    Parameters
    ----------
    file_bytes : Raw bytes from st.file_uploader.
    file_name  : Original file name, used to determine the file type.

    Returns
    -------
    List of page dicts, same format as load_txt_file / load_pdf_file.
    """
    import tempfile

    # Get the file extension (lower-case)
    ext = Path(file_name).suffix.lower()

    # Write bytes to a temporary file so our existing loaders can process it
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        if ext == ".txt":
            pages = load_txt_file(tmp_path)
        elif ext == ".pdf":
            pages = load_pdf_file(tmp_path)
        else:
            print(f"[WARNING] Unsupported file type: {ext}. Only .txt and .pdf are supported.")
            return []
    finally:
        # Always clean up the temporary file
        os.unlink(tmp_path)

    # Override the source name with the original uploaded file name
    for p in pages:
        p["source"] = file_name

    return pages


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_pages(
    pages: list[dict],
    chunk_size: int = DEFAULT_CONFIG.chunk_size,
    overlap: int = DEFAULT_CONFIG.overlap,
) -> list[dict]:
    """
    Split a list of page-level text dicts into smaller, overlapping chunks.

    PARAMETERS EXPLAINED
    --------------------
    chunk_size (default 800):
        The target number of characters per chunk. We measure in characters
        rather than words because it is simpler and consistent. 800 characters
        is roughly 120-150 words — enough context for meaningful retrieval
        without exceeding embedding-model limits.

    overlap (default 150):
        The number of characters at the END of one chunk that are REPEATED at
        the START of the next chunk. This prevents important sentences from
        being cut in half between two chunks.

        Example (with chunk_size=20, overlap=5):
          Original: "ABCDEFGHIJKLMNOPQRSTU"
          Chunk 0:  "ABCDEFGHIJKLMNOPQRST"    (chars 0-19)
          Chunk 1:  "PQRSTUVWXYZ..."           (chars 15-34, repeating PQRST)

    WHY NOT JUST SPLIT BY SENTENCES?
    ----------------------------------
    Sentence-boundary splitting is more elegant but harder to implement
    reliably across mixed content (bullet points, equations, headers).
    Fixed-size character-based chunking with overlap is simple, well-understood,
    and works well in practice for this application.

    Parameters
    ----------
    pages      : Output of load_txt_file or load_pdf_file.
    chunk_size : Target character length per chunk.
    overlap    : Number of characters shared between consecutive chunks.

    Returns
    -------
    A flat list of chunk dicts. Each dict has keys:
      "text", "source", "page", "chunk_index"
    """
    chunks = []
    chunk_index = 0

    for page_dict in pages:
        text = page_dict["text"]
        source = page_dict["source"]
        page = page_dict["page"]

        # Strip excessive blank lines to reduce noise in the chunks
        text = re.sub(r'\n{3,}', '\n\n', text)

        start = 0  # Pointer to the start of the current chunk

        while start < len(text):
            end = start + chunk_size  # Compute the end of this chunk

            # Extract the candidate chunk text
            chunk_text = text[start:end]

            # If this is not the last chunk, try to find a sentence boundary
            # near the end so we don't cut mid-sentence.
            if end < len(text):
                # Look for the last period, newline, or sentence-end within
                # the last 20% of the chunk.
                search_zone = chunk_text[int(chunk_size * 0.8):]
                last_boundary = max(
                    search_zone.rfind('. '),
                    search_zone.rfind('\n'),
                    search_zone.rfind('? '),
                    search_zone.rfind('! '),
                )
                if last_boundary != -1:
                    # Adjust end to the boundary position
                    adjusted_end = start + int(chunk_size * 0.8) + last_boundary + 1
                    chunk_text = text[start:adjusted_end]
                    end = adjusted_end

            chunk_text = chunk_text.strip()

            if len(chunk_text) > 20:  # Skip trivially short fragments
                chunks.append(make_chunk(
                    text=chunk_text,
                    source=source,
                    page=page,
                    chunk_index=chunk_index,
                ))
                chunk_index += 1

            # Move start forward by (chunk_size - overlap) to create the overlap
            # If we adjusted the end, we still step forward from the original
            # position to ensure consistent overlap.
            step = max(end - start - overlap, 1)  # Avoid infinite loop
            start += step

    return chunks


# ---------------------------------------------------------------------------
# Convenience: load all sample materials from a directory
# ---------------------------------------------------------------------------

def load_all_from_directory(
    directory: str,
    chunk_size: int = DEFAULT_CONFIG.chunk_size,
    overlap: int = DEFAULT_CONFIG.overlap,
) -> list[dict]:
    """
    Load and chunk every .txt and .pdf file found in a directory.

    Parameters
    ----------
    directory  : Path to a folder containing study material files.
    chunk_size : Target characters per chunk (from PipelineConfig).
    overlap    : Character overlap between consecutive chunks.

    Returns
    -------
    A flat list of chunk dicts ready for embedding.
    """
    dir_path = Path(directory)
    all_chunks = []

    if not dir_path.exists():
        print(f"[ERROR] Directory does not exist: {directory}")
        return []

    # Collect all supported files
    files = list(dir_path.glob("*.txt")) + list(dir_path.glob("*.pdf"))

    if not files:
        print(f"[WARNING] No .txt or .pdf files found in: {directory}")
        return []

    for file_path in sorted(files):
        # Skip the verified QA JSON file — it is not a study document
        if file_path.name == "verified_qa_set.json":
            continue

        print(f"[INFO] Loading: {file_path.name}")

        ext = file_path.suffix.lower()
        if ext == ".txt":
            pages = load_txt_file(str(file_path))
        elif ext == ".pdf":
            pages = load_pdf_file(str(file_path))
        else:
            continue  # skip unsupported types

        file_chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
        print(f"       → {len(file_chunks)} chunks extracted")
        all_chunks.extend(file_chunks)

    print(f"[INFO] Total chunks loaded: {len(all_chunks)}")
    return all_chunks
