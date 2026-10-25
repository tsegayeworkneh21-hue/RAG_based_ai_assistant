"""
llm_provider.py
===============
STEP 8 of the RAG pipeline: Language Model Generation.

WHAT THIS MODULE DOES
---------------------
- Provides a clean interface to the configured language model (Gemini by default).
- Loads the API key from environment variables or Streamlit secrets.
- Handles the case where no API key is available by returning a "retrieval-only"
  response that displays the raw retrieved passages instead of a generated answer.
- Separates LLM logic from the rest of the pipeline so it can be swapped easily.

WHY GEMINI?
-----------
Google's Gemini API:
  - Has a free tier suitable for a student project.
  - Has an official Python library (google-generativeai).
  - Supports long context windows, which is important when passing multiple
    retrieved passages as context.
  - Does not require a credit card for the free tier.

HOW TO GET A KEY
----------------
1. Go to https://aistudio.google.com/
2. Sign in with a Google account.
3. Click "Get API key" and create a key.
4. Copy the key into your .env file as: GEMINI_API_KEY=your_key_here

RETRIEVAL-ONLY MODE
-------------------
If the API key is missing or invalid, the system falls back to displaying
the retrieved passages directly. This is honest: we label it clearly as
"Retrieval-Only Mode" so the user knows they are seeing raw passages, not
a generated answer. We do NOT pretend this is a successful RAG response.
"""

import os
from typing import Optional

# Try to import python-dotenv for loading .env files
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if present — runs at import time
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


def get_api_key() -> Optional[str]:
    """
    Retrieve the Gemini API key from the environment.

    We check in this order:
    1. Environment variable GEMINI_API_KEY (set by .env or system environment).
    2. Streamlit's st.secrets (used when deployed on Streamlit Community Cloud).

    Returns
    -------
    The API key string, or None if not found.
    """
    # Option 1: Check environment variable (works locally with .env file)
    key = os.getenv("GEMINI_API_KEY")
    if key and key.strip():
        return key.strip()

    # Option 2: Try Streamlit secrets (works on Streamlit Cloud deployment)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass  # Streamlit not available in test environment

    return None


def is_api_available() -> bool:
    """
    Check whether a valid API key is configured.
    Returns True if a key exists, False otherwise.
    Does NOT make a network call — just checks if the key string is present.
    """
    return get_api_key() is not None


def format_generation_error(error: str) -> str:
    """Return actionable wording for common Gemini connectivity failures."""
    lowered_error = error.lower()
    if any(
        marker in lowered_error
        for marker in (
            "winerror 10051",
            "10051",
            "unreachable network",
            "network is unreachable",
            "failed to establish a new connection",
        )
    ):
        return (
            "This computer could not connect to Gemini. Check that your internet "
            "connection is active and that a VPN, proxy, or firewall is not "
            "blocking the connection, then try again."
        )

    if any(
        marker in error.lower()
        for marker in (
            "deadline exceeded",
            "deadline expired",
            "timed out",
            "timeout",
            "504",
        )
    ):
        return (
            "Gemini returned a timeout/deadline error (504). The app is showing "
            "the retrieved course passages instead. This may be temporary; try "
            "again, and check your connection if it keeps happening."
        )
    return error


def generate_answer(prompt: str, model_name: str = "gemini-3.8-flash") -> dict:
    """
    Send the prompt to the Gemini API and return the generated answer.

    WHY gemini-3.8-flash?
    ----------------------
    Gemini reports that gemini-2.5-flash is no longer available to new users
    and recommends gemini-3.8-flash for new requests.

    Parameters
    ----------
    prompt     : The full prompt string (system instructions + context + question).
    model_name : The Gemini model variant to use.

    Returns
    -------
    A dict with:
      "answer"      : The generated text answer.
      "model"       : The model used.
      "success"     : True if generation succeeded.
      "error"       : Error message string if success is False.
      "mode"        : "generated" or "retrieval_only" or "error".
    """
    api_key = get_api_key()

    if not api_key:
        return {
            "answer": None,
            "model": None,
            "success": False,
            "error": "GEMINI_API_KEY not found in environment. See .env.example for setup instructions.",
            "mode": "no_key",
        }

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        interaction = client.interactions.create(
            model=model_name,
            input=prompt,
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 2048,
            },
            store=False,
        )

        answer_text = interaction.output_text
        if not answer_text:
            raise ValueError("Gemini returned no text in its response.")

        return {
            "answer": answer_text,
            "model": model_name,
            "success": True,
            "error": None,
            "mode": "generated",
        }

    except ImportError:
        return {
            "answer": None,
            "model": None,
            "success": False,
            "error": "google-genai is not installed. Run: python -m pip install -r requirements.txt",
            "mode": "error",
        }
    except Exception as e:
        raw_error = str(e)
        error_msg = format_generation_error(raw_error)

        # Detect common API errors and give the user actionable messages
        if error_msg == raw_error:
            if "API_KEY_INVALID" in error_msg or "API key not valid" in error_msg:
                error_msg = "The API key is invalid. Please check your GEMINI_API_KEY in the .env file."
            elif "QUOTA_EXCEEDED" in error_msg or "quota" in error_msg.lower():
                error_msg = "API quota exceeded. The free tier has been used up. Try again tomorrow."
            elif "RESOURCE_EXHAUSTED" in error_msg:
                error_msg = "Rate limit reached. Please wait a moment before asking another question."

        return {
            "answer": None,
            "model": model_name,
            "success": False,
            "error": error_msg,
            "mode": "error",
        }


def format_retrieval_only_response(
    retrieval_result: dict,
    generation_error: Optional[str] = None,
) -> str:
    """
    Show the strongest course passage when an AI-generated response is unavailable.

    The passage is displayed verbatim, not presented as a generated explanation.

    Parameters
    ----------
    retrieval_result : The dict returned by retrieve() in retrieval.py.
    generation_error : Optional reason Gemini generation was unavailable.

    Returns
    -------
    A clearly labelled answer from the most relevant retrieved course passage.
    """
    chunks = retrieval_result.get("chunks", [])

    if not chunks:
        return "No relevant passages were found in the course materials."

    chunk = chunks[0]
    source = chunk.get("source", "Unknown")
    page = chunk.get("page")
    citation = f"{source}, page {page}" if page else source
    passage = chunk.get("text", "").strip()
    if not passage:
        return "No readable text was found in the retrieved course passage."
    quoted_passage = "\n> ".join(passage.splitlines())

    if generation_error:
        heading = (
            "**AI-generated explanation unavailable.** Showing the closest "
            "passage from your course notes verbatim:"
        )
    else:
        heading = (
            "**AI-generated explanation unavailable because no Gemini API key "
            "is configured.** Showing the closest passage from your course "
            "notes verbatim:"
        )

    return f"{heading}\n\n> {quoted_passage}\n\n*Source: {citation}*"
