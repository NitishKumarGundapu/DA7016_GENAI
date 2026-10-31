"""Document ingestion.

Plain ``.txt``/``.md`` files are read locally. PDFs and images are sent unchanged
to OpenRouter's OCR/vision endpoint, because the pipeline deliberately avoids any
local PDF rendering or image processing.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path

import requests

from .config import Settings
from .prompts import TRANSCRIBE_SYSTEM

IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}
TEXT_EXT = {".txt", ".md"}

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OCR_TIMEOUT = (10, 180)  # (connect seconds, read seconds)
OCR_INSTRUCTIONS = (
    "Extract all text accurately, in original page order. Return only the "
    "transcription. Convert diagrams, graphs and tables into detailed "
    "structured text. Omit crossed-out/struck-through text and cut-off, "
    "cropped or partial fragments as described above; never guess or "
    "reconstruct them."
)


def _read_text(document: Path) -> str:
    """Read a UTF-8 text document."""
    return document.read_text(encoding="utf-8")


def _encode_attachment(document: Path, extension: str) -> dict:
    """Return an OpenRouter content part carrying the original file bytes."""
    raw = document.read_bytes()
    if not raw:
        raise ValueError(f"{document.name}: document is empty.")

    mime_type = "application/pdf" if extension == ".pdf" else IMAGE_MIME_TYPES[extension]
    data_url = f"data:{mime_type};base64,{base64.b64encode(raw).decode('ascii')}"
    if extension == ".pdf":
        return {"type": "file", "file": {"filename": document.name, "file_data": data_url}}
    return {"type": "image_url", "image_url": {"url": data_url}}


def _build_ocr_request(
    document: Path, extension: str, settings: Settings, attachment: dict
) -> dict:
    """Build the chat-completions payload for one PDF or image."""
    payload = {
        "model": settings.openrouter_vision_model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": TRANSCRIBE_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_INSTRUCTIONS},
                    attachment,
                ],
            },
        ],
    }
    payload["reasoning"] = {"enabled": settings.extraction_reasoning}
    if extension == ".pdf":
        payload["plugins"] = [{"id": "file-parser", "pdf": {"engine": "mistral-ocr"}}]
    return payload


def _ocr_document(document: Path, extension: str, settings: Settings) -> str:
    """Send one PDF/image to OpenRouter and return the validated transcription."""
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is required for PDF/image OCR, including with Ollama grading."
        )

    attachment = _encode_attachment(document, extension)
    payload = _build_ocr_request(document, extension, settings, attachment)
    response = requests.post(
        OPENROUTER_CHAT_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=OCR_TIMEOUT,
    )
    response.raise_for_status()
    result = response.json()

    if not isinstance(result, dict):
        raise ValueError(f"{document.name}: invalid OCR response.")
    if result.get("error"):
        raise RuntimeError(f"{document.name}: OpenRouter OCR error: {result['error']}")
    try:
        choice = result["choices"][0]
        text = choice["message"]["content"]
        finish_reason = choice["finish_reason"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            f"{document.name}: OCR response is missing transcription or completion status."
        ) from exc
    if finish_reason != "stop":
        raise ValueError(
            f"{document.name}: OCR did not complete (finish_reason={finish_reason!r})."
        )
    return text


def extract_document(path: str, settings: Settings) -> str:
    """Return one complete document transcript, preserving its original language."""
    document = Path(path)
    extension = document.suffix.lower()

    if extension in TEXT_EXT:
        text = _read_text(document)
    elif extension == ".pdf" or extension in IMAGE_MIME_TYPES:
        text = _ocr_document(document, extension, settings)
    else:
        raise ValueError(f"Unsupported document type: {document.name}")

    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{document.name}: no readable text extracted.")
    return text
