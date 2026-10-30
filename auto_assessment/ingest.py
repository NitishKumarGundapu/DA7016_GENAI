"""Read text directly; send PDFs and images unchanged to OpenRouter for OCR."""
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


def extract_document(path: str, settings: Settings) -> str:
    """Return one complete document transcript, preserving its original language."""
    document = Path(path)
    ext = document.suffix.lower()
    if ext in TEXT_EXT:
        text = document.read_text(encoding="utf-8")
    else:
        if ext != ".pdf" and ext not in IMAGE_MIME_TYPES:
            raise ValueError(f"Unsupported document type: {document.name}")
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required for PDF/image OCR, including with Ollama grading.")
        raw = document.read_bytes()
        if not raw:
            raise ValueError(f"{document.name}: document is empty.")
        mime_type = "application/pdf" if ext == ".pdf" else IMAGE_MIME_TYPES[ext]
        data_url = f"data:{mime_type};base64,{base64.b64encode(raw).decode('ascii')}"
        attachment = (
            {"type": "file", "file": {"filename": document.name, "file_data": data_url}}
            if ext == ".pdf" else
            {"type": "image_url", "image_url": {"url": data_url}}
        )
        payload = {
            "model": settings.openrouter_vision_model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": TRANSCRIBE_SYSTEM},
                {"role": "user", "content": [
                    {"type": "text", "text": "Extract all text accurately, in original page order. Return only the transcription."},
                    attachment,
                ]},
            ],
        }
        if ext == ".pdf":
            payload["plugins"] = [{"id": "file-parser", "pdf": {"engine": "mistral-ocr"}}]
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=(10, 180),
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
            raise ValueError(f"{document.name}: OCR response is missing transcription or completion status.") from exc
        if finish_reason != "stop":
            raise ValueError(f"{document.name}: OCR did not complete (finish_reason={finish_reason!r}).")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{document.name}: no readable text extracted.")
    return text
