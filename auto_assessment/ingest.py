"""File loading: text documents, PDFs and handwritten scans -> model-ready inputs."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
TEXT_EXT = {".txt", ".md"}


def read_text_document(path: str) -> str:
    """Question paper / rubric / model answers -> plain text (txt, md, json, text-based pdf)."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in TEXT_EXT or ext == ".json":
        return p.read_text(encoding="utf-8", errors="ignore")
    if ext == ".pdf":
        import fitz  # pymupdf

        with fitz.open(p) as doc:
            text = "\n\n".join(page.get_text() for page in doc)
        if len(text.strip()) < 20:
            raise ValueError(
                f"{p.name} looks like a scanned PDF with no text layer. "
                "Provide a text PDF / .txt / .md, or a structured .json spec."
            )
        return text
    raise ValueError(f"Unsupported document type: {p.name}")


def load_spec_json(path: str) -> dict | None:
    """If the file is a JSON exam spec, return it as a dict, else None."""
    p = Path(path)
    if p.suffix.lower() != ".json":
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _encode_pil(img, max_side: int) -> str:
    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(img).convert("RGB")  # phone photos are often rotated
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def load_page_images(paths: list[str], dpi: int = 200, max_side: int = 1800) -> list[dict]:
    """Answer-sheet files (images and/or PDFs) -> ordered list of page dicts with data URLs."""
    from PIL import Image

    pages: list[dict] = []
    for path in paths:
        p = Path(path)
        ext = p.suffix.lower()
        if ext in IMAGE_EXT:
            with Image.open(p) as im:
                pages.append({"page": len(pages) + 1, "source": p.name, "data_url": _encode_pil(im, max_side)})
        elif ext == ".pdf":
            import fitz

            with fitz.open(p) as doc:
                for pg in doc:
                    pix = pg.get_pixmap(dpi=dpi)
                    im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    pages.append(
                        {"page": len(pages) + 1, "source": f"{p.name}#p{pg.number + 1}",
                         "data_url": _encode_pil(im, max_side)}
                    )
        elif ext in TEXT_EXT:
            continue  # handled as already-typed text
        else:
            raise ValueError(f"Unsupported answer-sheet file: {p.name}")
    return pages


def load_typed_text(paths: list[str]) -> str:
    """Already-typed answer files (.txt/.md) - lets you test the pipeline without scans."""
    return "\n\n".join(Path(p).read_text(encoding="utf-8") for p in paths if Path(p).suffix.lower() in TEXT_EXT)
