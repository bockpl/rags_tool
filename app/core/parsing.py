"""Text parsing and extraction utilities."""

from __future__ import annotations

import html
import logging
import pathlib
import re
from typing import List

import html2text
import markdown2

try:
    import PyPDF2  # type: ignore
    HAS_PDF = True
except Exception:  # pragma: no cover - optional dependency
    HAS_PDF = False

logger = logging.getLogger("rags_tool")

# Supported file extensions for ingest
SUPPORTED_EXT = {".txt", ".md", ".markdown", ".html", ".htm", ".pdf"}


def read_text_file(path: pathlib.Path) -> str:
    ext = path.suffix.lower()
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin1", errors="ignore")
    if ext in {".md", ".markdown"}:
        try:
            html_content = markdown2.markdown(text)
            text = html_to_text(html_content)
        except Exception:
            pass
    elif ext in {".html", ".htm"}:
        try:
            text = html_to_text(text)
        except Exception:
            pass
    return text


def html_to_text(content_html: str) -> str:
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True
    h.ignore_emphasis = False
    h.bypass_tables = False
    content_html = html.unescape(content_html)
    return h.handle(content_html)


def extract_text(path: pathlib.Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf" and HAS_PDF:
        try:
            with path.open("rb") as f:
                rdr = PyPDF2.PdfReader(f)
                pages = [p.extract_text() or "" for p in rdr.pages]
                return "\n\n".join(pages)
        except Exception as exc:  # fallback to bytes decode
            logger.debug("PDF extract failed for %s: %s", path, exc)
    return read_text_file(path)


def split_into_paragraphs(text: str) -> List[str]:
    # Normalize newlines, split on blank lines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    raw = re.split(r"\n\s*\n", text)
    paras = [p.strip() for p in raw if p and p.strip()]
    return paras

