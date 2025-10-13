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


# Read text file and normalize basic markup (md/html) to plain text.
def read_text_file(path: pathlib.Path) -> str:
    """Read text from a file with light format normalization.

    - Decodes bytes as UTF-8 (fallback Latin-1 ignoring errors).
    - For Markdown and HTML, converts to plain text via markdown2/html2text.
    """
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


# Convert HTML to plain text with conservative settings.
def html_to_text(content_html: str) -> str:
    """Convert HTML to readable text using html2text with conservative options."""
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True
    h.ignore_emphasis = False
    h.bypass_tables = False
    content_html = html.unescape(content_html)
    return h.handle(content_html)


# Extract text from supported formats (txt/md/html/pdf when available).
def extract_text(path: pathlib.Path) -> str:
    """Extract textual content from supported file types (txt, md, html, pdf).

    Uses PyPDF2 for PDFs when available; otherwise falls back to bytes decoding
    and post-processing as in `read_text_file`.
    """
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


# Split text into paragraphs using blank lines as separators.
def split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs using blank lines as separators."""
    # Normalize newlines, split on blank lines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    raw = re.split(r"\n\s*\n", text)
    paras = [p.strip() for p in raw if p and p.strip()]
    return paras
