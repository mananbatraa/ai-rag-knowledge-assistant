"""Split pages into overlapping, paragraph-aligned chunks that keep their source."""
import re
from dataclasses import dataclass

from .loaders import Page


@dataclass
class Chunk:
    id: int
    source: str
    page: int | None
    text: str

    @property
    def citation(self) -> str:
        return f"{self.source}, p. {self.page}" if self.page else self.source


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    """Break an oversized paragraph on sentence boundaries (hard-split as a last resort)."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    parts, current = [], ""
    for s in sentences:
        while len(s) > max_chars:
            parts.append(s[:max_chars])
            s = s[max_chars:]
        if current and len(current) + len(s) + 1 > max_chars:
            parts.append(current)
            current = s
        else:
            current = f"{current} {s}".strip()
    if current:
        parts.append(current)
    return parts


def chunk_pages(pages: list[Page], max_chars: int = 700, overlap: int = 1) -> list[Chunk]:
    """Pack paragraphs into chunks of up to `max_chars`, repeating the last
    `overlap` paragraph(s) at the start of the next chunk so answers that span
    a boundary are still retrievable."""
    chunks: list[Chunk] = []
    for page in pages:
        paragraphs = []
        for para in re.split(r"\n\s*\n", page.text):
            para = " ".join(para.split())
            if para:
                paragraphs.extend(_split_long(para, max_chars))
        window: list[str] = []
        for para in paragraphs:
            if window and sum(len(p) + 1 for p in window) + len(para) > max_chars:
                chunks.append(Chunk(len(chunks), page.source, page.page, "\n".join(window)))
                window = window[-overlap:] if overlap else []
                if sum(len(p) + 1 for p in window) + len(para) > max_chars:
                    window = []
            window.append(para)
        if window:
            chunks.append(Chunk(len(chunks), page.source, page.page, "\n".join(window)))
    return chunks
