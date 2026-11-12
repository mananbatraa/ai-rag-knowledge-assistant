"""Turn uploaded files (PDF, DOCX, TXT, Markdown) into page-level text records."""
import io
from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {".pdf", ".docx", ".txt", ".md"}


@dataclass
class Page:
    source: str          # file name shown in citations
    page: int | None     # 1-based page number for PDFs, None otherwise
    text: str


def load_bytes(name: str, data: bytes) -> list[Page]:
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return [Page(name, i, (p.extract_text() or "").strip())
                for i, p in enumerate(reader.pages, 1)]
    if suffix == ".docx":
        from docx import Document

        doc = Document(io.BytesIO(data))
        text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [Page(name, None, text)]
    if suffix in {".txt", ".md"}:
        return [Page(name, None, data.decode("utf-8", errors="replace"))]
    raise ValueError(f"Unsupported file type {suffix!r}; expected one of {sorted(SUPPORTED)}")


def load_path(path: Path) -> list[Page]:
    path = Path(path)
    files = [path] if path.is_file() else sorted(
        p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED)
    pages = []
    for f in files:
        pages.extend(load_bytes(f.name, f.read_bytes()))
    return [p for p in pages if p.text.strip()]
