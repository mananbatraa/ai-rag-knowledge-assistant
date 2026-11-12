import json
from types import SimpleNamespace

import pytest

from rag.answer import NOT_FOUND, Answerer
from rag.chunking import chunk_pages
from rag.loaders import Page, load_bytes, load_path
from rag.retrieval import VectorIndex

DOCS = [
    Page("refunds.md", None, "Annual plans can be refunded within 30 days of purchase.\n\n"
                             "Refunds are paid to the original payment method within 5 to 10 business days."),
    Page("security.pdf", 2, "Customer data is encrypted at rest with AES-256.\n\n"
                            "Backups are kept for 35 days."),
]


class FakeClient:
    def __init__(self, payload=None, stop_reason="end_turn"):
        self.calls = []
        text = json.dumps(payload) if payload is not None else ""
        response = SimpleNamespace(stop_reason=stop_reason,
                                   content=[SimpleNamespace(type="text", text=text)])

        def create(**kw):
            self.calls.append(kw)
            return response

        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))


@pytest.fixture
def index():
    return VectorIndex().build(chunk_pages(DOCS))


def test_chunking_respects_size_and_keeps_source():
    long_page = Page("long.txt", None, "\n\n".join(f"Paragraph {i}. " + "word " * 15 for i in range(20)))
    chunks = chunk_pages([long_page], max_chars=300)
    assert len(chunks) > 5
    assert all(len(c.text) <= 300 for c in chunks)
    assert all(c.source == "long.txt" for c in chunks)
    # overlap: the last paragraph of one chunk starts the next
    assert chunks[1].text.split("\n")[0] == chunks[0].text.split("\n")[-1]


def test_citation_includes_page_for_pdfs():
    chunks = chunk_pages(DOCS)
    assert {c.citation for c in chunks} == {"refunds.md", "security.pdf, p. 2"}


def test_retrieval_finds_relevant_and_rejects_unrelated(index):
    assert index.search("how long until my refund arrives")[0].chunk.source == "refunds.md"
    assert index.search("encryption at rest")[0].chunk.source == "security.pdf"
    assert index.search("best sourdough bread recipe") == []


def test_index_save_and_load_roundtrip(index, tmp_path):
    index.save(tmp_path)
    loaded = VectorIndex.load(tmp_path)
    assert loaded.search("backups")[0].chunk.text == index.search("backups")[0].chunk.text


def test_no_match_skips_the_llm(index):
    client = FakeClient({"answerable": True, "answer": "x", "citations": [1]})
    ans = Answerer(index, client=client).ask("best sourdough bread recipe")
    assert ans.text == NOT_FOUND and ans.mode == "no_match"
    assert client.calls == []


def test_grounded_answer_keeps_only_valid_citations(index):
    client = FakeClient({"answerable": True, "answer": "Within 5-10 business days [1].",
                         "citations": [1, 1, 9]})
    ans = Answerer(index, client=client).ask("how long do refunds take")
    assert ans.answerable and len(ans.sources) == 1
    assert ans.sources[0].chunk.source == "refunds.md"
    request = client.calls[0]
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert "<passage n=\"1\"" in request["messages"][0]["content"]


def test_model_says_unanswerable(index):
    client = FakeClient({"answerable": False, "answer": "The documents don't say.", "citations": []})
    ans = Answerer(index, client=client).ask("how long are refunds for monthly plans")
    assert not ans.answerable and ans.sources == []


def test_answer_without_valid_citation_is_rejected(index):
    client = FakeClient({"answerable": True, "answer": "Made up.", "citations": [7]})
    ans = Answerer(index, client=client).ask("refund timing")
    assert not ans.answerable and ans.text == NOT_FOUND


def test_refusal_is_reported(index):
    ans = Answerer(index, client=FakeClient(stop_reason="refusal")).ask("refund timing")
    assert not ans.answerable and "declined" in ans.text


def test_extractive_mode_without_client(index):
    ans = Answerer(index, client=None).ask("refund timing")
    assert ans.mode == "extractive" and ans.sources


def test_loaders_read_pdf_and_docx(tmp_path):
    pytest.importorskip("reportlab")
    from docx import Document
    from reportlab.pdfgen import canvas

    pdf = tmp_path / "a.pdf"
    c = canvas.Canvas(str(pdf))
    c.drawString(72, 720, "Page one text")
    c.showPage()
    c.drawString(72, 720, "Page two text")
    c.save()
    pages = load_bytes("a.pdf", pdf.read_bytes())
    assert [(p.page, p.text) for p in pages] == [(1, "Page one text"), (2, "Page two text")]

    doc = Document()
    doc.add_paragraph("Hello from Word")
    doc.save(tmp_path / "b.docx")
    assert "Hello from Word" in load_path(tmp_path / "b.docx")[0].text

    with pytest.raises(ValueError):
        load_bytes("x.exe", b"")
