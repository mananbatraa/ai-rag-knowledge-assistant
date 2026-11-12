"""Streamlit UI: upload documents, ask questions, see answers with their sources.

    streamlit run app.py
"""
from pathlib import Path

import streamlit as st

from rag.answer import Answerer, make_client
from rag.chunking import chunk_pages
from rag.loaders import SUPPORTED, load_bytes, load_path
from rag.retrieval import VectorIndex, make_embedder

SAMPLE_DIR = Path(__file__).parent / "sample_docs"

st.set_page_config(page_title="RAG Knowledge Assistant", page_icon="📚", layout="wide")
st.title("📚 RAG Knowledge Assistant")
st.caption("Ask questions about your documents. Answers cite the passages they come from.")

client = make_client()

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Upload PDF, DOCX, TXT or Markdown files",
                               type=[s.lstrip(".") for s in sorted(SUPPORTED)],
                               accept_multiple_files=True)
    use_samples = st.checkbox("Use the sample documents", value=not uploads)
    retriever = st.selectbox("Retriever", ["tfidf", "embeddings"],
                             help="'embeddings' needs `pip install sentence-transformers`.")
    k = st.slider("Passages per question", 2, 8, 4)
    st.divider()
    if client:
        st.success("LLM: Claude connected")
    else:
        st.warning("LLM: off. Set ANTHROPIC_API_KEY for generated answers. "
                   "Until then the app shows the most relevant passages.")


@st.cache_resource(show_spinner="Indexing documents…")
def build_index(files: tuple, samples: bool, retriever: str) -> VectorIndex:
    pages = load_path(SAMPLE_DIR) if samples else []
    for name, data in files:
        pages.extend(p for p in load_bytes(name, data) if p.text.strip())
    return VectorIndex(make_embedder(retriever)).build(chunk_pages(pages))


files = tuple((f.name, f.getvalue()) for f in uploads or [])
if not files and not use_samples:
    st.info("Upload documents or tick “Use the sample documents” to begin.")
    st.stop()

try:
    index = build_index(files, use_samples, retriever)
except ImportError:
    st.error("The embeddings retriever needs `pip install sentence-transformers`.")
    st.stop()
except ValueError as exc:
    st.error(str(exc))
    st.stop()

sources = sorted({c.source for c in index.chunks})
st.sidebar.write(f"Indexed **{len(index.chunks)}** chunks from {len(sources)} file(s):")
st.sidebar.write("\n".join(f"- {s}" for s in sources))

answerer = Answerer(index, client=client, k=k)
if "history" not in st.session_state:
    st.session_state.history = []


def render(ans):
    st.markdown(ans.text)
    if ans.sources:
        with st.expander(f"Sources ({len(ans.sources)})"):
            for i, hit in enumerate(ans.sources, 1):
                st.markdown(f"**[{i}] {hit.chunk.citation}** · similarity {hit.score:.2f}")
                st.caption(hit.chunk.text)
    elif not ans.answerable:
        st.caption("No supporting passage in the documents, so no answer was given.")


for question, ans in st.session_state.history:
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        render(ans)

if question := st.chat_input("Ask a question about the documents"):
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching…"):
            try:
                ans = answerer.ask(question)
            except Exception as exc:  # network/auth errors shouldn't crash the app
                st.error(f"{type(exc).__name__}: {exc}")
                st.stop()
        render(ans)
    st.session_state.history.append((question, ans))
