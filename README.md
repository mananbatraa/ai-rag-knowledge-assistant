# AI-Powered RAG Knowledge Assistant

Upload documents, ask questions, get answers that cite the passages they came from. Questions the documents can't answer get "I couldn't find that" instead of a made-up answer.

Built with Python, Streamlit, scikit-learn (TF-IDF), optional sentence embeddings, and Claude (Anthropic API) for answer generation.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.10+
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # optional; without it the app shows the best passages instead
streamlit run app.py
```

Command line:

```bash
python -m rag.cli ask sample_docs "Can I get a refund on an annual plan?"
python -m rag.cli ask path/to/your/docs          # interactive
```

Tests and evaluation:

```bash
pip install -r requirements-dev.txt
pytest
python eval/run_eval.py            # retrieval metrics, offline
python eval/run_eval.py --llm      # also grades Claude's answers (makes API calls)
```

## How it works

```
upload ─► load (PDF/DOCX/TXT/MD) ─► chunk ─► vector index ─► top-k passages ─► Claude ─► answer + citations
                rag/loaders.py      rag/chunking.py  rag/retrieval.py              rag/answer.py
```

- **Loading:** PDFs are read page by page so citations can say `file.pdf, p. 2`.
- **Chunking:** paragraphs are packed into chunks of up to 700 characters. The last paragraph of each chunk is repeated at the start of the next, so an answer spanning a boundary can still be found. Oversized paragraphs are split on sentence boundaries.
- **Retrieval:** a small vector index with cosine similarity. Two backends:
  - **TF-IDF** (default): unigrams and bigrams with a light stemmer (so *refund / refunds / refunded* match). No downloads, fully offline.
  - **Sentence embeddings** (`pip install sentence-transformers`, model `all-MiniLM-L6-v2`): matches meaning rather than words.
  - The index can be saved and reloaded (`VectorIndex.save/load`).
- **Generation:** Claude (`claude-opus-5-5`, low effort for fast chat answers) receives the numbered passages and must return JSON matching a schema: `answerable`, `answer`, `citations`.

### Handling questions the documents can't answer
1. **Retrieval gate:** if no passage clears the similarity threshold, the app answers "not found" without calling the model at all.
2. **Model gate:** the system prompt forbids outside knowledge, and the model must set `answerable: false` when the passages don't contain the answer.
3. **Citation check:** citations must point to passages that were actually shown. An "answer" with no valid citation is replaced by "not found".

Passages are treated as untrusted content (the model is told to ignore instructions inside documents). Safety declines are retried on a fallback model by the API (`fallbacks="default"`).

## Evaluation

`eval/questions.json` has 20 questions about the sample documents: 15 answerable, each with the document that contains the answer, and 5 unanswerable.

| Metric (TF-IDF, k=4) | Result |
|---|---|
| Correct document ranked first (hit@1) | 15/15 |
| Correct document in top 4 (hit@4) | 15/15 |
| Unanswerable questions rejected by retrieval alone | 3/5 |

The two unanswerable questions that get past retrieval share words with the documents (e.g. "Lumora", "plan"), which is why the model gate exists. `python eval/run_eval.py --llm` measures the full pipeline, including how often Claude declines correctly. Those numbers aren't included here because they need an API key.

**Caveat:** this is a small set over three short documents written for the demo. It checks the pipeline works end to end, not how it performs on large, messy real-world collections.

## Limitations and next steps
- Scanned PDFs (images only) have no text layer; OCR is not included.
- The index lives in memory per session. For large collections, swap `VectorIndex` for a vector database such as Chroma, FAISS or pgvector. The interface is a single `search(query, k)` method.
- Next steps: hybrid TF-IDF + embedding search, reranking, streaming answers, and a larger eval set built from real questions.
