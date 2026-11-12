"""Command line: index a folder of documents and ask questions about it.

    python -m rag.cli ask sample_docs "How long do refunds take?"
    python -m rag.cli ask sample_docs            # interactive
"""
import argparse

from .answer import Answerer, make_client
from .chunking import chunk_pages
from .loaders import load_path
from .retrieval import VectorIndex, make_embedder


def print_answer(ans):
    print(f"\n{ans.text}")
    for i, hit in enumerate(ans.sources, 1):
        print(f"  [{i}] {hit.chunk.citation}  (similarity {hit.score:.2f})")
    print(f"  mode: {ans.mode}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask")
    ask.add_argument("docs", help="file or folder of PDF/DOCX/TXT/MD documents")
    ask.add_argument("question", nargs="?")
    ask.add_argument("--retriever", choices=["tfidf", "embeddings"], default="tfidf")
    ask.add_argument("--model", default=None)
    ask.add_argument("--offline", action="store_true", help="never call the LLM")
    args = parser.parse_args(argv)

    chunks = chunk_pages(load_path(args.docs))
    index = VectorIndex(make_embedder(args.retriever)).build(chunks)
    client = None if args.offline else make_client()
    kwargs = {"model": args.model} if args.model else {}
    answerer = Answerer(index, client=client, **kwargs)
    print(f"Indexed {len(chunks)} chunks. LLM: {'on' if client else 'off (extractive mode)'}")

    if args.question:
        print_answer(answerer.ask(args.question))
        return
    while True:
        try:
            q = input("\nQuestion (blank to quit): ").strip()
        except EOFError:
            break
        if not q:
            break
        print_answer(answerer.ask(q))


if __name__ == "__main__":
    main()
