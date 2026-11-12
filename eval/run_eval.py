"""Measure retrieval quality and no-answer handling on eval/questions.json.

    python eval/run_eval.py                   # retrieval-only, offline, free
    python eval/run_eval.py --llm             # also grade Claude's answers (uses the API)

Retrieval metrics (answerable questions): hit@1 / hit@k = the expected document
is the top result / among the top k. No-answer metrics (unanswerable
questions): how often retrieval alone returned nothing, and with --llm how
often the final answer correctly declined.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.answer import Answerer, make_client  # noqa: E402
from rag.chunking import chunk_pages  # noqa: E402
from rag.loaders import load_path  # noqa: E402
from rag.retrieval import VectorIndex, make_embedder  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retriever", choices=["tfidf", "embeddings"], default="tfidf")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--llm", action="store_true")
    args = parser.parse_args(argv)

    cases = json.loads((ROOT / "eval" / "questions.json").read_text())
    index = VectorIndex(make_embedder(args.retriever)).build(chunk_pages(load_path(ROOT / "sample_docs")))
    client = make_client() if args.llm else None
    if args.llm and client is None:
        sys.exit("--llm needs ANTHROPIC_API_KEY (or `ant auth login`).")
    answerer = Answerer(index, client=client, k=args.k)

    answerable = [c for c in cases if c["answerable"]]
    unanswerable = [c for c in cases if not c["answerable"]]
    hit1 = hitk = rejected = 0
    llm_correct = llm_declined = 0
    misses = []
    for case in cases:
        hits = index.search(case["question"], args.k)
        sources = [h.chunk.source for h in hits]
        if case["answerable"]:
            hit1 += bool(sources) and sources[0] == case["source"]
            hitk += case["source"] in sources
            if case["source"] not in sources:
                misses.append(case["question"])
        else:
            rejected += not hits
        if client:
            ans = answerer.ask(case["question"])
            if case["answerable"]:
                llm_correct += ans.answerable and any(h.chunk.source == case["source"]
                                                      for h in ans.sources)
            else:
                llm_declined += not ans.answerable

    report = {
        "retriever": args.retriever, "k": args.k,
        "answerable_questions": len(answerable), "unanswerable_questions": len(unanswerable),
        "hit@1": round(hit1 / len(answerable), 3),
        f"hit@{args.k}": round(hitk / len(answerable), 3),
        "retrieval_rejected_unanswerable": round(rejected / len(unanswerable), 3),
        "retrieval_misses": misses,
    }
    if client:
        report["llm_answered_with_correct_source"] = round(llm_correct / len(answerable), 3)
        report["llm_declined_unanswerable"] = round(llm_declined / len(unanswerable), 3)
    print(json.dumps(report, indent=2))
    out = ROOT / "eval" / f"results_{args.retriever}{'_llm' if client else ''}.json"
    out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
