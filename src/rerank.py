import argparse
import time

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.data import DATA_DIR, load_corpus, load_qrels, load_queries
from src.dense import DenseSearch, doc_text, encode_corpus
from src.evaluate import bootstrap_ci, evaluate
from src.hybrid import Hybrid
from src.keyword import build_bm25
from src.score_dense import BGE_PREFIX
from src.splits import load_splits

BGE = "BAAI/bge-base-en-v1.5"
CROSS_ENCODERS = ["cross-encoder/ms-marco-MiniLM-L6-v2", "BAAI/bge-reranker-base"]
CANDIDATES = 50
KS = [20, 50]
TIMED_QUERIES = 30
OUT = DATA_DIR.parent / "rerank_val_per_query.csv"


def rerank(first_stage, scores, k):
    order = np.argsort(-np.asarray(scores[:k]), kind="stable")
    return [first_stage[i] for i in order] + list(first_stage[k:])


def score_candidates(model, query, first_stage, corpus, batch_size=16):
    pairs = [(query, doc_text(corpus[d])) for d in first_stage[:CANDIDATES]]
    return model.predict(pairs, batch_size=batch_size, show_progress_bar=False)


def main(models, n_queries):
    corpus = load_corpus()
    texts = load_queries()
    qrels = load_qrels("train")
    val = load_splits().val[:n_queries]

    bm25 = build_bm25(corpus)
    bge = SentenceTransformer(BGE, device="cpu")
    doc_ids, emb, _ = encode_corpus(bge, corpus, BGE)
    fused = Hybrid(bm25, DenseSearch(bge, doc_ids, emb, BGE_PREFIX), doc_ids, "wsum", 0.2)
    hybrid = fused.system()
    first = {texts[q]: hybrid(texts[q]) for q in val}
    base = evaluate(lambda q: first[q], texts, qrels, val)
    print(f"first stage (hybrid): nDCG@10 {base.per_query['ndcg@10'].mean():.3f}, recall@10 {base.per_query['recall@10'].mean():.3f}")

    timing = []
    for t in val[:TIMED_QUERIES]:
        fused.cache.clear()
        s = time.perf_counter()
        hybrid(texts[t])
        timing.append(time.perf_counter() - s)
    print(f"first stage latency (BM25 + bge-base query encoding + scoring): median {1000 * np.median(timing):.0f} ms\n")

    rows = []
    for name in models:
        model = CrossEncoder(name, device="cpu")
        score_candidates(model, texts[val[0]], first[texts[val[0]]], corpus)
        scores = {}
        for q in val:
            scores[texts[q]] = score_candidates(model, texts[q], first[texts[q]], corpus)
        for k in KS:
            result = evaluate(lambda q, k=k: rerank(first[q], scores[q], k), texts, qrels, val)
            diff = result.per_query["ndcg@10"] - base.per_query["ndcg@10"]
            mean, lo, hi = bootstrap_ci(diff)
            s = result.summary
            row = {"cross-encoder": name, "candidates": k, "ndcg@10": s.loc["ndcg@10", "mean"], "ndcg_lo": s.loc["ndcg@10", "lo"], "ndcg_hi": s.loc["ndcg@10", "hi"], "mrr@10": s.loc["mrr@10", "mean"], "recall@10": s.loc["recall@10", "mean"], "recall@100": s.loc["recall@100", "mean"], "diff": mean, "diff_lo": lo, "diff_hi": hi, "better": int((diff > 0).sum()), "worse": int((diff < 0).sum())}
            rows.append(row)
            print({a: (round(b, 3) if isinstance(b, float) else b) for a, b in row.items()})
        for batch_size in (16, 50):
            times = []
            for q in val[:TIMED_QUERIES]:
                s = time.perf_counter()
                score_candidates(model, texts[q], first[texts[q]], corpus, batch_size)
                times.append(time.perf_counter() - s)
            print(f"{name}: re-scoring {CANDIDATES} candidates, batch {batch_size}: median {1000 * np.median(times):.0f} ms, p95 {1000 * np.quantile(times, 0.95):.0f} ms")
            rows.append({"cross-encoder": name, "candidates": CANDIDATES, "batch size": batch_size, "median ms": 1000 * np.median(times), "p95 ms": 1000 * np.quantile(times, 0.95)})
    if n_queries is None:
        pd.DataFrame(rows).to_csv(OUT, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=CROSS_ENCODERS[:1])
    parser.add_argument("--queries", type=int, default=None, help="use only the first N validation queries")
    args = parser.parse_args()
    main(args.models, args.queries)
