import pandas as pd
from sentence_transformers import SentenceTransformer

from src.data import DATA_DIR, load_corpus, load_qrels, load_queries
from src.dense import DenseSearch, encode_corpus
from src.evaluate import bootstrap_ci
from src.hybrid import Hybrid
from src.keyword import build_bm25
from src.metrics import ndcg_at_k, recall_at_k
from src.score_dense import BGE_PREFIX, DEFAULT_MODEL
from src.splits import load_splits
from src.train import MODELS_DIR

BGE = "BAAI/bge-base-en-v1.5"
FT_RUN = "all-MiniLM-L6-v2_hard_s0"
FT_EPOCH = 3
KS = [10, 20, 50, 100]
UNION_EACH = 50
OUT = DATA_DIR.parent / "ceiling_val_per_query.csv"


def oracle_ranking(candidates, relevant):
    hits = [d for d in candidates if d in relevant]
    return hits + [d for d in candidates if d not in relevant]


def union_candidates(first, second, n_each):
    head = list(first[:n_each])
    return head + [d for d in second[:n_each] if d not in set(head)]


def candidate_metrics(candidates, relevant):
    ordered = oracle_ranking(candidates, relevant)
    return {
        "recall": recall_at_k(candidates, relevant, len(candidates)),
        "any_hit": float(any(d in relevant for d in candidates)),
        "oracle_ndcg@10": ndcg_at_k(ordered, relevant, 10),
    }


def main():
    corpus = load_corpus()
    queries = load_queries()
    qrels = load_qrels("train")
    val = load_splits().val

    bm25 = build_bm25(corpus)
    rankers = {"stemmed BM25": bm25.system()}
    dense = {}
    for name, model_name, prefix, path in [
        ("MiniLM pretrained", DEFAULT_MODEL, "", None),
        ("MiniLM fine-tuned", None, "", MODELS_DIR / FT_RUN),
        ("bge-base", BGE, BGE_PREFIX, None),
    ]:
        model = SentenceTransformer(str(path or model_name), device="cpu")
        cache_name = f"ft_{FT_RUN}_epoch{FT_EPOCH}" if path else model_name
        doc_ids, emb, _ = encode_corpus(model, corpus, cache_name)
        dense[name] = DenseSearch(model, doc_ids, emb, prefix)
        rankers[name] = dense[name].system()
    rankers["hybrid BM25 + bge-base"] = Hybrid(bm25, dense["bge-base"], doc_ids, "wsum", 0.2).system()

    rows = []
    for qid in val:
        relevant = set(qrels[qid])
        ranked = {name: r(queries[qid]) for name, r in rankers.items()}
        for name, ranking in ranked.items():
            for k in KS:
                rows.append({"qid": qid, "system": name, "k": k, **candidate_metrics(ranking[:k], relevant)})
        both = union_candidates(ranked["stemmed BM25"], ranked["bge-base"], UNION_EACH)
        rows.append({"qid": qid, "system": f"BM25 + bge-base, top {UNION_EACH} each", "k": "up to 100", "size": len(both), **candidate_metrics(both, relevant)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    def cell(values):
        mean, lo, hi = bootstrap_ci(values)
        return f"{mean:.3f} ({lo:.3f} to {hi:.3f})"

    for metric, title in (("recall", "recall of the candidates"), ("any_hit", "share of queries with a relevant document among the candidates"), ("oracle_ndcg@10", "nDCG@10 of a perfect re-ranker")):
        print(f"\n{title}")
        for (name, k), part in df.groupby(["system", "k"], sort=False):
            print(f"  {name:36s} k={k!s:9s} {cell(part.sort_values('qid')[metric])}")
    sizes = df["size"].dropna()
    print(f"\nunion candidate set size: mean {sizes.mean():.1f}, range {sizes.min():.0f} to {sizes.max():.0f}")


if __name__ == "__main__":
    main()
