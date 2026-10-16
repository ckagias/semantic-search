import pandas as pd
from sentence_transformers import SentenceTransformer

from src.data import DATA_DIR, load_corpus, load_qrels, load_queries
from src.dense import DenseSearch, encode_corpus
from src.evaluate import bootstrap_ci, evaluate
from src.hybrid import Hybrid
from src.keyword import build_bm25
from src.score_dense import BGE_PREFIX
from src.splits import load_splits

MODEL = "BAAI/bge-base-en-v1.5"
ALPHAS = [round(0.1 * i, 1) for i in range(11)]
OUT = DATA_DIR.parent / "hybrid_grid.csv"


def main():
    corpus = load_corpus()
    queries = load_queries()
    qrels = load_qrels("train")
    val = load_splits().val

    model = SentenceTransformer(MODEL, device="cpu")
    doc_ids, emb, _ = encode_corpus(model, corpus, MODEL)
    hybrid = Hybrid(build_bm25(corpus), DenseSearch(model, doc_ids, emb, BGE_PREFIX), doc_ids)

    rows, per_query = [], {}
    settings = [("wsum", a, 60) for a in ALPHAS] + [("rrf", a, k) for k in (10, 60, 200) for a in ALPHAS]
    for method, alpha, k in settings:
        hybrid.method, hybrid.alpha, hybrid.k = method, alpha, k
        result = evaluate(hybrid.system(), queries, qrels, val, n_boot=10)
        per_query[(method, alpha, k)] = result.per_query
        rows.append({"method": method, "k": k if method == "rrf" else None, "alpha": alpha, **result.per_query.mean().to_dict()})
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT, index=False)

    pd.set_option("display.width", 200)
    for name, part in grid.groupby(["method", "k"], dropna=False):
        print(name, " ".join(f"{a:.1f}:{v:.3f}" for a, v in zip(part["alpha"], part["ndcg@10"])))

    best = grid.sort_values("ndcg@10", ascending=False).iloc[0]
    key = (best["method"], best["alpha"], 60 if pd.isna(best["k"]) else int(best["k"]))
    print(f"\nbest: {key}")
    hybrid.method, hybrid.alpha, hybrid.k = key
    full = evaluate(hybrid.system(), queries, qrels, val)
    print(full.summary.round(3))

    for name, path in [("stemmed BM25", "bm25_val_per_query.csv"), ("bge-base alone", "BAAI_bge_base_en_v1_5_val_per_query.csv")]:
        other = pd.read_csv(DATA_DIR.parent / path, index_col=0)
        other.index = other.index.astype(str)
        diff = full.per_query["ndcg@10"] - other.loc[full.per_query.index, "ndcg@10"]
        mean, lo, hi = bootstrap_ci(diff)
        print(f"nDCG@10 hybrid minus {name}: {mean:+.3f} ({lo:+.3f} to {hi:+.3f}), better on {(diff > 0).sum()}, worse on {(diff < 0).sum()}")
    full.per_query.to_csv(DATA_DIR.parent / "hybrid_val_per_query.csv")


if __name__ == "__main__":
    main()
