import itertools

import pandas as pd

from src.data import DATA_DIR, load_corpus, load_qrels, load_queries
from src.evaluate import bootstrap_ci, evaluate
from src.index import InvertedIndex
from src.keyword import BM25
from src.splits import load_splits

K1_GRID = [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.5, 3.0]
B_GRID = [round(0.1 * i, 1) for i in range(11)]
IDF_GRID = ["okapi", "lucene"]
OUT = DATA_DIR.parent / "bm25_grid.csv"


def main():
    corpus = load_corpus()
    queries = load_queries()
    qrels = load_qrels("train")
    val = load_splits().val
    index = InvertedIndex(corpus)

    rows, per_query = [], {}
    for idf, k1, b in itertools.product(IDF_GRID, K1_GRID, B_GRID):
        system = BM25(index, k1=k1, b=b, idf=idf).system()
        result = evaluate(system, queries, qrels, val, n_boot=10)
        per_query[(idf, k1, b)] = result.per_query["ndcg@10"]
        rows.append({"idf": idf, "k1": k1, "b": b, **result.per_query.mean().to_dict()})

    grid = pd.DataFrame(rows)
    grid.to_csv(OUT, index=False)
    ranked = grid.sort_values("ndcg@10", ascending=False)
    print(ranked.head(10).round(4).to_string(index=False))

    best = ranked.iloc[0]
    default = evaluate(BM25(index).system(), queries, qrels, val, n_boot=10).per_query["ndcg@10"]
    chosen = per_query[(best["idf"], best["k1"], best["b"])]
    mean, lo, hi = bootstrap_ci(chosen - default)
    print(f"\nbest vs rank_bm25 defaults, paired nDCG@10 difference: {mean:.4f} ({lo:.4f} to {hi:.4f})")
    for idf in IDF_GRID:
        part = grid[grid["idf"] == idf]
        print(f"{idf}: nDCG@10 min {part['ndcg@10'].min():.3f}, median {part['ndcg@10'].median():.3f}, max {part['ndcg@10'].max():.3f}")


if __name__ == "__main__":
    main()
