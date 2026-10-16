from src.data import DATA_DIR, load_corpus, load_qrels, load_queries
from src.evaluate import evaluate
from src.keyword import BM25_TUNED, build_bm25
from src.splits import load_splits

OUT = DATA_DIR.parent / "bm25_val_per_query.csv"


def main():
    corpus = load_corpus()
    queries = load_queries()
    qrels = load_qrels("train")
    val = load_splits().val
    system = build_bm25(corpus).system()

    result = evaluate(system, queries, qrels, val)
    result.per_query.to_csv(OUT)
    print(f"BM25 {BM25_TUNED}, {len(val)} validation queries")
    print(result.summary.round(3))
    print(f"queries with no relevant document in the top 100: {(result.per_query['recall@100'] == 0).sum()}")
    print(f"queries with the relevant document at rank 1: {(result.per_query['mrr@10'] == 1).sum()}")


if __name__ == "__main__":
    main()
