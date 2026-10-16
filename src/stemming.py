import itertools

from src.data import load_corpus, load_qrels, load_queries
from src.evaluate import bootstrap_ci, evaluate
from src.index import InvertedIndex, tokenize, tokenize_stemmed
from src.keyword import BM25
from src.splits import load_splits

K1_GRID = [1.2, 1.8, 2.1, 2.5, 3.0]
B_GRID = [0.3, 0.5, 0.75, 1.0]


def best_setting(index, queries, qrels, val):
    best = None
    for k1, b in itertools.product(K1_GRID, B_GRID):
        result = evaluate(BM25(index, k1=k1, b=b, idf="lucene").system(), queries, qrels, val, n_boot=10)
        score = result.per_query["ndcg@10"].mean()
        if best is None or score > best[0]:
            best = (score, k1, b, result.per_query)
    return best


def main():
    corpus = load_corpus()
    queries = load_queries()
    qrels = load_qrels("train")
    val = load_splits().val

    found = {}
    for name, tokenizer in [("no stemming", tokenize), ("snowball stemming", tokenize_stemmed)]:
        index = InvertedIndex(corpus, tokenizer)
        score, k1, b, per_query = best_setting(index, queries, qrels, val)
        found[name] = per_query
        full = evaluate(BM25(index, k1=k1, b=b, idf="lucene").system(), queries, qrels, val)
        print(f"{name}: {len(index.postings)} terms, best k1 {k1}, b {b}")
        print(full.summary.round(3), "\n")

    diff = found["snowball stemming"]["ndcg@10"] - found["no stemming"]["ndcg@10"]
    mean, lo, hi = bootstrap_ci(diff)
    print(f"nDCG@10 stemming minus no stemming (paired): {mean:.3f} ({lo:.3f} to {hi:.3f})")
    print(f"queries better with stemming: {(diff > 0).sum()}, worse: {(diff < 0).sum()}")


if __name__ == "__main__":
    main()
