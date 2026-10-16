import numpy as np

from src.data import load_corpus, load_qrels, load_queries
from src.evaluate import DEPTH, evaluate
from src.splits import load_splits


def doc_words(doc):
    return len((doc["title"] + " " + doc["text"]).split())


def random_system(doc_ids, seed):
    rng = np.random.default_rng(seed)

    def system(query):
        return [doc_ids[i] for i in rng.permutation(len(doc_ids))[:DEPTH]]

    return system


def shortest_first_system(corpus):
    ranking = sorted(corpus, key=lambda d: (doc_words(corpus[d]), d))[:DEPTH]

    def system(query):
        return ranking

    return system


def main():
    corpus = load_corpus()
    queries = load_queries()
    qrels = load_qrels("train")
    val = load_splits().val
    doc_ids = sorted(corpus)

    print(f"validation queries: {len(val)}")
    for seed in range(5):
        result = evaluate(random_system(doc_ids, seed), queries, qrels, val)
        print(f"random, seed {seed}\n{result.summary.round(4)}\n")
    result = evaluate(shortest_first_system(corpus), queries, qrels, val)
    print(f"shortest first\n{result.summary.round(4)}")


if __name__ == "__main__":
    main()
