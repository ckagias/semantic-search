import math


def _top_k(ranking, relevant, k):
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        raise ValueError("query has no relevant documents")
    top = list(ranking[:k])
    if len(set(top)) != len(top):
        raise ValueError("ranking contains a document more than once")
    return top


def recall_at_k(ranking, relevant, k):
    top = _top_k(ranking, relevant, k)
    return len(set(top) & set(relevant)) / len(relevant)


def mrr_at_k(ranking, relevant, k=10):
    top = _top_k(ranking, relevant, k)
    for position, doc in enumerate(top, start=1):
        if doc in relevant:
            return 1 / position
    return 0.0


def ndcg_at_k(ranking, relevant, k=10):
    top = _top_k(ranking, relevant, k)
    dcg = sum(
        1 / math.log2(position + 1)
        for position, doc in enumerate(top, start=1)
        if doc in relevant
    )
    ideal = sum(1 / math.log2(position + 1) for position in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal
