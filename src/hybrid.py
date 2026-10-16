import numpy as np

from src.evaluate import DEPTH


def top_ranks(scores, depth=DEPTH):
    order = np.argsort(-scores, kind="stable")[:depth]
    return order


def minmax(scores):
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def fuse_scores(bm25, dense, method, alpha, k=60):
    if method == "wsum":
        return alpha * minmax(bm25) + (1 - alpha) * minmax(dense)
    if method == "rrf":
        fused = np.zeros_like(dense)
        for weight, scores in ((alpha, bm25), (1 - alpha, dense)):
            for rank, doc in enumerate(top_ranks(scores), start=1):
                if scores[doc] > 0:
                    fused[doc] += weight / (k + rank)
        return fused
    raise ValueError(f"unknown method: {method}")


class Hybrid:
    def __init__(self, bm25, dense, doc_ids, method="wsum", alpha=0.5, k=60):
        self.bm25 = bm25
        self.dense = dense
        self.doc_ids = doc_ids
        self.position = {d: i for i, d in enumerate(doc_ids)}
        self.method = method
        self.alpha = alpha
        self.k = k
        self.cache = {}

    def component_scores(self, query):
        if query not in self.cache:
            sparse = np.zeros(len(self.doc_ids))
            for doc, value in self.bm25.scores(self.bm25.index.tokenizer(query)).items():
                sparse[self.position[doc]] = value
            self.cache[query] = (sparse, self.dense.scores(query))
        return self.cache[query]

    def system(self):
        def rank(query):
            sparse, dense = self.component_scores(query)
            fused = fuse_scores(sparse, dense, self.method, self.alpha, self.k)
            return [self.doc_ids[i] for i in top_ranks(fused)]

        return rank
