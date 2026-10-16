import numpy as np
import pytest

from src.hybrid import Hybrid, fuse_scores, minmax, top_ranks

BM25 = np.array([4.0, 0.0, 2.0, 0.0])
DENSE = np.array([0.2, 0.9, 0.5, 0.1])


def order(scores):
    return list(np.argsort(-scores, kind="stable"))


def test_minmax_scales_to_unit_interval():
    assert minmax(np.array([2.0, 4.0, 6.0])) == pytest.approx([0, 0.5, 1])
    assert minmax(np.array([3.0, 3.0])) == pytest.approx([0, 0])


def test_weighted_sum_endpoints_equal_single_systems():
    assert order(fuse_scores(BM25, DENSE, "wsum", 1.0)) == order(BM25)
    assert order(fuse_scores(BM25, DENSE, "wsum", 0.0)) == order(DENSE)


def test_weighted_sum_by_hand():
    expected = 0.5 * np.array([1, 0, 0.5, 0]) + 0.5 * np.array([0.125, 1, 0.5, 0])
    assert fuse_scores(BM25, DENSE, "wsum", 0.5) == pytest.approx(expected)


def test_rrf_by_hand():
    fused = fuse_scores(BM25, DENSE, "rrf", 0.5, k=60)
    bm25_ranks = {0: 1, 2: 2}
    dense_ranks = {1: 1, 2: 2, 0: 3, 3: 4}
    for doc in range(4):
        expected = 0.5 / (60 + dense_ranks[doc])
        if doc in bm25_ranks:
            expected += 0.5 / (60 + bm25_ranks[doc])
        assert fused[doc] == pytest.approx(expected)


def test_rrf_endpoints_follow_single_systems():
    assert order(fuse_scores(BM25, DENSE, "rrf", 1.0))[:2] == [0, 2]
    assert order(fuse_scores(BM25, DENSE, "rrf", 0.0)) == order(DENSE)


def test_rrf_only_counts_documents_inside_the_depth():
    bm25 = np.zeros(300)
    bm25[0] = 1.0
    dense = np.arange(300, dtype=float)
    fused = fuse_scores(bm25, dense, "rrf", 0.5, k=60)
    assert fused[1] == 0.0
    assert fused[299] > fused[200] > 0


def test_unknown_method():
    with pytest.raises(ValueError):
        fuse_scores(BM25, DENSE, "nope", 0.5)


def test_top_ranks_breaks_ties_by_index():
    assert list(top_ranks(np.array([1.0, 1.0, 2.0]), depth=3)) == [2, 0, 1]


class StubBM25:
    class index:
        tokenizer = staticmethod(lambda q: q.split())

    def scores(self, tokens):
        return {"a": 4.0, "c": 2.0}


class StubDense:
    def scores(self, query):
        return DENSE


def test_hybrid_system_returns_document_ids():
    hybrid = Hybrid(StubBM25(), StubDense(), ["a", "b", "c", "d"], method="wsum", alpha=0.0)
    assert hybrid.system()("q")[:2] == ["b", "c"]
    hybrid.alpha = 1.0
    assert hybrid.system()("q")[:2] == ["a", "c"]
