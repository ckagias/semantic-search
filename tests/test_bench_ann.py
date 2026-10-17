import numpy as np
import pytest

from src.bench_ann import exact_top, overlap, synthetic_corpus


def unit(rows):
    rows = np.asarray(rows, dtype=np.float32)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def test_exact_top_matches_a_full_sort_across_chunks():
    rng = np.random.default_rng(0)
    emb, queries = unit(rng.normal(size=(500, 16))), unit(rng.normal(size=(7, 16)))
    ids, scores = exact_top(emb, queries, 5, chunk=64)
    for q, got, got_scores in zip(queries, ids, scores):
        order = np.argsort(-(emb @ q))[:5]
        assert list(got) == list(order)
        assert got_scores == pytest.approx((emb @ q)[order], abs=1e-6)


def test_overlap_counts_shared_documents_per_query():
    found = np.array([[1, 2, 3], [4, 5, 6]])
    truth = np.array([[3, 2, 9], [7, 8, 9]])
    assert overlap(found, truth) == pytest.approx((2 / 3 + 0) / 2)


def test_overlap_by_source_document_ignores_which_copy_was_returned():
    found = np.array([[5 + 10, 7 + 20]])
    truth = np.array([[5, 7 + 10]])
    assert overlap(found, truth) == 0.0
    assert overlap(found, truth, 10) == 1.0


def test_synthetic_corpus_keeps_the_originals_and_unit_length():
    base = unit(np.random.default_rng(1).normal(size=(20, 8)))
    big = synthetic_corpus(base, 4, 0.05)
    assert big.shape == (80, 8)
    assert np.array_equal(big[:20], base)
    assert np.linalg.norm(big, axis=1) == pytest.approx(1.0, abs=1e-5)
    assert np.sum(big[20:40] * base, axis=1).mean() > 0.8


def test_synthetic_corpus_is_reproducible():
    base = unit(np.random.default_rng(1).normal(size=(20, 8)))
    assert np.array_equal(synthetic_corpus(base, 3, 0.05, seed=2), synthetic_corpus(base, 3, 0.05, seed=2))
