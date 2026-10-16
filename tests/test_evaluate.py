import numpy as np
import pytest

from src.evaluate import bootstrap_ci, evaluate, rank_from_scores

DOCS = [f"d{i}" for i in range(200)]
QUERIES = {f"q{i}": f"text {i}" for i in range(40)}
QRELS = {f"q{i}": {f"d{i}": 1} for i in range(40)}
BY_TEXT = {text: qid for qid, text in QUERIES.items()}


def perfect(text):
    answer = f"d{BY_TEXT[text][1:]}"
    return [answer] + [d for d in DOCS if d != answer]


def worst(text):
    answer = f"d{BY_TEXT[text][1:]}"
    return [d for d in DOCS if d != answer] + [answer]


def test_rank_from_scores_orders_by_score():
    assert rank_from_scores(["a", "b", "c"], [0.1, 0.9, 0.5]) == ["b", "c", "a"]


def test_rank_from_scores_breaks_ties_by_doc_id():
    ids = ["c", "a", "b", "d"]
    assert rank_from_scores(ids, [1, 1, 1, 0]) == ["a", "b", "c", "d"]
    assert rank_from_scores(ids[::-1], [0, 1, 1, 1]) == ["a", "b", "c", "d"]


def test_rank_from_scores_truncates():
    assert rank_from_scores(["a", "b", "c"], [3, 2, 1], top_n=2) == ["a", "b"]


def test_perfect_system_scores_one():
    result = evaluate(perfect, QUERIES, QRELS, list(QUERIES), n_boot=200)
    assert (result.summary["mean"] == 1).all()
    assert (result.summary["lo"] == 1).all() and (result.summary["hi"] == 1).all()


def test_worst_system_scores_zero_inside_depth():
    result = evaluate(worst, QUERIES, QRELS, list(QUERIES), n_boot=200)
    assert (result.summary["mean"] == 0).all()


def test_per_query_has_one_row_per_query():
    qids = list(QUERIES)[:10]
    result = evaluate(perfect, QUERIES, QRELS, qids, n_boot=100)
    assert list(result.per_query.index) == qids
    assert set(result.per_query.columns) == {"recall@10", "recall@100", "mrr@10", "ndcg@10"}


def test_system_only_sees_query_text():
    seen = []

    def spy(text):
        seen.append(text)
        return perfect(text)

    evaluate(spy, QUERIES, QRELS, ["q3"], n_boot=50)
    assert seen == ["text 3"]


def test_bootstrap_interval_contains_mean_and_is_ordered():
    values = np.random.default_rng(1).random(100)
    mean, lo, hi = bootstrap_ci(values)
    assert lo <= mean <= hi
    assert mean == pytest.approx(values.mean())


def test_bootstrap_constant_values_have_zero_width():
    mean, lo, hi = bootstrap_ci([0.3] * 20)
    assert mean == lo == hi == pytest.approx(0.3)


def test_bootstrap_same_seed_same_interval():
    values = np.random.default_rng(2).random(50)
    assert bootstrap_ci(values, seed=5) == bootstrap_ci(values, seed=5)


def test_bootstrap_width_shrinks_with_more_queries():
    rng = np.random.default_rng(3)
    small = bootstrap_ci(rng.random(30))
    large = bootstrap_ci(rng.random(3000))
    assert (large[2] - large[1]) < (small[2] - small[1])
