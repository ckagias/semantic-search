import math

import pytest

from src.metrics import mrr_at_k, ndcg_at_k, recall_at_k

RANKING = list("abcdefghij")


def disc(position):
    return 1 / math.log2(position + 1)


def test_single_relevant_at_rank_3():
    relevant = {"c"}
    assert recall_at_k(RANKING, relevant, 2) == 0
    assert recall_at_k(RANKING, relevant, 3) == 1
    assert mrr_at_k(RANKING, relevant) == pytest.approx(1 / 3)
    assert ndcg_at_k(RANKING, relevant) == pytest.approx(0.5)


def test_two_relevant_one_inside_k_one_outside():
    relevant = {"c", "f"}
    assert recall_at_k(RANKING, relevant, 5) == pytest.approx(0.5)
    assert recall_at_k(RANKING, relevant, 6) == pytest.approx(1)
    assert mrr_at_k(RANKING, relevant) == pytest.approx(1 / 3)
    expected = (disc(3) + disc(6)) / (disc(1) + disc(2))
    assert ndcg_at_k(RANKING, relevant) == pytest.approx(expected)
    assert ndcg_at_k(RANKING, relevant, k=5) == pytest.approx(disc(3) / (disc(1) + disc(2)))


def test_nothing_relevant_retrieved():
    relevant = {"x", "y"}
    assert recall_at_k(RANKING, relevant, 10) == 0
    assert mrr_at_k(RANKING, relevant) == 0
    assert ndcg_at_k(RANKING, relevant) == 0


def test_relevant_just_outside_k():
    ranking = list("abcdefghijk")
    assert recall_at_k(ranking, {"k"}, 10) == 0
    assert mrr_at_k(ranking, {"k"}, 10) == 0
    assert ndcg_at_k(ranking, {"k"}, 10) == 0
    assert mrr_at_k(ranking, {"k"}, 11) == pytest.approx(1 / 11)


def test_perfect_ranking():
    relevant = {"a", "b"}
    assert recall_at_k(RANKING, relevant, 10) == 1
    assert mrr_at_k(RANKING, relevant) == 1
    assert ndcg_at_k(RANKING, relevant) == pytest.approx(1)


def test_ndcg_prefers_higher_rank():
    assert ndcg_at_k(RANKING, {"a"}) > ndcg_at_k(RANKING, {"b"}) > ndcg_at_k(RANKING, {"c"})
    assert ndcg_at_k(RANKING, {"b"}) == pytest.approx(disc(2))


def test_more_relevant_documents_than_k():
    relevant = set("abcdefghijklmnopqrst")
    assert recall_at_k(RANKING, relevant, 10) == pytest.approx(0.5)
    assert ndcg_at_k(RANKING, relevant, 10) == pytest.approx(1)


def test_ranking_shorter_than_k():
    assert recall_at_k(["a"], {"a", "b"}, 10) == pytest.approx(0.5)
    assert mrr_at_k(["a"], {"a"}) == 1
    assert ndcg_at_k(["a"], {"a"}) == pytest.approx(1)


def test_metrics_stay_in_unit_interval():
    for relevant in ({"a"}, {"e", "j"}, {"b", "c", "d"}, {"z"}):
        for k in (1, 3, 10):
            assert 0 <= recall_at_k(RANKING, relevant, k) <= 1
            assert 0 <= mrr_at_k(RANKING, relevant, k) <= 1
            assert 0 <= ndcg_at_k(RANKING, relevant, k) <= 1


@pytest.mark.parametrize("metric", [recall_at_k, mrr_at_k, ndcg_at_k])
def test_invalid_inputs(metric):
    with pytest.raises(ValueError):
        metric(RANKING, set(), 10)
    with pytest.raises(ValueError):
        metric(RANKING, {"a"}, 0)
    with pytest.raises(ValueError):
        metric(["a", "a", "b"], {"a"}, 10)
