import pytest

from src.ceiling import candidate_metrics, oracle_ranking, union_candidates


def test_oracle_ranking_puts_relevant_documents_first_and_keeps_the_rest_in_order():
    assert oracle_ranking(["a", "b", "c", "d"], {"c", "d"}) == ["c", "d", "a", "b"]


def test_union_keeps_the_first_list_and_adds_only_new_documents():
    assert union_candidates(["a", "b", "c"], ["b", "d", "e"], 2) == ["a", "b", "d"]


def test_perfect_reranker_scores_one_when_all_relevant_documents_are_present():
    m = candidate_metrics(["x", "r1", "y", "r2"], {"r1", "r2"})
    assert m == {"recall": 1.0, "any_hit": 1.0, "oracle_ndcg@10": pytest.approx(1.0)}


def test_missing_relevant_documents_cap_recall_and_ndcg():
    m = candidate_metrics(["x", "r1", "y"], {"r1", "r2"})
    assert m["recall"] == 0.5 and m["any_hit"] == 1.0
    assert m["oracle_ndcg@10"] == pytest.approx(1 / (1 + 1 / 1.5849625007211563))


def test_no_relevant_document_among_candidates_gives_zero():
    assert candidate_metrics(["x", "y"], {"r"}) == {"recall": 0.0, "any_hit": 0.0, "oracle_ndcg@10": 0.0}
