import pytest

from src.data import load_qrels
from src.splits import build_splits, group_queries, load_splits

TRAIN = {
    "q1": {"d1": 1},
    "q2": {"d1": 1, "d2": 1},
    "q3": {"d2": 1},
    "q4": {"d3": 1},
    "q5": {"d4": 1},
    "q6": {"d5": 1},
    "q7": {"d5": 1},
    "q8": {"d6": 1},
    "q9": {"d7": 1},
    "q10": {"d8": 1},
}
TEST = {"t1": {"d3": 1}, "t2": {"d9": 1}}


def test_groups_follow_shared_documents():
    groups = {tuple(g) for g in group_queries(TRAIN)}
    assert ("q1", "q2", "q3") in groups
    assert ("q6", "q7") in groups
    assert ("q4",) in groups
    assert sum(len(g) for g in groups) == len(TRAIN)


def test_toy_split_keeps_groups_together():
    s = build_splits(TRAIN, TEST, seed=1, val_fraction=0.3)
    for group in group_queries(TRAIN):
        assert set(group) <= set(s.train) or set(group) <= set(s.val)
    assert set(s.train) | set(s.val) == set(TRAIN)
    assert not set(s.train) & set(s.val)
    assert s.test == ["t1", "t2"]


def test_toy_leaky_flag():
    s = build_splits(TRAIN, TEST)
    assert s.leaky == {"q4"}


def test_same_seed_same_split():
    a = build_splits(TRAIN, TEST, seed=3)
    b = build_splits(TRAIN, TEST, seed=3)
    assert (a.train, a.val) == (b.train, b.val)


@pytest.fixture(scope="module")
def real():
    return load_splits(), load_qrels("train"), load_qrels("test")


def test_no_query_in_two_splits(real):
    s, _, _ = real
    assert not set(s.train) & set(s.val)
    assert not set(s.train) & set(s.test)
    assert not set(s.val) & set(s.test)


def test_real_split_covers_all_queries(real):
    s, train_qrels, test_qrels = real
    assert set(s.train) | set(s.val) == set(train_qrels)
    assert set(s.test) == set(test_qrels)


def test_no_shared_document_between_train_and_val(real):
    s, train_qrels, _ = real
    train_docs = {d for q in s.train for d in train_qrels[q]}
    val_docs = {d for q in s.val for d in train_qrels[q]}
    assert not train_docs & val_docs


def test_val_size_close_to_target(real):
    s, train_qrels, _ = real
    assert abs(len(s.val) - 0.2 * len(train_qrels)) < 15


def test_leaky_matches_test_overlap(real):
    s, train_qrels, test_qrels = real
    test_docs = {d for v in test_qrels.values() for d in v}
    for qid, docs in train_qrels.items():
        assert (qid in s.leaky) == bool(set(docs) & test_docs)
