from src.data import load_qrels
from src.pairs import build_pairs, train_pairs
from src.splits import load_splits

CORPUS = {
    "10": {"title": "T10", "text": "body10"},
    "20": {"title": "T20", "text": "body20"},
    "30": {"title": "T30", "text": "body30"},
}
QUERIES = {"1": "first", "2": "second", "3": "third"}
QRELS = {"1": {"10": 1}, "2": {"10": 1, "20": 1, "30": 0}, "3": {"30": 1}}


def test_one_pair_per_relevant_document():
    pairs = build_pairs(["1", "2"], QUERIES, QRELS, CORPUS)
    assert [(p["qid"], p["doc_id"]) for p in pairs] == [("1", "10"), ("2", "10"), ("2", "20")]


def test_pair_text_is_query_and_title_plus_abstract():
    pair = build_pairs(["1"], QUERIES, QRELS, CORPUS)[0]
    assert pair == {"qid": "1", "query": "first", "doc_id": "10", "doc": "T10 body10"}


def test_only_requested_queries_are_used():
    assert {p["qid"] for p in build_pairs(["3"], QUERIES, QRELS, CORPUS)} == {"3"}


def test_clean_pairs_contain_no_validation_or_test_query_and_no_test_document():
    splits = load_splits()
    pairs = train_pairs(clean_only=True)
    test_docs = {d for v in load_qrels("test").values() for d in v}
    assert {p["qid"] for p in pairs} <= set(splits.train)
    assert not {p["qid"] for p in pairs} & set(splits.val)
    assert not {p["qid"] for p in pairs} & set(splits.test)
    assert not {p["doc_id"] for p in pairs} & test_docs
    assert not {p["qid"] for p in pairs} & splits.leaky


def test_all_pairs_include_leaky_queries_and_cover_every_train_query():
    splits = load_splits()
    pairs = train_pairs(clean_only=False)
    assert {p["qid"] for p in pairs} == set(splits.train)
    assert len(pairs) > len(train_pairs(clean_only=True))
