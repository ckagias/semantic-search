import math

import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from src.data import load_corpus, load_queries
from src.index import InvertedIndex, doc_tokens, tokenize
from src.keyword import BM25, TfidfSearch
from src.splits import load_splits

CORPUS = {
    "d1": {"title": "cats", "text": "cats chase mice and mice run"},
    "d2": {"title": "dogs", "text": "dogs bark and dogs sleep"},
    "d3": {"title": "birds", "text": "birds fly and sing"},
    "d4": {"title": "cats and dogs", "text": "cats and dogs are pets"},
}
QUERIES = [["cats"], ["dogs", "bark"], ["cats", "dogs", "fly"], ["mice", "mice"], ["unknown"], ["and"]]


def reference(corpus, **kwargs):
    ids = sorted(corpus)
    return ids, BM25Okapi([doc_tokens(corpus[d]) for d in ids], **kwargs)


@pytest.mark.parametrize("k1,b", [(1.5, 0.75), (1.2, 0.5), (2.0, 1.0), (0.9, 0.0)])
def test_bm25_matches_rank_bm25_on_toy_corpus(k1, b):
    ids, ref = reference(CORPUS, k1=k1, b=b)
    ours = BM25(InvertedIndex(CORPUS), k1=k1, b=b)
    for query in QUERIES:
        expected = ref.get_scores(query)
        got = ours.scores(query)
        for doc_id, value in zip(ids, expected):
            assert got.get(doc_id, 0.0) == pytest.approx(value)


def test_bm25_single_term_by_hand():
    index = InvertedIndex(CORPUS)
    n, df, tf, dl, avgdl = 4, 2, 2, 7, index.avgdl
    idf = math.log((n - df + 0.5) / (df + 0.5))
    expected = idf * tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / avgdl))
    assert BM25(index).scores(["cats"])["d1"] == pytest.approx(expected)


def test_lucene_idf_is_never_negative():
    bm25 = BM25(InvertedIndex(CORPUS), idf="lucene")
    assert min(bm25.idf.values()) > 0


def test_okapi_floor_replaces_negative_idf():
    bm25 = BM25(InvertedIndex(CORPUS))
    assert min(bm25.idf.values()) >= 0
    assert bm25.idf["and"] > 0


def test_tfidf_single_term_by_hand():
    index = InvertedIndex(CORPUS)
    tfidf = TfidfSearch(index)
    idf = math.log(4 / 2)
    w = (1 + math.log(2)) * idf
    doc_vec = [
        (1 + math.log(tf)) * math.log(4 / len(index.postings[t]))
        for t, p in index.postings.items()
        for d, tf in p.items()
        if d == "d1"
    ]
    norm = math.sqrt(sum(v * v for v in doc_vec))
    assert tfidf.scores(["cats"])["d1"] == pytest.approx(w * idf / norm)


def test_tfidf_ignores_repeated_query_terms():
    tfidf = TfidfSearch(InvertedIndex(CORPUS))
    assert tfidf.scores(["cats", "cats"]) == tfidf.scores(["cats"])


def test_rankings_put_matching_documents_first():
    index = InvertedIndex(CORPUS)
    assert BM25(index).system()("birds fly")[0] == "d3"
    assert TfidfSearch(index).system()("birds fly")[0] == "d3"
    assert BM25(index).system()("xyz") == []


def test_bm25_matches_rank_bm25_on_scifact_validation_queries():
    corpus = load_corpus()
    queries = load_queries()
    val = load_splits().val
    ids, ref = reference(corpus)
    ours = BM25(InvertedIndex(corpus))
    for qid in val[:40]:
        tokens = tokenize(queries[qid])
        got = ours.scores(tokens)
        expected = ref.get_scores(tokens)
        mine = np.array([got.get(d, 0.0) for d in ids])
        assert np.allclose(mine, expected)
