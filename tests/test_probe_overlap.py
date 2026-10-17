import pytest

from src import probe_overlap
from src.probe_overlap import build_variants, delete_words, overlapping_words

CORPUS = {
    "d1": {"title": "BRCA1 mutations", "text": "Breast cancer risk rises with BRCA1 mutations in women."},
    "d2": {"title": "Diet", "text": "Vegetables and fruit lower risk."},
}
DF = {"brca1": 1, "mutat": 1, "breast": 1, "cancer": 1, "risk": 2, "the": 100, "in": 100, "of": 100, "women": 1}
N_DOCS = 100


@pytest.fixture(autouse=True)
def no_wordnet(monkeypatch):
    monkeypatch.setattr(probe_overlap, "synonym", lambda word, banned: None)


def test_overlapping_words_are_rare_and_in_the_relevant_document():
    tokens = ["the", "brca1", "mutations", "in", "mice"]
    positions = overlapping_words(tokens, {"brca1", "mutat", "the"}, {"brca1", "mutat", "mice"})
    assert positions == [1, 2]


def test_delete_words_keeps_order_of_the_rest():
    assert delete_words(["a", "b", "c", "d"], [1, 3]) == ["a", "c"]


def test_overlap_deleted_variant_removes_exactly_the_overlapping_rare_words():
    variants, n_removed, n_replaced, n_tokens = build_variants("q", "the BRCA1 mutations in cancer cells", ["d1"], CORPUS, DF, N_DOCS)
    assert variants["control"] == "the brca1 mutations in cancer cells"
    assert variants["overlap_deleted"] == "the in cells"
    assert (n_removed, n_replaced, n_tokens) == (3, 0, 6)


def test_rarest_variant_removes_one_word_and_random_variants_remove_as_many_as_overlap():
    variants, *_ = build_variants("q", "the BRCA1 mutations in cancer cells", ["d1"], CORPUS, DF, N_DOCS)
    assert len(variants["rarest_deleted"].split()) == 5
    assert len(variants["random_0"].split()) == 3
    assert len(variants["random1_0"].split()) == 5


def test_query_without_enough_words_left_is_skipped():
    assert build_variants("q", "BRCA1 mutations cancer", ["d1"], CORPUS, DF, N_DOCS) is None


def test_query_without_overlap_is_skipped():
    assert build_variants("q", "the vegetables in mice", ["d1"], CORPUS, DF, N_DOCS) is None


def test_variants_are_reproducible():
    a, *_ = build_variants("q", "the BRCA1 mutations in cancer cells and women", ["d1"], CORPUS, DF, N_DOCS)
    b, *_ = build_variants("q", "the BRCA1 mutations in cancer cells and women", ["d1"], CORPUS, DF, N_DOCS)
    assert a == b
