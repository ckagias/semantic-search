from src.index import InvertedIndex, tokenize

CORPUS = {
    "d1": {"title": "Cats and dogs", "text": "Cats chase mice."},
    "d2": {"title": "Dogs", "text": "Dogs bark. Dogs sleep!"},
    "d3": {"title": "TNF-alpha", "text": "BRCA1 and IL-6"},
}


def test_tokenize_lowercases_and_splits_on_punctuation():
    assert tokenize("Cats chase mice.") == ["cats", "chase", "mice"]
    assert tokenize("TNF-alpha, IL-6") == ["tnf", "alpha", "il", "6"]
    assert tokenize("BRCA1") == ["brca1"]


def test_tokenize_keeps_non_ascii_letters():
    assert tokenize("TGF-β signalling") == ["tgf", "β", "signalling"]


def test_tokenize_empty():
    assert tokenize("") == []
    assert tokenize("... !!") == []


def test_postings_hold_term_frequencies():
    index = InvertedIndex(CORPUS)
    assert index.postings["cats"] == {"d1": 2}
    assert index.postings["dogs"] == {"d1": 1, "d2": 3}
    assert index.postings["brca1"] == {"d3": 1}


def test_document_frequency():
    index = InvertedIndex(CORPUS)
    assert index.df("dogs") == 2
    assert index.df("cats") == 1
    assert index.df("missing") == 0


def test_lengths_and_average():
    index = InvertedIndex(CORPUS)
    assert index.doc_len == {"d1": 6, "d2": 5, "d3": 6}
    assert index.n_docs == 3
    assert index.avgdl == 17 / 3


def test_term_frequencies_sum_to_document_length():
    index = InvertedIndex(CORPUS)
    for doc_id, length in index.doc_len.items():
        total = sum(p.get(doc_id, 0) for p in index.postings.values())
        assert total == length


def test_candidates_are_documents_with_any_query_term():
    index = InvertedIndex(CORPUS)
    assert index.candidates(["cats", "bark"]) == {"d1", "d2"}
    assert index.candidates(["missing"]) == set()
    assert index.candidates([]) == set()


def test_stemmed_tokenizer_merges_inflections():
    from src.index import tokenize_stemmed

    assert tokenize_stemmed("bullied bullying") == ["bulli", "bulli"]
    assert tokenize_stemmed("Mutations mutation") == ["mutat", "mutat"]


def test_index_uses_the_given_tokenizer():
    from src.index import tokenize_stemmed

    index = InvertedIndex(CORPUS, tokenizer=tokenize_stemmed)
    assert index.postings["cat"] == {"d1": 2}
    assert index.tokenizer is tokenize_stemmed
