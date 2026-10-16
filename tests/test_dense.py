import numpy as np
import pytest

from src.dense import DenseSearch, doc_text, slug


class StubModel:
    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, text, normalize_embeddings=True, show_progress_bar=False):
        v = np.asarray(self.vectors[text], dtype=float)
        return v / np.linalg.norm(v) if normalize_embeddings else v


DOC_IDS = ["a", "b", "c", "d"]
DOC_EMB = np.array([[1, 0], [0, 1], [1, 1], [1, 0]], dtype=float)
DOC_EMB = DOC_EMB / np.linalg.norm(DOC_EMB, axis=1, keepdims=True)


def test_cosine_scores_match_manual_values():
    model = StubModel({"q": [1, 0]})
    scores = DenseSearch(model, DOC_IDS, DOC_EMB).scores("q")
    assert scores == pytest.approx([1.0, 0.0, 1 / np.sqrt(2), 1.0])


def test_ranking_orders_by_similarity_and_breaks_ties_by_index():
    model = StubModel({"q": [1, 0]})
    ranking = DenseSearch(model, DOC_IDS, DOC_EMB).system()("q")
    assert ranking == ["a", "d", "c", "b"]


def test_scale_of_query_vector_does_not_change_ranking():
    small = DenseSearch(StubModel({"q": [1, 0.2]}), DOC_IDS, DOC_EMB).system()("q")
    large = DenseSearch(StubModel({"q": [100, 20]}), DOC_IDS, DOC_EMB).system()("q")
    assert small == large


def test_doc_text_joins_title_and_abstract():
    assert doc_text({"title": "T", "text": "body"}) == "T body"


def test_slug_is_filesystem_safe():
    assert slug("sentence-transformers/all-MiniLM-L6-v2") == "sentence_transformers_all_MiniLM_L6_v2"
