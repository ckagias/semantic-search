import re
import time

import numpy as np

from src.data import DATA_DIR
from src.evaluate import DEPTH

EMB_DIR = DATA_DIR.parent / "emb"


def doc_text(doc):
    return doc["title"] + " " + doc["text"]


def slug(model_name):
    return re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_")


def encode_corpus(model, corpus, model_name, batch_size=32, use_cache=True):
    doc_ids = sorted(corpus)
    path = EMB_DIR / f"{slug(model_name)}.npy"
    if use_cache and path.exists():
        return doc_ids, np.load(path), None
    start = time.perf_counter()
    emb = model.encode(
        [doc_text(corpus[d]) for d in doc_ids],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    seconds = time.perf_counter() - start
    EMB_DIR.mkdir(exist_ok=True)
    np.save(path, emb)
    return doc_ids, emb, seconds


class DenseSearch:
    def __init__(self, model, doc_ids, doc_emb):
        self.model = model
        self.doc_ids = doc_ids
        self.doc_emb = doc_emb

    def scores(self, query):
        q = self.model.encode(query, normalize_embeddings=True, show_progress_bar=False)
        return self.doc_emb @ q

    def system(self):
        def rank(query):
            order = np.argsort(-self.scores(query), kind="stable")[:DEPTH]
            return [self.doc_ids[i] for i in order]

        return rank
