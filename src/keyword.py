import math

from src.evaluate import rank_from_scores
from src.index import InvertedIndex, tokenize_stemmed

BM25_TUNED = {"k1": 1.8, "b": 0.5, "idf": "lucene"}


class TfidfSearch:
    def __init__(self, index):
        self.index = index
        n = index.n_docs
        self.idf = {t: math.log(n / len(p)) for t, p in index.postings.items()}
        sq = dict.fromkeys(index.doc_len, 0.0)
        for term, posting in index.postings.items():
            for doc_id, tf in posting.items():
                sq[doc_id] += ((1 + math.log(tf)) * self.idf[term]) ** 2
        self.norm = {d: math.sqrt(v) for d, v in sq.items()}

    def scores(self, query_tokens):
        out = {}
        for term in set(query_tokens):
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for doc_id, tf in self.index.postings[term].items():
                w = (1 + math.log(tf)) * idf * idf
                out[doc_id] = out.get(doc_id, 0.0) + w / self.norm[doc_id]
        return out

    def system(self):
        def rank(query):
            return rank_from_scores_dict(self.scores(self.index.tokenizer(query)))

        return rank


class BM25:
    def __init__(self, index, k1=1.5, b=0.75, idf="okapi", epsilon=0.25):
        self.index = index
        self.k1 = k1
        self.b = b
        n = index.n_docs
        if idf == "okapi":
            raw = {t: math.log((n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in index.postings.items()}
            floor = epsilon * sum(raw.values()) / len(raw)
            self.idf = {t: v if v >= 0 else floor for t, v in raw.items()}
        elif idf == "lucene":
            self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in index.postings.items()}
        else:
            raise ValueError(f"unknown idf: {idf}")

    def scores(self, query_tokens):
        out = {}
        index = self.index
        for term in query_tokens:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for doc_id, tf in index.postings[term].items():
                length_norm = 1 - self.b + self.b * index.doc_len[doc_id] / index.avgdl
                out[doc_id] = out.get(doc_id, 0.0) + idf * tf * (self.k1 + 1) / (tf + self.k1 * length_norm)
        return out

    def system(self):
        def rank(query):
            return rank_from_scores_dict(self.scores(self.index.tokenizer(query)))

        return rank


def rank_from_scores_dict(scores):
    ids = list(scores)
    return rank_from_scores(ids, [scores[d] for d in ids])


def build_bm25(corpus):
    return BM25(InvertedIndex(corpus, tokenize_stemmed), **BM25_TUNED)
