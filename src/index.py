import re
from collections import Counter

TOKEN = re.compile(r"\w+")


def tokenize(text):
    return TOKEN.findall(text.lower())


def doc_tokens(doc):
    return tokenize(doc["title"] + " " + doc["text"])


class InvertedIndex:
    def __init__(self, corpus):
        self.postings = {}
        self.doc_len = {}
        for doc_id in sorted(corpus):
            tokens = doc_tokens(corpus[doc_id])
            self.doc_len[doc_id] = len(tokens)
            for term, tf in Counter(tokens).items():
                self.postings.setdefault(term, {})[doc_id] = tf
        self.n_docs = len(self.doc_len)
        self.avgdl = sum(self.doc_len.values()) / self.n_docs

    def df(self, term):
        return len(self.postings.get(term, ()))

    def candidates(self, query_terms):
        docs = set()
        for term in query_terms:
            docs.update(self.postings.get(term, ()))
        return docs
