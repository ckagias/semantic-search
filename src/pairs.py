from src.data import load_corpus, load_qrels, load_queries
from src.dense import doc_text
from src.splits import load_splits


def build_pairs(qids, queries, qrels, corpus):
    pairs = []
    for qid in sorted(qids, key=int):
        for doc_id in sorted(qrels[qid]):
            if qrels[qid][doc_id] > 0:
                pairs.append(
                    {"qid": qid, "query": queries[qid], "doc_id": doc_id, "doc": doc_text(corpus[doc_id])}
                )
    return pairs


def train_pairs(clean_only=True):
    corpus = load_corpus()
    queries = load_queries()
    qrels = load_qrels("train")
    splits = load_splits()
    qids = [q for q in splits.train if not (clean_only and q in splits.leaky)]
    return build_pairs(qids, queries, qrels, corpus)
