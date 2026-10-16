import random

from src.data import load_corpus, load_qrels, load_queries
from src.dense import doc_text
from src.keyword import build_bm25
from src.splits import load_splits

HARD_TOP_K = 10


def build_pairs(qids, queries, qrels, corpus):
    pairs = []
    for qid in sorted(qids, key=int):
        for doc_id in sorted(qrels[qid]):
            if qrels[qid][doc_id] > 0:
                pairs.append(
                    {"qid": qid, "query": queries[qid], "doc_id": doc_id, "doc": doc_text(corpus[doc_id])}
                )
    return pairs


def add_hard_negatives(pairs, rank, qrels, corpus, banned, seed=0, top_k=HARD_TOP_K):
    rankings = {}
    out = []
    for pair in pairs:
        qid = pair["qid"]
        if qid not in rankings:
            rankings[qid] = rank(pair["query"])
        relevant = {d for d, grade in qrels[qid].items() if grade > 0}
        pool = [d for d in rankings[qid] if d not in relevant and d not in banned][:top_k]
        if not pool:
            pool = sorted(set(corpus) - relevant - banned)
        neg_id = random.Random(f"{seed}-{qid}-{pair['doc_id']}").choice(pool)
        out.append({**pair, "neg_id": neg_id, "neg": doc_text(corpus[neg_id])})
    return out


def train_pairs(clean_only=True, hard_negatives=False, seed=0):
    corpus = load_corpus()
    queries = load_queries()
    qrels = load_qrels("train")
    splits = load_splits()
    qids = [q for q in splits.train if not (clean_only and q in splits.leaky)]
    pairs = build_pairs(qids, queries, qrels, corpus)
    if hard_negatives:
        banned = {d for docs in load_qrels("test").values() for d in docs}
        pairs = add_hard_negatives(pairs, build_bm25(corpus).system(), qrels, corpus, banned, seed)
    return pairs
