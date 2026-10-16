import random
from dataclasses import dataclass

from src.data import load_qrels

SEED = 0
VAL_FRACTION = 0.2


@dataclass
class Splits:
    train: list
    val: list
    test: list
    leaky: set


def group_queries(qrels):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for qid, docs in qrels.items():
        for doc in docs:
            parent[find(("q", qid))] = find(("d", doc))

    groups = {}
    for qid in qrels:
        groups.setdefault(find(("q", qid)), []).append(qid)
    return [sorted(g) for g in groups.values()]


def build_splits(train_qrels, test_qrels, seed=SEED, val_fraction=VAL_FRACTION):
    groups = sorted(group_queries(train_qrels))
    random.Random(seed).shuffle(groups)

    target = round(val_fraction * len(train_qrels))
    train, val = [], []
    for group in groups:
        (val if len(val) < target else train).extend(group)

    test_docs = {doc for docs in test_qrels.values() for doc in docs}
    leaky = {qid for qid, docs in train_qrels.items() if set(docs) & test_docs}
    return Splits(sorted(train), sorted(val), sorted(test_qrels), leaky)


def load_splits(seed=SEED, val_fraction=VAL_FRACTION):
    return build_splits(load_qrels("train"), load_qrels("test"), seed, val_fraction)
