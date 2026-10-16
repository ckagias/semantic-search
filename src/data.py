import csv
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "scifact"


def load_corpus(data_dir=DATA_DIR):
    corpus = {}
    with open(Path(data_dir) / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            corpus[row["_id"]] = {"title": row["title"], "text": row["text"]}
    return corpus


def load_queries(data_dir=DATA_DIR):
    queries = {}
    with open(Path(data_dir) / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            queries[row["_id"]] = row["text"]
    return queries


def load_qrels(split, data_dir=DATA_DIR):
    qrels = {}
    with open(Path(data_dir) / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    return qrels
