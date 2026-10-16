import sys

import pandas as pd
from sentence_transformers import SentenceTransformer

from src.data import DATA_DIR, load_corpus, load_qrels, load_queries
from src.dense import DenseSearch, encode_corpus, slug
from src.evaluate import bootstrap_ci, evaluate
from src.splits import load_splits

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BM25_PER_QUERY = DATA_DIR.parent / "bm25_val_per_query.csv"


def main(model_name=DEFAULT_MODEL):
    corpus = load_corpus()
    queries = load_queries()
    qrels = load_qrels("train")
    val = load_splits().val

    model = SentenceTransformer(model_name, device="cpu")
    print(f"{model_name}: max_seq_length {model.max_seq_length}, dim {model.get_embedding_dimension()}")
    doc_ids, emb, seconds = encode_corpus(model, corpus, model_name)
    if seconds is not None:
        print(f"encoded {len(doc_ids)} documents in {seconds:.0f}s ({1000 * seconds / len(doc_ids):.0f} s per 1000)")

    result = evaluate(DenseSearch(model, doc_ids, emb).system(), queries, qrels, val)
    result.per_query.to_csv(DATA_DIR.parent / f"{slug(model_name)}_val_per_query.csv")
    print(result.summary.round(3))

    bm25 = pd.read_csv(BM25_PER_QUERY, index_col=0, dtype={0: str})
    bm25.index = bm25.index.astype(str)
    diff = result.per_query["ndcg@10"] - bm25.loc[result.per_query.index, "ndcg@10"]
    mean, lo, hi = bootstrap_ci(diff)
    print(f"nDCG@10 minus BM25 (paired): {mean:.3f} ({lo:.3f} to {hi:.3f})")
    print(f"queries where the model wins: {(diff > 0).sum()}, loses: {(diff < 0).sum()}, ties: {(diff == 0).sum()}")


if __name__ == "__main__":
    main(*sys.argv[1:])
