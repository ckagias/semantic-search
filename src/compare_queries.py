import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.data import load_corpus, load_qrels, load_queries
from src.dense import DenseSearch, encode_corpus
from src.index import InvertedIndex, doc_tokens, tokenize
from src.keyword import BM25, BM25_TUNED
from src.score_dense import BGE_PREFIX
from src.splits import load_splits

MODEL = "BAAI/bge-base-en-v1.5"
SHOW = 8


def first_rank(ranking, relevant):
    for position, doc in enumerate(ranking, start=1):
        if doc in relevant:
            return position
    return None


def main():
    corpus = load_corpus()
    queries = load_queries()
    qrels = load_qrels("train")
    val = load_splits().val

    bm25 = BM25(InvertedIndex(corpus), **BM25_TUNED).system()
    model = SentenceTransformer(MODEL, device="cpu")
    doc_ids, emb, _ = encode_corpus(model, corpus, MODEL)
    dense = DenseSearch(model, doc_ids, emb, BGE_PREFIX).system()

    rows = []
    for qid in val:
        relevant = set(qrels[qid])
        q_tokens = set(tokenize(queries[qid]))
        doc_vocab = set().union(*(doc_tokens(corpus[d]) for d in relevant))
        rows.append(
            {
                "qid": qid,
                "query": queries[qid],
                "title": corpus[next(iter(relevant))]["title"],
                "bm25_rank": first_rank(bm25(queries[qid]), relevant),
                "dense_rank": first_rank(dense(queries[qid]), relevant),
                "overlap": len(q_tokens & doc_vocab) / len(q_tokens),
                "n_words": len(tokenize(queries[qid])),
            }
        )
    df = pd.DataFrame(rows)
    big = 1000
    df["bm25_rr"] = 1 / df["bm25_rank"].fillna(big)
    df["dense_rr"] = 1 / df["dense_rank"].fillna(big)
    df["diff"] = df["dense_rr"] - df["bm25_rr"]

    print("outcome by overlap tercile (share of query words found in the relevant document)")
    df["bucket"] = pd.qcut(df["overlap"], 3, labels=["low", "mid", "high"])
    print(df.groupby("bucket", observed=True).agg(
        n=("qid", "size"), overlap=("overlap", "mean"),
        bm25_mrr=("bm25_rr", "mean"), dense_mrr=("dense_rr", "mean"),
        dense_wins=("diff", lambda x: (x > 0).sum()), bm25_wins=("diff", lambda x: (x < 0).sum()),
    ).round(3))
    print(f"\ncorrelation of (dense - bm25) reciprocal rank with overlap: {np.corrcoef(df['diff'], df['overlap'])[0, 1]:.3f}")

    df["short"] = df["n_words"] <= df["n_words"].median()
    print(df.groupby("short").agg(n=("qid", "size"), bm25_mrr=("bm25_rr", "mean"), dense_mrr=("dense_rr", "mean")).round(3))

    cols = ["query", "title", "bm25_rank", "dense_rank", "overlap"]
    pd.set_option("display.width", 250, "display.max_colwidth", 90)
    print("\nBM25 wins by most:")
    print(df.sort_values("diff").head(SHOW)[cols].round(2).to_string(index=False))
    print("\ndense wins by most:")
    print(df.sort_values("diff", ascending=False).head(SHOW)[cols].round(2).to_string(index=False))
    print(f"\nboth miss the top 100: {(df['bm25_rank'].isna() & df['dense_rank'].isna()).sum()}, "
          f"only BM25 misses: {(df['bm25_rank'].isna() & df['dense_rank'].notna()).sum()}, "
          f"only dense misses: {(df['bm25_rank'].notna() & df['dense_rank'].isna()).sum()}")


if __name__ == "__main__":
    main()
