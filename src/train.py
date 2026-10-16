import argparse
import random
import time

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import get_linear_schedule_with_warmup

from src.data import DATA_DIR, load_corpus, load_qrels, load_queries
from src.dense import DenseSearch, encode_corpus
from src.evaluate import evaluate
from src.loss import in_batch_loss
from src.pairs import train_pairs
from src.score_dense import DEFAULT_MODEL, QUERY_PREFIX
from src.splits import load_splits

SEED = 0
BATCH_SIZE = 16
EPOCHS = 4
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1
MODELS_DIR = DATA_DIR.parent.parent / "models"


def make_batches(pairs, batch_size, seed):
    rng = random.Random(seed)
    remaining = list(pairs)
    rng.shuffle(remaining)
    batches = []
    while remaining:
        batch, qids, positives, negatives, rest = [], set(), set(), set(), []
        for pair in remaining:
            neg = pair.get("neg_id")
            clash = (
                pair["qid"] in qids
                or pair["doc_id"] in positives
                or pair["doc_id"] in negatives
                or neg in positives
            )
            if len(batch) < batch_size and not clash:
                batch.append(pair)
                qids.add(pair["qid"])
                positives.add(pair["doc_id"])
                negatives.add(neg)
            else:
                rest.append(pair)
        batches.append(batch)
        remaining = rest
    return [b for b in batches if len(b) >= 2]


def embed(model, texts):
    return model(model.preprocess(texts))["sentence_embedding"]


def train_epoch(model, optimizer, scheduler, batches, prefix):
    model.train()
    losses = []
    for batch in batches:
        queries = embed(model, [prefix + p["query"] for p in batch])
        docs = embed(model, [p["doc"] for p in batch])
        if "neg" in batch[0]:
            docs = torch.cat([docs, embed(model, [p["neg"] for p in batch])])
        loss = in_batch_loss(queries, docs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    return sum(losses) / len(losses)


def score_val(model, run, epoch, corpus, queries, qrels, val, prefix, cached_name=None):
    model.eval()
    with torch.no_grad():
        if cached_name:
            doc_ids, emb, _ = encode_corpus(model, corpus, cached_name)
        else:
            doc_ids, emb, _ = encode_corpus(model, corpus, f"ft_{run}_epoch{epoch}", use_cache=False)
        return evaluate(DenseSearch(model, doc_ids, emb, prefix).system(), queries, qrels, val)


def main(model_name, negatives, seed, epochs, batch_size, lr, n_queries):
    run = f"{model_name.split('/')[-1]}_{negatives}" + (f"_n{n_queries}" if n_queries else "") + f"_s{seed}"
    torch.manual_seed(seed)

    corpus = load_corpus()
    queries = load_queries()
    qrels = load_qrels("train")
    val = load_splits().val
    pairs = train_pairs(clean_only=True, hard_negatives=negatives == "hard", seed=seed, n_queries=n_queries)
    prefix = QUERY_PREFIX.get(model_name, "")

    model = SentenceTransformer(model_name, device="cpu")
    steps_per_epoch = len(make_batches(pairs, batch_size, seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, round(WARMUP_FRACTION * epochs * steps_per_epoch), epochs * steps_per_epoch
    )
    print(f"{run}: {len(pairs)} pairs from {len({p['qid'] for p in pairs})} queries, {steps_per_epoch} batches of up to {batch_size} per epoch, lr {lr}")

    rows = []
    best = -1.0
    for epoch in range(epochs + 1):
        start = time.perf_counter()
        loss = None
        if epoch > 0:
            loss = train_epoch(model, optimizer, scheduler, make_batches(pairs, batch_size, seed + epoch), prefix)
        train_seconds = time.perf_counter() - start
        result = score_val(model, run, epoch, corpus, queries, qrels, val, prefix, model_name if epoch == 0 else None)
        s = result.summary
        rows.append(
            {
                "epoch": epoch,
                "train_loss": loss,
                "ndcg@10": s.loc["ndcg@10", "mean"],
                "ndcg_lo": s.loc["ndcg@10", "lo"],
                "ndcg_hi": s.loc["ndcg@10", "hi"],
                "mrr@10": s.loc["mrr@10", "mean"],
                "recall@10": s.loc["recall@10", "mean"],
                "recall@100": s.loc["recall@100", "mean"],
                "train_seconds": train_seconds,
            }
        )
        print(pd.DataFrame(rows).tail(1).round(3).to_string(header=(epoch == 0), index=False))
        if epoch > 0 and s.loc["ndcg@10", "mean"] > best:
            best = s.loc["ndcg@10", "mean"]
            model.save(str(MODELS_DIR / run))
            result.per_query.to_csv(DATA_DIR.parent / f"ft_{run}_val_per_query.csv")
            best_epoch = epoch

    pd.DataFrame(rows).to_csv(DATA_DIR.parent / f"ft_{run}_log.csv", index=False)
    print(f"best epoch {best_epoch}: nDCG@10 {best:.3f}, saved to {MODELS_DIR / run}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--negatives", choices=["in-batch", "hard"], default="in-batch")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--n-queries", type=int, default=None)
    args = parser.parse_args()
    main(args.model, args.negatives, args.seed, args.epochs, args.batch_size, args.lr, args.n_queries)
