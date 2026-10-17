# Semantic Search

Keyword search (BM25), pretrained embeddings and a fine-tuned bi-encoder, compared on BEIR SciFact with the same metrics and the same queries. The aim is to understand retrieval end to end: index, query, rank, metrics, and contrastive fine-tuning. A tie with BM25 is a valid result if it can be explained.

Status: work in progress. Phases 0 to 6 are done (setup, data, evaluation, BM25, pretrained embeddings, fine-tuning, error analysis and the final test scoring). Scaling and the API are still to do. The first table is the final score on the 300 test queries, scored once; the second table is the validation numbers behind the choices.

## Results so far

### Test queries (300, scored once)

nDCG@10 with a 95% bootstrap interval over queries. Settings were fixed on validation before this run. Full write-up in [results.md](results.md).

| system | nDCG@10 | recall@10 | recall@100 |
|---|---|---|---|
| Random ranking (5 seeds) | 0.000 to 0.002 | n/a | 0.006 to 0.028 |
| BM25 (tuned, Snowball stemming) | 0.686 (0.639 to 0.730) | 0.808 | 0.916 |
| all-MiniLM-L6-v2, pretrained | 0.645 (0.599 to 0.689) | 0.783 | 0.925 |
| all-MiniLM-L6-v2, fine-tuned, BM25 hard negatives (3 seeds) | 0.652 to 0.668 | 0.795 to 0.809 | 0.927 to 0.930 |
| bge-base-en-v1.5, pretrained | 0.740 (0.700 to 0.779) | 0.874 | 0.967 |
| Hybrid: BM25 + bge-base, weighted sum (alpha 0.2) | 0.752 (0.711 to 0.791) | 0.875 | 0.980 |

- bge-base beats BM25 by +0.055 (+0.022 to +0.086); the hybrid beats BM25 by +0.067 (+0.040 to +0.093) but is not distinguishable from bge-base alone (+0.012, -0.005 to +0.029).
- Fine-tuned MiniLM beats pretrained MiniLM by +0.016 (+0.001 to +0.031), less than half of the validation gain, and is not distinguishable from BM25 (-0.025, -0.059 to +0.009).
- A perfect re-ranker over the hybrid's top 100 would reach 0.968 nDCG@10 on validation, so the first stage still leaves about 0.17 of headroom.

### Validation queries (165)

nDCG@10 on the 165 validation queries, 95% bootstrap interval over queries.

| system | nDCG@10 | recall@10 | recall@100 |
|---|---|---|---|
| Random ranking | 0.000 to 0.004 | 0.000 to 0.012 | 0.018 to 0.054 |
| BM25 (tuned, Snowball stemming) | 0.698 (0.637 to 0.757) | 0.829 | 0.929 |
| all-MiniLM-L6-v2, pretrained | 0.625 (0.563 to 0.687) | 0.760 | 0.943 |
| all-MiniLM-L6-v2, fine-tuned, in-batch negatives (3 seeds) | 0.631 to 0.650 | 0.783 | 0.944 |
| all-MiniLM-L6-v2, fine-tuned, BM25 hard negatives (3 seeds) | 0.661 to 0.662 | 0.811 | 0.908 |
| bge-small-en-v1.5, pretrained | 0.724 (0.668 to 0.780) | n/a | 0.968 |
| bge-base-en-v1.5, pretrained | 0.759 (0.705 to 0.810) | n/a | 0.974 |
| Hybrid: BM25 + bge-base, weighted sum (alpha 0.2) | 0.794 (0.742 to 0.842) | 0.908 | 0.968 |

- Pretrained bge-base beats stemmed BM25 by +0.061 (+0.017 to +0.106); MiniLM and mpnet-base lose to it.
- The hybrid beats both parts: +0.096 over BM25 and +0.034 over bge-base alone.
- Fine-tuning MiniLM on 475 train pairs, three seeds: in-batch negatives give +0.006 to +0.024 over pretrained; BM25 hard negatives give about +0.036 in every seed (0.661 to 0.662). Hard negatives beat in-batch in all three seeds, but the margin (+0.012 to +0.031) is only clearly above zero in one. The fine-tuned model stays below BM25 and loses recall@100.
- A pretrained cross-encoder (MS MARCO MiniLM) re-ranking the hybrid's top 50 lowered validation nDCG@10 from 0.794 to 0.737 (-0.057, -0.088 to -0.027) and costs 1.8 s per query against 0.1 s for the first stage.
- Selection on the same validation queries makes the best rows slightly optimistic; on the test queries the hybrid and fine-tuning margins shrink (see above).

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Data: BEIR SciFact (`scifact.zip` from the BEIR dataset mirror), unpacked so that `data/scifact/` holds `corpus.jsonl`, `queries.jsonl` and `qrels/{train,test}.tsv`. Torch is the CPU build; every experiment here runs on the CPU.

## Running

```
python -m pytest
python -m src.score_bm25                    # tuned stemmed BM25 on validation
python -m src.score_dense BAAI/bge-base-en-v1.5
python -m src.score_hybrid
python -m src.train --negatives hard        # fine-tune MiniLM; --n-queries and --seed for the ablations
python -m src.summarize_finetune
python -m src.analyze_errors                 # fine-tuned vs BM25, per query
python -m src.probe_overlap                 # delete or replace overlapping query words (downloads NLTK WordNet)
python -m src.bench_scale                   # exact search time at 5k, 50k and 500k vectors
python -m src.bench_ann                     # FAISS IVF and HNSW against exact search
python -m src.rerank                        # cross-encoder re-ranking of the hybrid's top 50 (MiniLM; --models for others)
```

Scripts write per-query score tables to `data/` and models to `models/`; both are untracked.

## Rules followed

- One evaluation function for every system, tested by hand before any model was scored.
- Settings chosen on validation queries only. Test queries are scored once per final system.
- Fine-tuning uses train queries that share no relevant document with a test query (415 of 644 train queries).
