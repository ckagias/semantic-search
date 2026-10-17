# Results (interim, validation only)

Dataset: BEIR SciFact, 5,183 documents (title + abstract), binary relevance, mean 1.14 relevant documents per train query. All scores are over the 165 validation queries unless stated; intervals are 95% bootstrap intervals over queries (10,000 resamples), paired when comparing two systems on the same queries. The 300 test queries have not been scored. Fine-tuning figures are for seed 0 unless a seed range is stated; the range over seeds 0 to 2 is in the Seeds section.

## Splits

644 train, 165 validation, 300 test queries. Train queries are grouped by shared relevant documents (490 groups) and whole groups move to validation, so no relevant document is shared between train and validation. 272 of the 809 labelled queries have a relevant document that also belongs to a test query; these are excluded from fine-tuning, which leaves 415 clean train queries (475 pairs, 292 unique documents). Tests assert that no clean pair contains a validation or test query or a test-relevant document.

## Floors

Random ranking: nDCG@10 0.000 to 0.004 over 5 seeds. Shortest document first: nDCG@10 0.000, recall@100 0.006. Document length carries no signal.

## Keyword search

BM25 and TF-IDF are implemented from an inverted index. BM25 with the rank_bm25 idf matches `rank_bm25.BM25Okapi` on all validation queries (largest absolute difference 4e-12). The rank_bm25 idf gives every term present in more than half the documents the same weight (1.81), so "the" weighs as much as "cancer"; the Lucene idf, ln(1 + (N - df + 0.5) / (df + 0.5)), does not.

Tuning (Lucene idf, k1 and b grid, 198 settings): the surface is a plateau, with the ten best settings within 0.003 nDCG@10. Snowball stemming adds +0.046 (+0.021 to +0.073), better on 34 queries and worse on 16.

| BM25 | nDCG@10 | MRR@10 | recall@10 | recall@100 |
|---|---|---|---|---|
| no stemming (k1 2.1, b 0.5) | 0.651 (0.589 to 0.713) | 0.614 | 0.796 | 0.906 |
| Snowball stemming (k1 1.8, b 0.5) | 0.698 (0.637 to 0.757) | 0.663 | 0.829 | 0.929 |

The stemmed, tuned BM25 is the baseline every later system is compared with. 98 of 165 queries have the answer at rank 1; 11 have none in the top 100.

## Pretrained embeddings

Exact cosine search over normalised embeddings, title + abstract, CPU. BGE models use the retrieval query prefix.

| model | params | max tokens | corpus encoding | nDCG@10 | recall@100 | vs stemmed BM25 (paired) |
|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 22M | 256 | 98 s | 0.625 (0.563 to 0.687) | 0.943 | -0.072 (-0.122 to -0.020) |
| all-mpnet-base-v2 | 110M | 384 | 1,290 s | 0.622 (0.558 to 0.687) | 0.927 | -0.075 (-0.126 to -0.025) |
| bge-small-en-v1.5 | 33M | 512 | 315 s | 0.724 (0.668 to 0.780) | 0.968 | +0.027 (-0.017 to +0.071) |
| bge-base-en-v1.5 | 110M | 512 | 1,053 s | 0.759 (0.705 to 0.810) | 0.974 | +0.061 (+0.017 to +0.106) |

Size does not explain the ranking: mpnet-base is five times the size of MiniLM and scores the same. Max sequence length, training recipe and the query prefix differ between the families and were not separated. Four models were compared on the same queries, so the best margin is slightly optimistic. Overlap between the BGE training data and SciFact was not checked.

Against unstemmed BM25, the gain of bge-base over BM25 grows as the query shares fewer words with its relevant document: on the lowest overlap tercile MRR is 0.491 against 0.270, on the highest 0.980 against 0.973. BM25 wins mainly on queries with rare exact names. This comparison predates stemming and is to be repeated against the stemmed baseline.

## Hybrid

Stemmed BM25 and bge-base fused by a weighted sum of min-max normalised scores (weight alpha on BM25) or by reciprocal rank fusion (k in {10, 60, 200}).

| system | nDCG@10 | MRR@10 | recall@10 | recall@100 |
|---|---|---|---|---|
| weighted sum, alpha 0.2 | 0.794 (0.742 to 0.842) | 0.764 | 0.908 | 0.968 |
| bge-base alone | 0.759 | n/a | n/a | 0.974 |
| BM25 alone | 0.698 | 0.663 | 0.829 | 0.929 |

Against BM25: +0.096 (+0.063 to +0.130), better on 50 queries, worse on 8. Against bge-base alone: +0.034 (+0.010 to +0.059), better on 32, worse on 16. Every fusion variant is flat between alpha 0.1 and 0.5 (0.76 to 0.79). 44 settings were tried on the same queries, so the maximum is optimistic; RRF without a fitted score scale gives 0.78.

## Fine-tuning

all-MiniLM-L6-v2, 475 clean pairs. The in-batch contrastive loss (cosine similarity times 20, cross-entropy against the diagonal) is written by hand and matches `MultipleNegativesRankingLoss` of sentence-transformers 6.1.0 in value and gradients (tolerance 1e-5 and 1e-6). Batches of up to 16 contain no repeated query or document, and a pair's hard negative is never another pair's positive in the same batch. AdamW, learning rate 2e-5, weight decay 0.01, 10% warmup then linear decay, 4 epochs; the full corpus is re-encoded and scored on validation after every epoch and the best epoch is kept. About 60 s (in-batch) or 100 s (hard negatives) of training and 100 s of validation per epoch on the CPU.

Hard negatives: per pair, one document drawn at random from the top 10 stemmed-BM25 results that are not judged relevant to the query and not relevant to any test query.

| setting | best epoch | nDCG@10 | recall@10 | recall@100 |
|---|---|---|---|---|
| pretrained | n/a | 0.625 (0.563 to 0.687) | 0.760 | 0.943 |
| in-batch negatives | 4 | 0.631 (0.568 to 0.692) | 0.771 | 0.942 |
| hard negatives | 3 | 0.662 (0.601 to 0.722) | 0.793 | 0.898 |

Paired nDCG@10 differences: hard negatives minus pretrained +0.036 (+0.011 to +0.062); hard negatives minus in-batch +0.031 (+0.011 to +0.053); in-batch minus pretrained +0.006 (-0.011 to +0.022); hard negatives minus stemmed BM25 -0.036 (-0.089 to +0.020).

- In-batch negatives alone: training loss falls from 0.215 to 0.07 within two epochs. The validation gain is small and depends on the seed (+0.006 to +0.024, see Seeds), as the 15 random negatives are mostly easy.
- Hard negatives: the gain appears from epoch 2 and is flat over epochs 2 to 4 (0.659 to 0.662). recall@10 rises (0.760 to 0.793) while recall@100 falls (0.943 to about 0.90).
- The fine-tuned MiniLM does not reach BM25, bge-small or bge-base.

### Ablation on the number of train queries (hard negatives, seed 0; three seeds below)

Queries are a seeded sample of the 415 clean train queries. 400 was replaced by 200 because only 415 are usable. Update steps scale with the data (28, 56 and 120), so queries and steps change together.

| train queries | pairs | best nDCG@10 | recall@10 | recall@100 |
|---|---|---|---|---|
| 0 (pretrained) | 0 | 0.625 | 0.760 | 0.943 |
| 100 | 111 | 0.631 | 0.768 | 0.934 |
| 200 | 219 | 0.656 | 0.805 | 0.934 |
| 415 | 475 | 0.662 | 0.793 | 0.898 |

### Seeds

Seeds 0, 1 and 2 for every setting. The seed changes the batch order, the choice of hard negatives and, in the ablation, which queries are sampled. Best-epoch validation nDCG@10 per run (pretrained 0.625):

| setting | seed 0 | seed 1 | seed 2 | mean | recall@10 (mean) | recall@100 (mean) |
|---|---|---|---|---|---|---|
| in-batch negatives, 415 queries | 0.631 | 0.639 | 0.650 | 0.640 | 0.783 | 0.944 |
| hard negatives, 100 queries | 0.631 | 0.635 | 0.642 | 0.636 | 0.770 | 0.929 |
| hard negatives, 200 queries | 0.656 | 0.648 | 0.653 | 0.652 | 0.811 | 0.928 |
| hard negatives, 415 queries | 0.662 | 0.661 | 0.662 | 0.662 | 0.811 | 0.908 |

Paired nDCG@10 differences on the validation queries (95% interval), per seed:

| comparison | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| hard negatives minus pretrained | +0.036 (+0.011 to +0.062) | +0.036 (+0.010 to +0.062) | +0.037 (+0.015 to +0.058) |
| in-batch minus pretrained | +0.006 (-0.011 to +0.022) | +0.014 (-0.002 to +0.029) | +0.024 (+0.007 to +0.041) |
| hard negatives minus in-batch | +0.031 (+0.011 to +0.053) | +0.022 (0.000 to +0.045) | +0.012 (-0.006 to +0.030) |
| hard negatives minus stemmed BM25 | -0.036 (-0.089 to +0.020) | -0.037 (-0.088 to +0.017) | -0.036 (-0.086 to +0.015) |

- Hard negatives with all queries are stable across seeds (0.661 to 0.662) and always beat the pretrained model by about +0.036. They never reach BM25 (-0.036 in every seed, intervals include zero).
- In-batch negatives are not a null result: the gain over pretrained is +0.006, +0.014 and +0.024 depending on the seed, so seed 0 understated it. The in-batch runs spread over 0.019, the hard-negative runs over 0.001.
- Hard negatives beat in-batch in all three seeds, by +0.012 to +0.031 (mean +0.022), but the per-seed interval excludes zero only for seed 0 (seed 1 touches zero, seed 2 includes it). The ordering is consistent; its size is not well pinned down.
- The best epoch is chosen per run on the same validation queries, which adds a little optimism to every row equally. The best epoch of the hard-negative runs was 3 or 4, of the in-batch runs 2 to 4.
- Data size: 100 queries give +0.011 (0.636), 200 give +0.027 (0.652) and 415 give +0.037 (0.662) over pretrained. The spread between seeds is at most 0.011 at 100 queries and 0.001 at 415.
- With all queries, recall@100 falls to 0.908 with hard negatives (0.898 to 0.922) while in-batch keeps 0.944.

## Error analysis: fine-tuned MiniLM against stemmed BM25 (validation, `src/analyze_errors.py`)

Fine-tuned model: MiniLM, hard negatives, all clean queries, seed 0, epoch 3 (nDCG@10 0.662). Reciprocal rank is 1 / rank of the first relevant document over the full ranking of the corpus (1/1000 when it is outside the top 100). Overlap is the share of a query's distinct words that occur in its relevant document. Per-query table in `data/error_analysis_val.csv`.

Rank of the first relevant document (165 queries):

| system | rank 1 | rank 2 to 10 | rank 11 to 100 | not in top 100 |
|---|---|---|---|---|
| stemmed BM25 | 98 | 42 | 14 | 11 |
| MiniLM pretrained | 79 | 48 | 29 | 9 |
| MiniLM fine-tuned | 88 | 44 | 17 | 16 |

Fine-tuned against BM25: better on 39 queries, worse on 46, equal on 80 (70 of them with both at rank 1, 5 with both outside the top 100). Of the 46 losses, 23 still have the fine-tuned model in the top 10 and 28 have BM25 at rank 1; 7 are large (BM25 in the top 3, fine-tuned outside the top 10). Of the 39 wins, 16 have BM25 outside the top 10 and 6 are large (fine-tuned in the top 3, BM25 outside the top 10).

By keyword overlap (terciles of the overlap):

| overlap tercile | queries | mean overlap | BM25 MRR | pretrained MRR | fine-tuned MRR | fine-tuned better | BM25 better |
|---|---|---|---|---|---|---|---|
| low | 63 | 0.35 | 0.340 | 0.343 | 0.413 | 29 | 21 |
| mid | 53 | 0.61 | 0.766 | 0.644 | 0.661 | 9 | 17 |
| high | 49 | 0.80 | 0.976 | 0.860 | 0.873 | 1 | 8 |

- The fine-tuned model is ahead only on the low-overlap third. In the high-overlap third, where the document repeats most of the query's words, BM25 is near perfect (0.976) and the fine-tuned model is not (0.873, +0.013 over pretrained). Fine-tuning raised all three terciles, most on the low one (+0.070). The correlation between (fine-tuned minus BM25) reciprocal rank and overlap is -0.22.
- The 10 largest BM25 wins were read. In 8, the fine-tuned top result is a paper on the same topic that is not the one the claim comes from (panic anxiety returns a prevalence review for the orexin paper; H. pylori and gastric cancer returns a mouse H. pylori mutation paper; bariatric surgery and colorectal cancer returns a colorectal cancer survival paper). In 2, the claim hinges on a rare name (HOXB4; Esrrb with Mbd3), BM25 puts the paper at rank 1, and the fine-tuned model returns an unrelated paper and misses the top 100.
- The 10 largest fine-tuned wins were read. In 8, the claim and the document use different words for the same thing: students against pupils (bullying), OCT3/4 against Oct4, a drug name against the transport process it inhibits (AM404 and anandamide), "neural progenitors differ across species" against "species-specific differences in progenitor behavior". In 6 of the 10, BM25's top result is off topic and matches only a few words of the claim ("pioneer factor" gives a Pax7 paper; the negated "no known interaction between Deltex and eIF3" gives a vitamin D paper); in the other 4, BM25 has the right topic but ranks the paper 4 to 9.
- Claims that differ only in direction (male/female swapped, positively/negatively related, "interacts" against "no known interaction") get the same rank for the same document in all three pairs found in validation. In SciFact the relevant paper is the same for both polarities, so this is not penalised here.
- Fine-tuning moves 9 queries to rank 1 relative to pretrained but pushes 7 more queries out of the top 100 (9 to 16), the same effect as the lower recall@100 above. The queries were picked by size of the rank gap, so rank-1 BM25 results are over-represented among the losses by construction.
- One model, one seed, 165 queries; the 20 queries were read by one person. The bge-base comparison against stemmed BM25 by overlap tercile (above, "Per-query comparison", run before stemming) has not been repeated.

## Not done yet

Error analysis: keyword-overlap probes (words removed or replaced by synonyms), recall@100 ceiling, test scores for each final system, exact versus approximate search timing, API and page.
