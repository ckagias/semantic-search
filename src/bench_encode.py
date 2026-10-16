import time

import torch
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
N = 1000
BATCH = 64

SENTENCE = (
    "Mutations in the BRCA1 gene increase the risk of breast and ovarian cancer "
    "in women who carry them, according to a large cohort study"
)


def main():
    t = time.perf_counter()
    model = SentenceTransformer(MODEL, device="cpu")
    print(f"load: {time.perf_counter() - t:.1f}s, threads: {torch.get_num_threads()}")

    vec = model.encode(SENTENCE)
    print(f"one sentence: shape {vec.shape}, norm {float((vec ** 2).sum() ** 0.5):.3f}")

    sentences = [f"{SENTENCE} (sample {i})" for i in range(N)]
    model.encode(sentences[:BATCH], batch_size=BATCH)
    times = []
    for _ in range(3):
        t = time.perf_counter()
        model.encode(sentences, batch_size=BATCH)
        times.append(time.perf_counter() - t)
    print(f"{N} sentences: " + ", ".join(f"{x:.1f}s" for x in times))


if __name__ == "__main__":
    main()
