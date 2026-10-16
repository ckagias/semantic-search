import torch
import torch.nn.functional as F


def in_batch_loss(queries, docs, scale=20.0):
    queries = F.normalize(queries, dim=-1)
    docs = F.normalize(docs, dim=-1)
    logits = scale * queries @ docs.T
    return F.cross_entropy(logits, torch.arange(len(queries)))
