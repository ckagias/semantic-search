import math

import pytest
import torch
from sentence_transformers import losses

from src.loss import in_batch_loss

LIBRARY = losses.MultipleNegativesRankingLoss(None)


def test_aligned_pairs_by_hand():
    eye = torch.eye(2)
    assert in_batch_loss(eye, eye, scale=1.0).item() == pytest.approx(math.log(1 + math.exp(-1)))


def test_swapped_pairs_by_hand():
    eye = torch.eye(2)
    swapped = eye.flip(0)
    assert in_batch_loss(eye, swapped, scale=1.0).item() == pytest.approx(math.log(1 + math.e))


def test_identical_documents_give_log_batch_size():
    docs = torch.ones(4, 8)
    queries = torch.randn(4, 8)
    assert in_batch_loss(queries, docs).item() == pytest.approx(math.log(4))


def test_duplicate_document_in_batch_puts_a_floor_under_the_loss():
    docs = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    queries = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    assert in_batch_loss(queries, docs).item() == pytest.approx(math.log(2))


def test_vector_length_does_not_matter():
    q, d = torch.randn(5, 16), torch.randn(5, 16)
    assert in_batch_loss(q, d).item() == pytest.approx(in_batch_loss(3 * q, 0.1 * d).item())


def test_better_alignment_gives_lower_loss():
    d = torch.randn(8, 32)
    noise = torch.randn(8, 32)
    assert in_batch_loss(d + 0.1 * noise, d) < in_batch_loss(d + 3 * noise, d)


@pytest.mark.parametrize("batch,dim,scale", [(2, 4, 20.0), (6, 384, 20.0), (16, 64, 5.0)])
def test_matches_sentence_transformers_value_and_gradient(batch, dim, scale):
    torch.manual_seed(batch)
    q1 = torch.randn(batch, dim, requires_grad=True)
    d1 = torch.randn(batch, dim, requires_grad=True)
    q2 = q1.detach().clone().requires_grad_(True)
    d2 = d1.detach().clone().requires_grad_(True)

    mine = in_batch_loss(q1, d1, scale=scale)
    library = losses.MultipleNegativesRankingLoss(None, scale=scale).compute_loss_from_embeddings([q2, d2], None)
    assert mine.item() == pytest.approx(library.item(), rel=1e-5)

    mine.backward()
    library.backward()
    assert torch.allclose(q1.grad, q2.grad, atol=1e-6)
    assert torch.allclose(d1.grad, d2.grad, atol=1e-6)
