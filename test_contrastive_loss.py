"""
Regression tests for contrastive_loss in train_joint_premise_draft.py against
MaskedCachedMultipleNegativesRankingLoss in loss.py.

Both losses compute the same CrossEntropy-over-cosine-scores formula but use
different premise orderings in the flat embedding matrix:

  train_joint (row-major):   p[i*n1 + c]  = query i's c-th premise  (c=0 positive)
  loss.py     (col-major):   p[c*B  + i]  = query i's c-th premise

labels[i] = i*n1  (row-major)   vs   labels[i] = i  (col-major)
"""

import torch
import torch.nn.functional as F

from loss import MaskedCachedMultipleNegativesRankingLoss
from sentence_transformers.util import cos_sim
from train_joint_premise_draft import contrastive_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _RepoLoss:
    """Thin wrapper that exposes only the two methods under test from loss.py.

    MaskedCachedMultipleNegativesRankingLoss.__init__ requires a live
    SentenceTransformer model whose API has changed across ST versions.
    calculate_loss and calculate_loss_and_cache_gradients only read
    self.scale / similarity_fct / cross_entropy_loss / mini_batch_size /
    show_progress_bar — never self.model — so we set those directly.
    """
    calculate_loss                  = MaskedCachedMultipleNegativesRankingLoss.calculate_loss
    calculate_loss_and_cache_gradients = MaskedCachedMultipleNegativesRankingLoss.calculate_loss_and_cache_gradients

    def __init__(self, temp, B):
        self.scale              = 1.0 / temp
        self.similarity_fct     = cos_sim
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mini_batch_size    = B + 1   # whole batch in one shot
        self.show_progress_bar  = False


def make_loss_fn(temp, B):
    return _RepoLoss(temp, B)


def make_reps(q, p_col, B, n1, requires_grad=False):
    """Build the reps list expected by loss.py from column-major premises.

    reps[0] = [q]                  queries     (B, H)
    reps[1] = [p_col[0:B]]         positives   (B, H)
    reps[c] = [p_col[(c-1)*B:c*B]] (c-1)-th negatives
    """
    def leaf(t):
        t = t.detach().clone()
        return t.requires_grad_(True) if requires_grad else t

    reps = [[leaf(q)]]
    for c in range(n1):
        reps.append([leaf(p_col[c * B : (c + 1) * B])])
    return reps


def to_col_major(p_row, mask_row, B, n1):
    """Convert row-major premises/mask to the column-major layout of loss.py.

    Row-major:    p_row[i*n1 + c]        = query i's c-th premise
    Column-major: p_col[c*B  + i]        = query i's c-th premise
    mask_row[i, qi*n1 + c] -> mask_col[i, c*B + qi]
    """
    p_col = p_row.view(B, n1, -1).permute(1, 0, 2).reshape(B * n1, -1)
    mask_col = torch.zeros_like(mask_row)
    for qi in range(B):
        for c in range(n1):
            mask_col[:, c * B + qi] = mask_row[:, qi * n1 + c]
    return p_col, mask_col


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_loss_values_no_mask():
    """Loss values match loss.py with an all-True retrieval mask."""
    torch.manual_seed(0)
    B, n_neg, H, temp = 4, 2, 16, 0.05
    n1 = 1 + n_neg

    q     = F.normalize(torch.randn(B, H), dim=-1)
    p_row = F.normalize(torch.randn(B * n1, H), dim=-1)
    mask  = torch.ones(B, B * n1, dtype=torch.bool)

    our = contrastive_loss(q, p_row, mask, temp)

    p_col, mask_col = to_col_major(p_row, mask, B, n1)
    fn  = make_loss_fn(temp, B)
    ref = fn.calculate_loss(make_reps(q, p_col, B, n1), mask_col)

    assert torch.allclose(our, ref, atol=1e-5), f"ours={our:.6f}  repo={ref:.6f}"
    print(f"PASS test_loss_values_no_mask           loss={our:.6f}")


def test_loss_values_with_mask():
    """Loss values match loss.py with some false negatives masked out."""
    torch.manual_seed(1)
    B, n_neg, H, temp = 4, 2, 16, 0.05
    n1 = 1 + n_neg

    q     = F.normalize(torch.randn(B, H), dim=-1)
    p_row = F.normalize(torch.randn(B * n1, H), dim=-1)
    mask  = torch.ones(B, B * n1, dtype=torch.bool)
    mask[0, 1 * n1 + 0] = False   # query 1's positive is also gold for query 0
    mask[2, 3 * n1 + 1] = False   # query 3's neg_0 is also gold for query 2

    our = contrastive_loss(q, p_row, mask, temp)

    p_col, mask_col = to_col_major(p_row, mask, B, n1)
    fn  = make_loss_fn(temp, B)
    ref = fn.calculate_loss(make_reps(q, p_col, B, n1), mask_col)

    assert torch.allclose(our, ref, atol=1e-5), f"ours={our:.6f}  repo={ref:.6f}"
    print(f"PASS test_loss_values_with_mask         loss={our:.6f}")


def test_cached_gradients_match_reference():
    """calculate_loss_and_cache_gradients caches same grads as calculate_loss.backward()."""
    torch.manual_seed(2)
    B, n_neg, H, temp = 4, 2, 16, 0.05
    n1 = 1 + n_neg

    q_data   = F.normalize(torch.randn(B, H), dim=-1)
    p_col    = F.normalize(torch.randn(B * n1, H), dim=-1)
    mask_col = torch.ones(B, B * n1, dtype=torch.bool)
    mask_col[0, B + 1] = False

    fn = make_loss_fn(temp, B)

    # Reference: calculate_loss + explicit backward
    reps_ref = make_reps(q_data, p_col, B, n1, requires_grad=True)
    loss_ref = fn.calculate_loss(reps_ref, mask_col)
    loss_ref.backward()
    ref_grads = [[r.grad.clone() for r in col] for col in reps_ref]

    # Cached path: calculate_loss_and_cache_gradients
    reps_cached = make_reps(q_data, p_col, B, n1, requires_grad=True)
    loss_cached = fn.calculate_loss_and_cache_gradients(reps_cached, mask_col)

    assert torch.allclose(loss_ref, loss_cached, atol=1e-5), \
        f"loss mismatch: ref={loss_ref:.6f}  cached={loss_cached:.6f}"

    for c, (ref_col, cache_col) in enumerate(zip(ref_grads, fn.cache)):
        for mb, (ref_g, cached_g) in enumerate(zip(ref_col, cache_col)):
            max_diff = (ref_g - cached_g).abs().max().item()
            assert max_diff < 1e-5, \
                f"grad mismatch at reps[{c}][{mb}]: max_diff={max_diff:.2e}"

    print(f"PASS test_cached_gradients_match_reference  loss={loss_ref:.6f}")


if __name__ == "__main__":
    test_loss_values_no_mask()
    test_loss_values_with_mask()
    test_cached_gradients_match_reference()
    print("\nAll tests passed.")
