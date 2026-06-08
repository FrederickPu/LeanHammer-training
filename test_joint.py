"""
End-to-end tests for train_joint_premise_draft.py.

Sections
--------
1. Helpers
2. contrastive_loss unit tests  — regression vs loss.py
3. Collate unit tests           — shapes, mask, draft labels
4. Integration                  — collate mask -> contrastive_loss; backward; vs repo loss
"""

import random
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from loss import MaskedCachedMultipleNegativesRankingLoss
from sentence_transformers.util import cos_sim
from train_joint_premise_draft import Collate, Cfg, contrastive_loss


# ── 1. Helpers ────────────────────────────────────────────────────────────────

class _Tok:
    """Fake tokenizer: maps each char to ord(c) % 90 + 10 (avoids special ids 0/1/2)."""
    pad_token_id = 0
    eos_token_id = 1
    def add_special_tokens(self, d): pass
    def convert_tokens_to_ids(self, t): return 2   # EMB id
    def __call__(self, text, **kw):
        return SimpleNamespace(input_ids=[ord(c) % 90 + 10 for c in text])


class _RepoLoss:
    """Exposes calculate_loss / calculate_loss_and_cache_gradients from loss.py
    without MaskedCachedMultipleNegativesRankingLoss.__init__ (requires a live
    SentenceTransformer model whose API changed in ST 5.x)."""
    calculate_loss = MaskedCachedMultipleNegativesRankingLoss.calculate_loss
    calculate_loss_and_cache_gradients = MaskedCachedMultipleNegativesRankingLoss.calculate_loss_and_cache_gradients

    def __init__(self, temp, B):
        self.scale              = 1.0 / temp
        self.similarity_fct     = cos_sim
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mini_batch_size    = B + 1
        self.show_progress_bar  = False


def make_loss_fn(temp, B):
    return _RepoLoss(temp, B)


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


def _collate(n_neg=1, library=("nA", "nB", "nC", "nD")):
    return Collate(_Tok(), Cfg(n_neg=n_neg, max_len=512), list(library))


# ── 2. contrastive_loss unit tests ───────────────────────────────────────────

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


# ── 3. Collate unit tests ─────────────────────────────────────────────────────

def test_collate_output_shapes():
    B, n_neg = 3, 2
    col = _collate(n_neg=n_neg)
    batch = [
        {"retrieval_query": f"rq{i}", "draft_query": f"dq{i}",
         "draft": f"out{i}", "premises": [f"p{i}"]}
        for i in range(B)
    ]
    random.seed(0)
    out = col(batch)
    n1 = 1 + n_neg
    Lp = out["premise_ids"].shape[2]

    assert out["retrieval_query_ids"].shape[0] == B
    assert out["premise_ids"].shape == torch.Size([B, n1, Lp])
    assert out["retrieval_mask"].shape == torch.Size([B, B * n1])
    assert out["draft_labels"].shape == out["draft_query_ids"].shape
    print("PASS test_collate_output_shapes")


def test_collate_mask_shared_positive():
    """mask[i, qi*n1] is False when qi's positive is also gold for query i."""
    col = _collate(n_neg=1)
    batch = [
        {"retrieval_query": "rq0", "draft_query": "dq0", "draft": "out0", "premises": ["shared"]},
        {"retrieval_query": "rq1", "draft_query": "dq1", "draft": "out1", "premises": ["shared"]},
    ]
    random.seed(0)
    mask = col(batch)["retrieval_mask"]   # (2, 4),  n1=2
    n1 = 2

    assert not mask[0, 1 * n1 + 0].item(), "row 1's positive should be masked for row 0"
    assert not mask[1, 0 * n1 + 0].item(), "row 0's positive should be masked for row 1"
    assert mask[0, 0 * n1 + 0].item(),     "own positive must be unmasked"
    assert mask[1, 1 * n1 + 0].item(),     "own positive must be unmasked"
    assert mask[0, 0 * n1 + 1].item(),     "own negative must be unmasked"
    assert mask[0, 1 * n1 + 1].item(),     "cross negative must be unmasked"
    print("PASS test_collate_mask_shared_positive")


def test_collate_mask_no_false_negatives():
    """When all positives are distinct, retrieval_mask is all True."""
    col = _collate(n_neg=1)
    batch = [
        {"retrieval_query": "rq0", "draft_query": "dq0", "draft": "out0", "premises": ["p0"]},
        {"retrieval_query": "rq1", "draft_query": "dq1", "draft": "out1", "premises": ["p1"]},
    ]
    random.seed(0)
    mask = col(batch)["retrieval_mask"]
    assert mask.all().item(), "no shared premises -> mask should be all True"
    print("PASS test_collate_mask_no_false_negatives")


def test_collate_draft_labels():
    """-100 on the draft_query prefix; actual token ids on the draft span."""
    col = _collate(n_neg=1)
    batch = [{"retrieval_query": "rq", "draft_query": "Q", "draft": "AB", "premises": ["p"]}]
    random.seed(0)
    labels = col(batch)["draft_labels"][0]
    q_len = len([ord(c) % 90 + 10 for c in "Q"])   # 1
    assert (labels[:q_len] == -100).all(), "draft_query prefix must be -100"
    assert (labels[q_len:] != -100).all(), "draft span must not be -100"
    print("PASS test_collate_draft_labels")


def test_collate_retrieval_query_ends_with_emb():
    """retrieval_query_ids last real token is the EMB token (id=2)."""
    col = _collate(n_neg=1)
    batch = [{"retrieval_query": "hello", "draft_query": "dq", "draft": "out", "premises": ["p"]}]
    random.seed(0)
    out  = col(batch)
    ids  = out["retrieval_query_ids"][0]
    attn = out["retrieval_query_mask"][0]
    last_real = ids[attn.sum() - 1].item()
    assert last_real == 2, f"expected EMB id=2, got {last_real}"
    print("PASS test_collate_retrieval_query_ends_with_emb")


# ── 4. Integration ────────────────────────────────────────────────────────────

def test_end_to_end_masking_changes_loss():
    """Collate mask -> contrastive_loss: false-negative masking changes the loss value."""
    torch.manual_seed(3)
    B, n_neg, H, temp = 2, 1, 8, 0.05

    col = _collate(n_neg=n_neg)
    batch = [
        {"retrieval_query": "rq0", "draft_query": "dq0", "draft": "d0", "premises": ["shared"]},
        {"retrieval_query": "rq1", "draft_query": "dq1", "draft": "d1", "premises": ["shared"]},
    ]
    random.seed(0)
    mask = col(batch)["retrieval_mask"]   # has False at [0,2] and [1,0]

    q     = F.normalize(torch.randn(B, H), dim=-1)
    p_row = F.normalize(torch.randn(B * (1 + n_neg), H), dim=-1)

    loss_masked   = contrastive_loss(q, p_row, mask,                      temp)
    loss_unmasked = contrastive_loss(q, p_row, torch.ones_like(mask), temp)
    assert not torch.allclose(loss_masked, loss_unmasked), \
        "masking should change the loss when premises are shared"
    print("PASS test_end_to_end_masking_changes_loss")


def test_end_to_end_gradients_flow():
    """Gradients flow through contrastive_loss with a collator-produced mask."""
    torch.manual_seed(3)
    B, n_neg, H, temp = 2, 1, 8, 0.05

    col = _collate(n_neg=n_neg)
    batch = [
        {"retrieval_query": "rq0", "draft_query": "dq0", "draft": "d0", "premises": ["shared"]},
        {"retrieval_query": "rq1", "draft_query": "dq1", "draft": "d1", "premises": ["shared"]},
    ]
    random.seed(0)
    mask = col(batch)["retrieval_mask"]

    q_leaf = torch.randn(B, H, requires_grad=True)
    p_leaf = torch.randn(B * (1 + n_neg), H, requires_grad=True)
    contrastive_loss(q_leaf, p_leaf, mask, temp).backward()
    assert q_leaf.grad is not None and p_leaf.grad is not None
    print("PASS test_end_to_end_gradients_flow")


def test_end_to_end_loss_matches_repo():
    """Collate mask + contrastive_loss == _RepoLoss.calculate_loss (col-major).

    Verifies that the full pipeline — collator-produced mask, row-major premise
    layout, contrastive_loss — produces the same value as the original repo's
    loss once the layout is converted to column-major.
    """
    torch.manual_seed(4)
    B, n_neg, H, temp = 3, 2, 16, 0.05
    n1 = 1 + n_neg

    col = _collate(n_neg=n_neg)
    # Row 0 and row 2 share "shared" as their positive -> two false negatives in mask
    batch = [
        {"retrieval_query": "rq0", "draft_query": "dq0", "draft": "d0",
         "premises": ["shared", "extra0"]},
        {"retrieval_query": "rq1", "draft_query": "dq1", "draft": "d1",
         "premises": ["p1"]},
        {"retrieval_query": "rq2", "draft_query": "dq2", "draft": "d2",
         "premises": ["shared", "extra2"]},
    ]
    random.seed(0)
    mask = col(batch)["retrieval_mask"]

    q     = F.normalize(torch.randn(B, H), dim=-1)
    p_row = F.normalize(torch.randn(B * n1, H), dim=-1)

    our = contrastive_loss(q, p_row, mask, temp)

    p_col, mask_col = to_col_major(p_row, mask, B, n1)
    ref = make_loss_fn(temp, B).calculate_loss(make_reps(q, p_col, B, n1), mask_col)

    assert torch.allclose(our, ref, atol=1e-5), f"ours={our:.6f}  repo={ref:.6f}"
    print(f"PASS test_end_to_end_loss_matches_repo          loss={our:.6f}")


if __name__ == "__main__":
    # contrastive_loss unit tests
    test_loss_values_no_mask()
    test_loss_values_with_mask()
    test_cached_gradients_match_reference()
    # Collate unit tests
    test_collate_output_shapes()
    test_collate_mask_shared_positive()
    test_collate_mask_no_false_negatives()
    test_collate_draft_labels()
    test_collate_retrieval_query_ends_with_emb()
    # Integration
    test_end_to_end_masking_changes_loss()
    test_end_to_end_gradients_flow()
    test_end_to_end_loss_matches_repo()
    print("\nAll tests passed.")
