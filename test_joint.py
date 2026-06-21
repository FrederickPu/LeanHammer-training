"""
End-to-end tests for train_joint_premise_draft.py.

Sections
--------
1. Helpers
2. contrastive_loss unit tests  — regression vs loss.py
3. Collate unit tests           — shapes, gold_mask, draft labels
4. Integration                  — collate gold_mask -> contrastive_loss; backward; vs repo loss
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


def make_reps(q, p_col, B, n1, requires_grad=False):
    """Build the reps list expected by loss.py.

    p_col is in col-major layout: p_col[c*B:(c+1)*B] = c-th premise for each query.
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


def gold_to_repo_mask(gold_mask):
    """Convert gold_mask -> repo_mask for _RepoLoss.calculate_loss.

    gold_mask[i,j]=True: premise j is gold for query i (labeled pos or false neg).
    repo_mask[i,j]=True: include column j in denominator for query i.
    Labeled positives (diagonal) are always included; false negatives excluded.
    """
    B = gold_mask.shape[0]
    repo_mask = ~gold_mask.clone()
    repo_mask[torch.arange(B), torch.arange(B)] = True
    return repo_mask


def _collate(n_neg=1):
    library = [f"neg{i}" for i in range(40)]   # large enough for any test
    return Collate(_Tok(), Cfg(n_neg=n_neg, max_len=512), library)


def _row(rq, premise, dq="dq", draft="d"):
    return {"retrieval_query": rq, "premise": premise, "draft_query": dq, "draft": draft}


# ── 2. contrastive_loss unit tests ───────────────────────────────────────────

def test_loss_values_no_mask():
    """Loss matches loss.py with diagonal-only gold_mask (no false negatives)."""
    torch.manual_seed(0)
    B, n_neg, H, temp = 4, 2, 16, 0.05
    n1 = 1 + n_neg

    q     = F.normalize(torch.randn(B, H), dim=-1)
    p_all = F.normalize(torch.randn(B * n1, H), dim=-1)  # col-major: p_all[0:B]=positives

    gold_mask = torch.zeros(B, B * n1, dtype=torch.bool)
    gold_mask[torch.arange(B), torch.arange(B)] = True   # only labeled positives

    our = contrastive_loss(q, p_all, gold_mask, temp)

    # p_all is already col-major so make_reps works directly
    repo_mask = gold_to_repo_mask(gold_mask)   # all True
    ref = make_loss_fn(temp, B).calculate_loss(make_reps(q, p_all, B, n1), repo_mask)

    assert torch.allclose(our, ref, atol=1e-5), f"ours={our:.6f}  repo={ref:.6f}"
    print(f"PASS test_loss_values_no_mask           loss={our:.6f}")


def test_loss_values_with_mask():
    """Loss matches loss.py when some off-diagonal entries are gold (false negatives)."""
    torch.manual_seed(1)
    B, n_neg, H, temp = 4, 2, 16, 0.05
    n1 = 1 + n_neg

    q     = F.normalize(torch.randn(B, H), dim=-1)
    p_all = F.normalize(torch.randn(B * n1, H), dim=-1)

    gold_mask = torch.zeros(B, B * n1, dtype=torch.bool)
    gold_mask[torch.arange(B), torch.arange(B)] = True
    gold_mask[0, 1] = True   # query 1's positive is also gold for query 0
    gold_mask[2, 3] = True   # query 3's positive is also gold for query 2

    our = contrastive_loss(q, p_all, gold_mask, temp)

    repo_mask = gold_to_repo_mask(gold_mask)
    ref = make_loss_fn(temp, B).calculate_loss(make_reps(q, p_all, B, n1), repo_mask)

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
    batch = [_row(f"rq{i}", f"p{i}", f"dq{i}", f"out{i}") for i in range(B)]
    random.seed(0)
    out = col(batch)
    M = B + n_neg * B   # B positives + n_neg*B negatives

    assert out["retrieval_query_ids"].shape[0] == B
    assert out["premise_ids"].shape[0] == M           # flat (M, L), not (B, n1, L)
    assert out["gold_mask"].shape == torch.Size([B, M])
    assert out["draft_labels"].shape == out["draft_query_ids"].shape
    print("PASS test_collate_output_shapes")


def test_collate_mask_shared_goal():
    """Rows with the same retrieval_query mark each other's positives as gold (false neg)."""
    col = _collate(n_neg=1)
    batch = [_row("same_goal", "prem0"), _row("same_goal", "prem1")]
    random.seed(0)
    gold_mask = col(batch)["gold_mask"]   # (2, M)

    assert gold_mask[0, 0].item(),  "own positive must be gold"
    assert gold_mask[1, 1].item(),  "own positive must be gold"
    assert gold_mask[0, 1].item(),  "row 1's positive must be gold for row 0 (same goal)"
    assert gold_mask[1, 0].item(),  "row 0's positive must be gold for row 1 (same goal)"
    # Shared negatives (columns 2+) must not be marked gold
    assert not gold_mask[0, 2].item(), "shared negative must not be gold"
    assert not gold_mask[1, 2].item(), "shared negative must not be gold"
    print("PASS test_collate_mask_shared_goal")


def test_collate_mask_distinct_goals():
    """Rows with different retrieval_query have only diagonal gold."""
    col = _collate(n_neg=1)
    batch = [_row("goal0", "p0"), _row("goal1", "p1")]
    random.seed(0)
    gold_mask = col(batch)["gold_mask"]
    B, M = gold_mask.shape
    expected = torch.zeros(B, M, dtype=torch.bool)
    expected[torch.arange(B), torch.arange(B)] = True
    assert (gold_mask == expected).all(), "distinct goals -> only diagonal should be gold"
    print("PASS test_collate_mask_distinct_goals")


def test_collate_draft_labels():
    """-100 on the draft_query prefix; actual token ids on the draft span."""
    col = _collate(n_neg=1)
    batch = [_row("rq", "p", dq="Q", draft="AB")]
    random.seed(0)
    labels = col(batch)["draft_labels"][0]
    q_len = len([ord(c) % 90 + 10 for c in "Q"])   # 1
    assert (labels[:q_len] == -100).all(), "draft_query prefix must be -100"
    assert (labels[q_len:] != -100).all(), "draft span must not be -100"
    print("PASS test_collate_draft_labels")


def test_collate_retrieval_query_ends_with_emb():
    """retrieval_query_ids last real token is the EMB token (id=2)."""
    col = _collate(n_neg=1)
    batch = [_row("hello", "p")]
    random.seed(0)
    out  = col(batch)
    ids  = out["retrieval_query_ids"][0]
    attn = out["retrieval_query_mask"][0]
    last_real = ids[attn.sum() - 1].item()
    assert last_real == 2, f"expected EMB id=2, got {last_real}"
    print("PASS test_collate_retrieval_query_ends_with_emb")


# ── 4. Integration ────────────────────────────────────────────────────────────

def test_end_to_end_masking_changes_loss():
    """gold_mask from collator changes loss when rows share the same goal."""
    torch.manual_seed(3)
    B, n_neg, H, temp = 2, 1, 8, 0.05

    col = _collate(n_neg=n_neg)
    batch = [_row("same_goal", "prem0"), _row("same_goal", "prem1")]
    random.seed(0)
    gold_mask = col(batch)["gold_mask"]   # has True at [0,1] and [1,0]

    M = gold_mask.shape[1]
    q     = F.normalize(torch.randn(B, H), dim=-1)
    p_all = F.normalize(torch.randn(M, H), dim=-1)

    gold_diag = torch.zeros_like(gold_mask)
    gold_diag[torch.arange(B), torch.arange(B)] = True

    loss_masked   = contrastive_loss(q, p_all, gold_mask, temp)
    loss_unmasked = contrastive_loss(q, p_all, gold_diag, temp)
    assert not torch.allclose(loss_masked, loss_unmasked), \
        "masking should change the loss when goals are shared"
    print("PASS test_end_to_end_masking_changes_loss")


def test_end_to_end_gradients_flow():
    """Gradients flow through contrastive_loss with a collator-produced gold_mask."""
    torch.manual_seed(3)
    B, n_neg, H, temp = 2, 1, 8, 0.05

    col = _collate(n_neg=n_neg)
    batch = [_row("same_goal", "prem0"), _row("same_goal", "prem1")]
    random.seed(0)
    gold_mask = col(batch)["gold_mask"]

    M = gold_mask.shape[1]
    q_leaf = torch.randn(B, H, requires_grad=True)
    p_leaf = torch.randn(M, H, requires_grad=True)
    contrastive_loss(q_leaf, p_leaf, gold_mask, temp).backward()
    assert q_leaf.grad is not None and p_leaf.grad is not None
    print("PASS test_end_to_end_gradients_flow")


def test_end_to_end_loss_matches_repo():
    """Collate gold_mask + contrastive_loss == _RepoLoss.calculate_loss.

    Verifies the full pipeline end-to-end: collator-produced gold_mask feeds into
    contrastive_loss and matches the original repo loss after mask conversion.
    Rows 0 and 2 share the same goal, producing two false negatives in the mask.
    """
    torch.manual_seed(4)
    B, n_neg, H, temp = 3, 2, 16, 0.05
    n1 = 1 + n_neg

    col = _collate(n_neg=n_neg)
    batch = [
        _row("goal_A", "prem0"),
        _row("goal_B", "prem1"),
        _row("goal_A", "prem2"),   # same goal as row 0 -> mutual false negatives
    ]
    random.seed(0)
    gold_mask = col(batch)["gold_mask"]   # (3, M), M = B*n1 = 9

    M = gold_mask.shape[1]
    q     = F.normalize(torch.randn(B, H), dim=-1)
    p_all = F.normalize(torch.randn(M, H), dim=-1)

    our = contrastive_loss(q, p_all, gold_mask, temp)

    # p_all layout: p_all[0:B]=positives, p_all[B:M]=negatives by column -> col-major
    repo_mask = gold_to_repo_mask(gold_mask)
    ref = make_loss_fn(temp, B).calculate_loss(make_reps(q, p_all, B, n1), repo_mask)

    assert torch.allclose(our, ref, atol=1e-5), f"ours={our:.6f}  repo={ref:.6f}"
    print(f"PASS test_end_to_end_loss_matches_repo          loss={our:.6f}")


if __name__ == "__main__":
    # contrastive_loss unit tests
    test_loss_values_no_mask()
    test_loss_values_with_mask()
    test_cached_gradients_match_reference()
    # Collate unit tests
    test_collate_output_shapes()
    test_collate_mask_shared_goal()
    test_collate_mask_distinct_goals()
    test_collate_draft_labels()
    test_collate_retrieval_query_ends_with_emb()
    # Integration
    test_end_to_end_masking_changes_loss()
    test_end_to_end_gradients_flow()
    test_end_to_end_loss_matches_repo()
    print("\nAll tests passed.")
