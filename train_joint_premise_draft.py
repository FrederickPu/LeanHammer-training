"""
Joint training of one Qwen backbone on two objectives per batch:
draft generation (cross-entropy) and premise selection (InfoNCE).

Retrieval rows {retrieval_query, premise} come from the premises dataset — one row
per (goal, premise) pair. Draft rows {draft_query, draft} come from the drafts dataset
and are sampled independently (no shared tactic state). DecoupledDataset merges them.
"""

import json, os, random, torch
import torch.nn.functional as F
from dataclasses import dataclass
from datasets import load_dataset, Dataset as HFDataset
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

EMB = "[EMB]"


@dataclass
class Cfg:
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    temp: float = 0.05      # InfoNCE temperature
    lam: float = 1.0        # weight on the premise loss
    n_neg: int = 3          # library negatives per query
    max_len: int = 1024


def setup(cfg):
    tok = AutoTokenizer.from_pretrained(cfg.model)
    tok.add_special_tokens({"additional_special_tokens": [EMB]})
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=torch.bfloat16)
    model.resize_token_embeddings(len(tok))
    return model, tok


def embed(model, ids, mask):
    # Use the base transformer directly to get only last_hidden_state, avoiding the
    # 25× memory overhead of output_hidden_states=True (all-layers tensors).
    # Tries .model (Qwen2) then .transformer (GPT-2); extend for other architectures.
    base = getattr(model, "model", None) or model.transformer
    h = base(input_ids=ids, attention_mask=mask).last_hidden_state
    pos = mask.sum(1) - 1                       # last real token
    return F.normalize(h[torch.arange(ids.size(0), device=ids.device), pos], dim=-1)


def contrastive_loss(q, p_all, gold_mask, temp):
    """Single-positive InfoNCE with cross-batch negatives and false-negative masking.

    q:         (B, H)
    p_all:     (M, H)  premise pool — positions 0..B-1 are each query's labeled positive,
                       remaining positions are library negatives
    gold_mask: (B, M)  True where premise j is also gold for query i

    Labeled positive for query i is at index i. Any other position that is gold
    for query i is a false negative and is masked out of the denominator.
    """
    B = len(q)
    scores  = q @ p_all.T / temp                        # (B, M)
    fn_mask = gold_mask.clone()
    fn_mask[torch.arange(B), torch.arange(B)] = False  # labeled positives are not false negatives
    scores[fn_mask] = float("-inf")
    return F.cross_entropy(scores, torch.arange(B, device=q.device))


def pad(seqs, fill):
    n = max(map(len, seqs))
    return torch.tensor([s + [fill] * (n - len(s)) for s in seqs])


class DecoupledDataset(TorchDataset):
    """Pairs each retrieval row with a randomly sampled draft row.

    Retrieval rows and draft rows come from different datasets with no shared
    tactic state, so they are sampled independently. Each __getitem__ call
    picks a fresh random draft, giving different pairings across epochs.
    """
    def __init__(self, retrieval_ds, draft_rows):
        self.retrieval_ds = retrieval_ds
        self.draft_rows   = draft_rows

    def __len__(self):
        return len(self.retrieval_ds)

    def __getitem__(self, idx):
        return {**self.retrieval_ds[idx], **random.choice(self.draft_rows)}


class Collate:
    def __init__(self, tok, cfg, library):
        self.tok, self.cfg, self.library = tok, cfg, library
        self.emb, self.eos, self.padid = (tok.convert_tokens_to_ids(EMB),
                                          tok.eos_token_id, tok.pad_token_id)

    def enc(self, text):
        return self.tok(text, add_special_tokens=False, truncation=True,
                        max_length=self.cfg.max_len).input_ids

    def enc_query(self, text):
        """Encode a retrieval query using the chat template (same prefix as draft queries)."""
        if getattr(self.tok, "chat_template", None):
            text = self.tok.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False, add_generation_prompt=True,
            )
        return self.enc(text)

    def __call__(self, batch):
        """batch: list of B rows from DecoupledDataset.
        Each row has one labeled positive: {retrieval_query, premise}
        plus draft fields {draft_query, draft} from an independently sampled proof state.

        Premise pool layout: positions 0..B-1 are each query's labeled positive;
        positions B.. are library negatives. gold_mask covers the full pool.

        Returns:
            retrieval_query_ids/mask  (B, Lq)
            premise_ids/mask          (M, Lp)
            gold_mask                 (B, M)
            draft_query_ids/mask      (B, Ld)
            draft_labels              (B, Ld)"""
        B = len(batch)
        retrieval_query = [self.enc_query(b["retrieval_query"]) for b in batch]
        draft_query     = [self.enc(b["draft_query"])     for b in batch]
        draft           = [self.enc(b["draft"]) + [self.eos] for b in batch]

        # Positions 0..B-1: each query's labeled positive.
        pos_strings = [b["premise"] for b in batch]
        pos_set     = set(pos_strings)
        extra       = random.sample(
            [p for p in self.library if p not in pos_set],
            min(self.cfg.n_neg * B, len(self.library) - len(pos_set)),
        )

        prem_list = pos_strings + extra   # (B + n_neg*B) premises

        # gold_mask[i, j]: True when prem_list[j] should not act as a negative for query i.
        # Two rows from the same goal share gold premises — detect via matching retrieval_query.
        gold_mask = torch.zeros(B, len(prem_list), dtype=torch.bool)
        for i in range(B):
            gold_mask[i, i] = True  # labeled positive
            for j in range(B):
                if j != i and batch[j]["retrieval_query"] == batch[i]["retrieval_query"]:
                    gold_mask[i, j] = True  # same goal → false negative

        premise_ids         = pad([self.enc(p) for p in prem_list], self.padid)
        retrieval_query_ids = pad([x + [self.emb] for x in retrieval_query], self.padid)
        draft_query_ids     = pad([x + o for x, o in zip(draft_query, draft)], self.padid)
        return dict(
            retrieval_query_ids=retrieval_query_ids,
            retrieval_query_mask=(retrieval_query_ids != self.padid).long(),
            premise_ids=premise_ids,
            premise_mask=(premise_ids != self.padid).long(),
            gold_mask=gold_mask,
            draft_query_ids=draft_query_ids,
            draft_query_mask=(draft_query_ids != self.padid).long(),
            draft_labels=pad([[-100] * len(x) + o for x, o in zip(draft_query, draft)], -100),
        )


class JointTrainer(Trainer):
    def __init__(self, *a, cfg, **k):
        super().__init__(*a, **k)
        self.cfg = cfg

    def compute_loss(self, model, x, return_outputs=False, **kw):
        q = embed(model, x["retrieval_query_ids"], x["retrieval_query_mask"])
        p = embed(model, x["premise_ids"], x["premise_mask"])
        contrastive = contrastive_loss(q, p, x["gold_mask"], self.cfg.temp)
        ce = model(input_ids=x["draft_query_ids"], attention_mask=x["draft_query_mask"],
                   labels=x["draft_labels"]).loss
        loss = ce + self.cfg.lam * contrastive
        # Print individual components periodically so we can see the loss balance
        if self.state.global_step % 50 == 0 and model.training:
            print(f"  [step {self.state.global_step}] ce={ce.item():.4f}  contrastive={contrastive.item():.4f}")
        return (loss, {"ce": ce, "contrastive": contrastive}) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return loss.detach(), None, None


def apply_chat_template_split(tok, messages):
    """Returns (query_str, draft_str) using the tokenizer's chat template."""
    query_str = tok.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full_str = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return query_str, full_str[len(query_str):]


def build_const_lookup(source: str) -> dict:
    """Build name -> full_string from a local JSON file or HF dataset id."""
    if os.path.exists(source):
        with open(source) as f:
            data = json.load(f)
        return {row["name"]: row["full_string"] for row in data}
    ds = load_dataset(source, split="train")
    return {row["name"]: row["full_string"] for row in ds}



def load_retrieval_data(premises_id: str, const_lookup: dict, n: int = None) -> HFDataset:
    """One row per (goal, premise) pair: {retrieval_query, premise}.

    Goals with no known premises are skipped. n caps the number of source goals
    (not output rows, which may be larger due to expansion).
    """
    ds = load_dataset(premises_id, split="train").shuffle(seed=42)
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
    rows = []
    for p in ds:
        prem_strs = [const_lookup[name] for name in p["premises"] if name in const_lookup]
        for prem in prem_strs:
            rows.append({"retrieval_query": p["goal"], "premise": prem})
    return HFDataset.from_list(rows)


def load_draft_data(tok, drafts_id: str, n: int = None, make_messages_fn=None) -> list:
    """Draft-only rows: [{draft_query, draft}] as a plain list for Collate sampling.

    make_messages_fn: callable(row) -> list[dict]  for datasets without a 'messages' column.
    If None, reads d['messages'] directly.
    """
    ds = load_dataset(drafts_id, split="train").shuffle(seed=123)
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
    rows = []
    for d in ds:
        msgs = make_messages_fn(d) if make_messages_fn is not None else d["messages"]
        query_str, draft_str = apply_chat_template_split(tok, msgs)
        rows.append({"draft_query": query_str, "draft": draft_str})
    return rows
