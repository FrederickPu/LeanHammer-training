"""
Premise retrieval training using HuggingFace Trainer.

String format matches hanwenzhu/lean-premise-server exactly:
    retrieval_query : raw Lean proof state (pretty-printed goal via Meta.ppGoal)
    premise         : full_string from consts dataset ("/-- doc -/\\nkind name args : type")

Encoding matches the server: plain tokenization with add_eos_token=True, pool at last token.
No chat template, no special [EMB] marker.

Loss: InfoNCE (contrastive_loss) with gold_mask false-negative masking.

Usage:
    python train_retrieval.py \\
        --retrieval_id UnluckyOrangutan/premises-mathlib-v4.30.0 \\
        --consts_id    UnluckyOrangutan/consts-mathlib-v4.30.0 \\
        --model        Qwen/Qwen2.5-0.5B-Instruct

    Or from Python:
        from train_retrieval import Cfg, run
        run(Cfg(retrieval_id="UnluckyOrangutan/premises-mathlib-v4.30.0"))

    consts_id / retrieval_id can each be a HuggingFace dataset id or a local JSON
    file with the same schema ({name, full_string} and {goal, premises} respectively).
"""

import json, os, random, torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List
from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


@dataclass
class Cfg:
    model: str        = "Qwen/Qwen2.5-0.5B-Instruct"
    retrieval_id: str = "UnluckyOrangutan/premises-mathlib-v4.30.0"
    consts_id: str    = "UnluckyOrangutan/consts-mathlib-v4.30.0"
    temp: float       = 0.05
    n_neg: int        = 3
    max_len: int      = 1024


def setup(cfg):
    tok = AutoTokenizer.from_pretrained(cfg.model, add_eos_token=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=torch.bfloat16)
    return model, tok


def embed(model, ids, mask):
    base = getattr(model, "model", None) or model.transformer
    h = base(input_ids=ids, attention_mask=mask).last_hidden_state
    pos = mask.sum(1) - 1
    return F.normalize(h[torch.arange(ids.size(0), device=ids.device), pos], dim=-1)


def contrastive_loss(q, p_all, gold_mask, temp):
    """InfoNCE with false-negative masking.

    q:         (B, H)
    p_all:     (M, H)  positions 0..B-1 are labeled positives
    gold_mask: (B, M)  True = gold for that query (labeled pos or false neg)
    """
    B = len(q)
    scores  = q @ p_all.T / temp
    fn_mask = gold_mask.clone()
    fn_mask[torch.arange(B), torch.arange(B)] = False   # labeled positives are not false negatives
    scores[fn_mask] = float("-inf")
    return F.cross_entropy(scores, torch.arange(B, device=q.device))


def pad(seqs, fill):
    n = max(map(len, seqs))
    return torch.tensor([s + [fill] * (n - len(s)) for s in seqs])


def build_const_lookup(source: str) -> dict:
    """name -> full_string from a HF dataset id or local JSON file."""
    if os.path.exists(source):
        with open(source) as f:
            data = json.load(f)
        return {row["name"]: row["full_string"] for row in data}
    ds = load_dataset(source, split="train")
    return {row["name"]: row["full_string"] for row in ds}


def load_retrieval_data(source: str, const_lookup: dict, n: int = None) -> HFDataset:
    """One row per (goal, premise) pair: {retrieval_query, premise}."""
    ds = load_dataset(source, split="train").shuffle(seed=42)
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
    rows = []
    for p in ds:
        for name in p["premises"]:
            if name in const_lookup:
                rows.append({"retrieval_query": p["goal"], "premise": const_lookup[name]})
    return HFDataset.from_list(rows)


class Collate:
    def __init__(self, tok, cfg: Cfg, library: List[str]):
        self.tok, self.cfg, self.library = tok, cfg, library
        self.padid = tok.pad_token_id

    def enc(self, text):
        return self.tok(text, add_special_tokens=True, truncation=True,
                        max_length=self.cfg.max_len).input_ids

    def __call__(self, batch):
        """batch: list of B rows {retrieval_query, premise}.

        Premise pool: positions 0..B-1 are labeled positives; positions B.. are negatives.

        Returns:
            retrieval_query_ids/mask  (B, Lq)
            premise_ids/mask          (M, Lp)
            gold_mask                 (B, M)   True = gold (labeled pos or false neg)
        """
        B = len(batch)
        retrieval_query = [self.enc(b["retrieval_query"]) for b in batch]

        pos_strings = [b["premise"] for b in batch]
        pos_set     = set(pos_strings)
        extra = random.sample(
            [p for p in self.library if p not in pos_set],
            min(self.cfg.n_neg * B, len(self.library) - len(pos_set)),
        )
        prem_list = pos_strings + extra

        gold_mask = torch.zeros(B, len(prem_list), dtype=torch.bool)
        for i in range(B):
            gold_mask[i, i] = True
            for j in range(B):
                if j != i and batch[j]["retrieval_query"] == batch[i]["retrieval_query"]:
                    gold_mask[i, j] = True

        retrieval_query_ids = pad(retrieval_query, self.padid)
        premise_ids         = pad([self.enc(p) for p in prem_list], self.padid)
        return dict(
            retrieval_query_ids=retrieval_query_ids,
            retrieval_query_mask=(retrieval_query_ids != self.padid).long(),
            premise_ids=premise_ids,
            premise_mask=(premise_ids != self.padid).long(),
            gold_mask=gold_mask,
        )


class RetrievalTrainer(Trainer):
    def __init__(self, *a, cfg: Cfg, **k):
        super().__init__(*a, **k)
        self.cfg = cfg

    def compute_loss(self, model, x, return_outputs=False, **kw):
        q = embed(model, x["retrieval_query_ids"], x["retrieval_query_mask"])
        p = embed(model, x["premise_ids"],         x["premise_mask"])
        loss = contrastive_loss(q, p, x["gold_mask"], self.cfg.temp)
        if self.state.global_step % 50 == 0 and model.training:
            print(f"  [step {self.state.global_step}] contrastive={loss.item():.4f}")
        return (loss, {}) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return loss.detach(), None, None


def run(cfg: Cfg):
    const_lookup = build_const_lookup(cfg.consts_id)
    library      = list(const_lookup.values())
    train_ds     = load_retrieval_data(cfg.retrieval_id, const_lookup)
    model, tok   = setup(cfg)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=16,
        learning_rate=1e-5,
        num_train_epochs=1,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=20,
        remove_unused_columns=False,
    )
    RetrievalTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=Collate(tok, cfg, library),
        cfg=cfg,
    ).train()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model",        default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--retrieval_id", default="UnluckyOrangutan/premises-mathlib-v4.30.0")
    p.add_argument("--consts_id",    default="UnluckyOrangutan/consts-mathlib-v4.30.0")
    p.add_argument("--temp",         type=float, default=0.05)
    p.add_argument("--n_neg",        type=int,   default=3)
    p.add_argument("--max_len",      type=int,   default=1024)
    a = p.parse_args()
    run(Cfg(**vars(a)))
