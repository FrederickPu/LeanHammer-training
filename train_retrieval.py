"""
Premise retrieval training using HuggingFace Trainer.

String format matches hanwenzhu/lean-premise-server (app/models.py is identical):
    retrieval_query : state.to_string()   -> self.state (pretty-printed goal via Meta.ppGoal)
    premise         : premise.to_string() -> "/-- doc -/\\nkind name args : type"

Architecture follows train_joint_premise_draft.py:
    [retrieval_query][EMB] -> last-token embedding, L2-normalized
    [premise]              -> last-token embedding, L2-normalized
    loss: InfoNCE (contrastive_loss) with gold_mask false-negative masking
"""

import random, torch
from dataclasses import dataclass
from typing import List
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from data import load_data
from train_joint_premise_draft import embed, contrastive_loss, pad, EMB


@dataclass
class Cfg:
    model: str       = "Qwen/Qwen2.5-0.5B-Instruct"
    data_dir: str    = ""
    temp: float      = 0.05
    n_neg: int       = 3
    max_len: int     = 1024
    mathlib_only: bool = False


def setup(cfg):
    tok = AutoTokenizer.from_pretrained(cfg.model)
    tok.add_special_tokens({"additional_special_tokens": [EMB]})
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=torch.bfloat16)
    model.resize_token_embeddings(len(tok))
    return model, tok


class PairDataset(TorchDataset):
    """Flattens RetrievalDataset pairs into {retrieval_query, premise} dicts."""
    def __init__(self, retrieval_ds):
        self.rows = [
            {"retrieval_query": state.to_string(), "premise": premise.to_string()}
            for state, premise in retrieval_ds.pairs
        ]

    def __len__(self):  return len(self.rows)
    def __getitem__(self, i): return self.rows[i]


class Collate:
    def __init__(self, tok, cfg: Cfg, library: List[str]):
        self.tok, self.cfg, self.library = tok, cfg, library
        self.emb   = tok.convert_tokens_to_ids(EMB)
        self.padid = tok.pad_token_id

    def enc(self, text):
        return self.tok(text, add_special_tokens=False, truncation=True,
                        max_length=self.cfg.max_len).input_ids

    def __call__(self, batch):
        """batch: list of B rows {retrieval_query, premise}.

        Premise pool layout: positions 0..B-1 are each query's labeled positive;
        positions B.. are library negatives shared across the batch.

        Returns:
            retrieval_query_ids/mask  (B, Lq)   [retrieval_query][EMB]
            premise_ids/mask          (M, Lp)   flat pool
            gold_mask                 (B, M)    True = gold (labeled pos or false neg)
        """
        B = len(batch)
        retrieval_query = [self.enc(b["retrieval_query"]) for b in batch]

        pos_strings = [b["premise"] for b in batch]
        pos_set     = set(pos_strings)
        extra = random.sample(
            [p for p in self.library if p not in pos_set],
            min(self.cfg.n_neg * B, len(self.library) - len(pos_set)),
        )
        prem_list = pos_strings + extra   # positives at 0..B-1

        # gold_mask[i,j]: True when prem_list[j] should not act as a negative for query i.
        # Two rows with the same retrieval_query share the same proof state -> false negatives.
        gold_mask = torch.zeros(B, len(prem_list), dtype=torch.bool)
        for i in range(B):
            gold_mask[i, i] = True
            for j in range(B):
                if j != i and batch[j]["retrieval_query"] == batch[i]["retrieval_query"]:
                    gold_mask[i, j] = True

        retrieval_query_ids = pad([x + [self.emb] for x in retrieval_query], self.padid)
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
    dataset_train, _, _ = load_data(
        cfg.data_dir,
        mathlib_only=cfg.mathlib_only,
        num_negatives_per_state=0,   # negatives are sampled in Collate
    )
    model, tok = setup(cfg)
    library = [p.to_string() for p in dataset_train.corpus.premises]
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
        train_dataset=PairDataset(dataset_train),
        data_collator=Collate(tok, cfg, library),
        cfg=cfg,
    ).train()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model",    default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--temp",     type=float, default=0.05)
    p.add_argument("--n_neg",    type=int,   default=3)
    p.add_argument("--max_len",  type=int,   default=1024)
    p.add_argument("--mathlib_only", action="store_true")
    a = p.parse_args()
    run(Cfg(**vars(a)))
