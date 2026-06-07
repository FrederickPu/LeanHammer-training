"""
Joint training of one Qwen backbone on two objectives per batch:
draft generation (cross-entropy) and premise selection (masked InfoNCE).

The input is used in two modes:
    draft :  [input] [output] [EOS]   -> CE on the output span (generation)
    query :  [input] [EMB]            -> last-token hidden state, L2-normalized
[EMB] marks "embed here", so the same input can be generated (no [EMB]) or embedded.
Premises carry no [EMB] but are encoded the same way ([premise] -> last token); each
input is scored against them by cosine similarity and pulled toward its gold premises.
Data rows are {input, output, premises}; `input` is pre-chat-templated by the caller.
"""

import random, torch
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

EMB = "[EMB]"


@dataclass
class Cfg:
    model: str = "Qwen/Qwen3.5-TODO"
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
    h = model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
    pos = mask.sum(1) - 1                       # last real token
    return F.normalize(h[torch.arange(ids.size(0), device=ids.device), pos], dim=-1)


def contrastive_loss(q, p_all, retrieval_mask, temp):
    """Diagonal CrossEntropy with false-negative masking (matches loss.py).

    q:              (B, H)
    p_all:          (B*(1+n_neg), H)  row-major: p_all[i*(1+n_neg)] is pos for query i
    retrieval_mask: (B, B*(1+n_neg))  False where a labelled-negative is actually gold
    """
    n1 = p_all.shape[0] // len(q)          # 1 + n_neg
    scores = q @ p_all.T / temp
    scores[~retrieval_mask] = float("-inf")
    labels = torch.arange(len(q), device=q.device) * n1
    return F.cross_entropy(scores, labels)


def pad(seqs, fill):
    n = max(map(len, seqs))
    return torch.tensor([s + [fill] * (n - len(s)) for s in seqs])


class Collate:
    def __init__(self, tok, cfg, library):
        self.tok, self.cfg, self.library = tok, cfg, library
        self.emb, self.eos, self.padid = (tok.convert_tokens_to_ids(EMB),
                                          tok.eos_token_id, tok.pad_token_id)

    def enc(self, text):
        return self.tok(text, add_special_tokens=False, truncation=True,
                        max_length=self.cfg.max_len).input_ids

    def __call__(self, batch):
        """batch: list of B rows {input: str, output: str, premises: list[str]}.
        q = query (the input), p = premise, d = draft (input + output).

        Dims:
            B            rows in the batch
            Lq, Lp, Ld   padded length of the q / p / d views

        Returns tensors (q, p encoded for embedding; only d is generated):
            query_ids,   query_mask     (B, Lq)
            premise_ids, premise_mask   (B, 1+n_neg, Lp)  dim 1: [pos, neg_0, ..., neg_{n-1}]
            retrieval_mask              (B, B*(1+n_neg))   False = false negative
            draft_ids,   draft_mask     (B, Ld)   [input][output][EOS]
            draft_labels                (B, Ld)   output span only (-100 elsewhere)"""
        ins  = [self.enc(b["input"])  for b in batch]
        outs = [self.enc(b["output"]) + [self.eos] for b in batch]
        prem_sets = [set(b["premises"]) for b in batch]

        B, n_neg = len(batch), self.cfg.n_neg
        pos_all  = set().union(*prem_sets)
        neg_pool = [p for p in self.library if p not in pos_all]

        # prem_groups[i] = [pos_i, neg_0_i, ..., neg_{n_neg-1}_i]
        prem_groups = [[b["premises"][0]] + random.choices(neg_pool, k=n_neg) for b in batch]

        # Tokenize flat, then reshape to (B, 1+n_neg, L)
        all_prem_seqs = [self.enc(p) for group in prem_groups for p in group]
        premise_ids   = pad(all_prem_seqs, self.padid).view(B, 1 + n_neg, -1)

        # retrieval_mask: (B, B*(1+n_neg))
        # flat_j = qi*(1+n_neg) + c  →  qi-th query's c-th premise (c=0 is its positive)
        mask = torch.ones(B, B * (1 + n_neg), dtype=torch.bool)
        for i in range(B):
            for qi in range(B):
                for c, prem in enumerate(prem_groups[qi]):
                    if qi == i and c == 0:              # labeled positive for query i
                        continue
                    if prem in prem_sets[i]:            # false negative for query i
                        mask[i, qi * (1 + n_neg) + c] = False

        query_ids = pad([x + [self.emb] for x in ins], self.padid)
        draft_ids = pad([x + o for x, o in zip(ins, outs)], self.padid)
        return dict(
            # input
            query_ids=query_ids,     query_mask=(query_ids != self.padid).long(),
            # premise
            premise_ids=premise_ids, premise_mask=(premise_ids != self.padid).long(),
            retrieval_mask=mask,
            # draft
            draft_ids=draft_ids,     draft_mask=(draft_ids != self.padid).long(),
            draft_labels=pad([[-100] * len(x) + o for x, o in zip(ins, outs)], -100),
        )


class JointTrainer(Trainer):
    def __init__(self, *a, cfg, **k):
        super().__init__(*a, **k)
        self.cfg = cfg

    def compute_loss(self, model, x, return_outputs=False, **kw):
        B, n1, _ = x["premise_ids"].shape
        q = embed(model, x["query_ids"], x["query_mask"])
        p = embed(model, x["premise_ids"].view(B * n1, -1), x["premise_mask"].view(B * n1, -1))
        contrastive = contrastive_loss(q, p, x["retrieval_mask"], self.cfg.temp)
        ce = model(input_ids=x["draft_ids"], attention_mask=x["draft_mask"],
                   labels=x["draft_labels"]).loss
        loss = ce + self.cfg.lam * contrastive
        return (loss, {"ce": ce, "contrastive": contrastive}) if return_outputs else loss


def run(cfg, dataset, library):
    model, tok = setup(cfg)
    args = TrainingArguments(output_dir="out", per_device_train_batch_size=16,
                             learning_rate=1e-5, num_train_epochs=1, bf16=True,
                             gradient_checkpointing=True, logging_steps=20,
                             remove_unused_columns=False)
    JointTrainer(model=model, args=args, train_dataset=dataset,
                 data_collator=Collate(tok, cfg, library), cfg=cfg).train()
