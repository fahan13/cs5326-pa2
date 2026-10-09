"""Core SFT/DPO math: batching, response-only log-probs, and losses."""
from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100  # cross_entropy skips this label value


def _pad_id(tokenizer) -> int:
    # Some tokenizers have no pad token; fall back to EOS, then 0.
    # Padding positions are masked anyway, so the ID only needs to be valid.
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", None)
    return 0 if pad is None else pad


def build_lm_batch(tokenizer, prompts, responses, max_length):
    """prompt + response + EOS -> right-padded tensors with response-only labels."""
    eos = getattr(tokenizer, "eos_token_id", None)
    pad = _pad_id(tokenizer)

    rows = []
    for prompt, response in zip(prompts, responses, strict=True):
        # Tokenize separately so the prompt/response boundary is exact (Idea 1).
        p_ids = tokenizer.encode(prompt, add_special_tokens=False)
        r_ids = tokenizer.encode(response, add_special_tokens=False)
        if eos is not None:
            r_ids = r_ids + [eos]                      # EOS appended BEFORE truncation
        ids = (p_ids + r_ids)[:max_length]
        labels = ([IGNORE_INDEX] * len(p_ids) + r_ids)[:max_length]
        rows.append((ids, labels))

    # Pad to the longest row in this batch (not to max_length): less wasted compute.
    width = max(len(ids) for ids, _ in rows)
    input_ids, attention_mask, label_rows = [], [], []
    for ids, labels in rows:
        n_pad = width - len(ids)
        input_ids.append(ids + [pad] * n_pad)
        attention_mask.append([1] * len(ids) + [0] * n_pad)   # by length, not by pad ID (Idea 2)
        label_rows.append(labels + [IGNORE_INDEX] * n_pad)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(label_rows, dtype=torch.long),
    }


def build_pair_batch(tokenizer, prompts, chosen, rejected, max_length):
    """DPO batch: the same prompts paired with chosen and with rejected responses."""
    return {
        "chosen": build_lm_batch(tokenizer, prompts, chosen, max_length),
        "rejected": build_lm_batch(tokenizer, prompts, rejected, max_length),
    }


def response_logps(model, batch):
    """Per-row summed log-prob of response tokens, plus the response token count."""
    device = next(model.parameters()).device          # policy and reference may sit on different GPUs
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits = logits[:, :-1, :].float()                # position t predicts t+1 (Idea 3); fp32 for stability
    targets = labels[:, 1:]

    # cross_entropy = -log p(target). reduction="none" keeps one value per token,
    # and ignored (-100) positions come back as exactly 0.
    nll = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view(targets.shape)

    logps = -nll.sum(dim=-1)                          # one number per row, row order preserved
    counts = targets.ne(IGNORE_INDEX).sum(dim=-1)
    return logps, counts


def sft_loss(model, batch):
    """Token-mean reduction: total response NLL / total response tokens in the batch."""
    logps, counts = response_logps(model, batch)
    return -logps.sum() / counts.sum().clamp_min(1)


def dpo_loss_from_logps(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta):
    """Batch-mean DPO loss: -log sigmoid(beta * (policy gap - reference gap))."""
    ref_chosen = ref_chosen.to(policy_chosen.device)
    ref_rejected = ref_rejected.to(policy_chosen.device)
    policy_gap = policy_chosen - policy_rejected
    ref_gap = ref_chosen - ref_rejected
    return -F.logsigmoid(beta * (policy_gap - ref_gap)).mean()


def dpo_loss(policy, reference, pair_batch, beta):
    policy_chosen, _ = response_logps(policy, pair_batch["chosen"])
    policy_rejected, _ = response_logps(policy, pair_batch["rejected"])
    with torch.no_grad():                             # reference is frozen: no graph, no gradients
        ref_chosen, _ = response_logps(reference, pair_batch["chosen"])
        ref_rejected, _ = response_logps(reference, pair_batch["rejected"])
    return dpo_loss_from_logps(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)