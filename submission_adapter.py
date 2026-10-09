"""Stable grading interface: thin wrappers around alignment.core."""
from __future__ import annotations

from alignment import core


def build_sft_batch(tokenizer, prompts, responses, max_length):
    return core.build_lm_batch(tokenizer, prompts, responses, max_length)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    batch = core.build_lm_batch(tokenizer, prompts, responses, max_length)
    logps, _ = core.response_logps(model, batch)
    return logps


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    batch = core.build_lm_batch(tokenizer, prompts, responses, max_length)
    return core.sft_loss(model, batch)


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    return core.build_pair_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    return core.dpo_loss_from_logps(
        policy_chosen_logps, policy_rejected_logps,
        reference_chosen_logps, reference_rejected_logps, beta,
    )


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    batch = core.build_pair_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    return core.dpo_loss(policy_model, reference_model, batch, beta)