"""Our own edge-case tests, beyond the public smoke tests.

Covers what the manual says the private grader checks: truncation boundaries,
unusual tokenizer configurations, mixed sequence lengths, row ordering,
exact numerics, and small optimization steps.

Run:  python -m pytest -q tests/
"""
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

import submission_adapter as adapter
from alignment import core
from alignment.prompts import format_prompt
from tests.toy_lm import ToyCausalLM, ToyTokenizer

IGNORE = core.IGNORE_INDEX


# ---------- unusual tokenizer configurations ----------

# Overrides happen after __init__ so the toy vocabulary itself is built normally.

class PadIsEosTokenizer(ToyTokenizer):
    """Like Qwen2.5 base: pad and EOS are the same token."""
    def __init__(self):
        super().__init__()
        self.pad_token, self.pad_token_id = self.eos_token, self.eos_token_id


class NoPadTokenizer(ToyTokenizer):
    def __init__(self):
        super().__init__()
        self.pad_token_id = None   # text form kept only so the toy encoder still works


class NoEosTokenizer(ToyTokenizer):
    def __init__(self):
        super().__init__()
        self.eos_token_id = None


def n_tokens(tok, text):
    return len(tok.encode(text, add_special_tokens=False))


def manual_logp(model, tok, response, add_eos=True):
    """Toy model logits ignore position, so log p(response) = sum of log_softmax(bias)[token]."""
    log_probs = F.log_softmax(model.logit_bias.detach(), dim=-1)
    ids = tok.encode(response, add_special_tokens=False) + ([tok.eos_token_id] if add_eos else [])
    return log_probs[ids].sum()


# ---------- batching ----------

def test_pad_equals_eos_keeps_real_eos_in_labels():
    tok = PadIsEosTokenizer()
    prompts, responses = ["Q: ", "Task: "], ["hello", "x"]
    batch = adapter.build_sft_batch(tok, prompts, responses, max_length=32)
    for row, (p, r) in enumerate(zip(prompts, responses)):
        end = n_tokens(tok, p) + n_tokens(tok, r) + 1          # +1 for EOS
        assert batch["labels"][row, end - 1] == tok.eos_token_id   # real EOS is learned
        assert batch["attention_mask"][row, end - 1] == 1          # ...and attended to
        assert batch["labels"][row, end:].eq(IGNORE).all()         # padding is masked
        assert batch["attention_mask"][row, end:].eq(0).all()
    # Row 1 is shorter, so it must contain padding that looks identical to EOS.
    assert batch["attention_mask"][1].eq(0).any()


def test_truncation_can_remove_eos():
    tok = ToyTokenizer()
    batch = adapter.build_sft_batch(tok, ["Q: "], ["abcdef"], max_length=7)
    labels = batch["labels"][0]
    assert batch["input_ids"].shape == (1, 7)
    assert labels[:3].eq(IGNORE).all()
    assert labels[3:].tolist() == tok.encode("abcd")             # cut mid-response
    assert tok.eos_token_id not in labels.tolist()               # EOS appended, then truncated away


def test_eos_survives_when_it_exactly_fits():
    tok = ToyTokenizer()
    batch = adapter.build_sft_batch(tok, ["Q: "], ["ab"], max_length=6)   # 3 + 2 + EOS = 6
    assert batch["labels"][0, -1] == tok.eos_token_id


def test_prompt_longer_than_max_length_gives_no_response_tokens():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.5)
    batch = adapter.build_sft_batch(tok, ["abcdefghij"], ["ok"], max_length=5)
    assert batch["labels"].eq(IGNORE).all()
    logps = adapter.compute_response_logprobs(model, tok, ["abcdefghij"], ["ok"], max_length=5)
    assert logps.item() == 0.0
    loss = adapter.compute_sft_loss(model, tok, ["abcdefghij"], ["ok"], max_length=5)
    assert torch.isfinite(loss)                                   # no divide-by-zero NaN


def test_missing_pad_token_falls_back_without_error():
    tok = NoPadTokenizer()
    batch = adapter.build_sft_batch(tok, ["Q: ", "Task: "], ["a", "bb"], max_length=16)
    assert batch["attention_mask"][0].eq(0).any()
    assert batch["labels"][batch["attention_mask"].eq(0)].eq(IGNORE).all()


def test_missing_eos_appends_nothing():
    tok = NoEosTokenizer()
    batch = adapter.build_sft_batch(tok, ["Q: "], ["abc"], max_length=16)
    assert batch["labels"][0].ne(IGNORE).sum() == 3


def test_dpo_batch_masks_the_same_prompt_on_both_sides():
    tok = ToyTokenizer()
    batch = adapter.build_dpo_batch(tok, ["Task: ", "Q: "], ["good", "y"], ["bad answer", "no"], 32)
    for row, p in enumerate(["Task: ", "Q: "]):
        k = n_tokens(tok, p)
        for side in ("chosen", "rejected"):
            assert batch[side]["labels"][row, :k].eq(IGNORE).all()
            assert batch[side]["labels"][row, k] != IGNORE


# ---------- log-probabilities ----------

def test_logps_match_manual_computation():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.7)
    responses = ["abc", "hello world"]
    logps = adapter.compute_response_logprobs(model, tok, ["Q: ", "Prompt: "], responses, 64)
    expected = torch.stack([manual_logp(model, tok, r) for r in responses])
    torch.testing.assert_close(logps, expected)


def test_row_order_is_preserved():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.7)
    prompts, responses = ["A: ", "B: ", "C: "], ["x", "longer text", "mid"]
    forward = adapter.compute_response_logprobs(model, tok, prompts, responses, 64)
    backward = adapter.compute_response_logprobs(model, tok, prompts[::-1], responses[::-1], 64)
    torch.testing.assert_close(forward, backward.flip(0))


def test_padding_does_not_change_a_rows_score():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.7)
    alone = adapter.compute_response_logprobs(model, tok, ["Q: "], ["hi"], 64)
    batched = adapter.compute_response_logprobs(
        model, tok, ["Q: ", "A much longer prompt: "], ["hi", "a much longer response"], 64
    )
    torch.testing.assert_close(alone[0], batched[0])


# ---------- losses ----------

def test_sft_loss_is_token_mean():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.3)
    responses = ["a", "longer"]
    loss = adapter.compute_sft_loss(model, tok, ["Do: ", "Say: "], responses, 64)
    total = -sum(manual_logp(model, tok, r) for r in responses)
    count = sum(n_tokens(tok, r) + 1 for r in responses)
    torch.testing.assert_close(loss, total / count)


def test_dpo_loss_matches_formula():
    torch.manual_seed(0)
    pc, pr, rc, rr = (torch.randn(8) * 5 for _ in range(4))
    beta = 0.1
    loss = adapter.compute_dpo_loss_from_logps(pc, pr, rc, rr, beta)
    z = beta * ((pc - pr) - (rc - rr))
    torch.testing.assert_close(loss, -torch.log(torch.sigmoid(z)).mean())


def test_dpo_loss_is_stable_for_extreme_values():
    pc = torch.tensor([-5000.0, 5000.0], requires_grad=True)
    zero = torch.zeros(2)
    loss = adapter.compute_dpo_loss_from_logps(pc, zero, zero, zero, beta=0.1)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pc.grad).all()


def test_dpo_loss_is_log2_when_policy_equals_reference():
    tok = ToyTokenizer()
    policy = ToyCausalLM(tok.vocab_size, bias_scale=0.5)
    reference = copy.deepcopy(policy)
    loss = adapter.compute_dpo_loss(policy, reference, tok, ["P: ", "Q: "], ["good", "yes"], ["bad", "no"], 0.1, 32)
    torch.testing.assert_close(loss, torch.log(torch.tensor(2.0)))


def test_one_dpo_step_widens_the_preference_gap():
    tok = ToyTokenizer()
    policy = ToyCausalLM(tok.vocab_size, bias_scale=0.5)
    reference = copy.deepcopy(policy)
    args = (tok, ["P: ", "Q: "], ["good", "yes"], ["bad", "no"])
    opt = torch.optim.SGD(policy.parameters(), lr=1.0)
    reference_snapshot = reference.logit_bias.detach().clone()

    before = adapter.compute_dpo_loss(policy, reference, *args, beta=0.1, max_length=32)
    before.backward()
    opt.step()
    after = adapter.compute_dpo_loss(policy, reference, *args, beta=0.1, max_length=32)

    assert after < before
    assert torch.equal(reference.logit_bias, reference_snapshot)   # reference never moved
    assert reference.logit_bias.grad is None


def test_one_sft_step_lowers_sft_loss():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size)
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    args = (tok, ["Q: "], ["hello"], 32)
    before = adapter.compute_sft_loss(model, *args)
    before.backward()
    opt.step()
    assert adapter.compute_sft_loss(model, *args) < before


# ---------- real Qwen tokenizer (downloads ~10 MB once; skipped offline) ----------

@pytest.fixture(scope="module")
def qwen_tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    except Exception as exc:  # no internet, HF down, etc.
        pytest.skip(f"Qwen tokenizer unavailable: {exc}")


def test_qwen_eos_is_learned_even_though_pad_equals_eos(qwen_tok):
    prompts = [format_prompt("Name a colour."), format_prompt("Write a long sentence about the sea.")]
    responses = ["Blue.", "The sea is vast, restless, and full of life that we barely understand."]
    batch = adapter.build_sft_batch(qwen_tok, prompts, responses, max_length=128)
    for row, (p, r) in enumerate(zip(prompts, responses)):
        end = n_tokens(qwen_tok, p) + n_tokens(qwen_tok, r) + 1
        assert batch["labels"][row, end - 1] == qwen_tok.eos_token_id
        assert batch["labels"][row, end:].eq(IGNORE).all()


def test_qwen_prompt_response_boundary_round_trips(qwen_tok):
    prompt, response = format_prompt("Say hi."), "Hi there!"
    p_ids = qwen_tok.encode(prompt, add_special_tokens=False)
    r_ids = qwen_tok.encode(response, add_special_tokens=False)
    assert qwen_tok.decode(p_ids + r_ids) == prompt + response
    # Qwen adds no BOS token, so nothing hidden sneaks in front of the prompt.
    assert qwen_tok.encode("", add_special_tokens=True) == []
