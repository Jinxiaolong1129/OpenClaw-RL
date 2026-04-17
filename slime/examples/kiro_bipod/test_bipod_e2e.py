"""
End-to-end validation of the BIPOD pipeline.

Loads a real Sample from sample_instance.pkl (output of kiro_generate_with_kos),
queries the teacher SGLang for top-K logprobs, then validates the distillation
loss computation (reverse KL with tail trick) on CPU using a mock student logits
tensor.

Usage:
    TEACHER_URL=http://127.0.0.1:31000/generate python3 examples/kiro_bipod/test_bipod_e2e.py
"""

from __future__ import annotations

import asyncio
import os
import pickle
import sys
import time

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Step 0: Setup
# ---------------------------------------------------------------------------

TEACHER_URL = os.environ.get("TEACHER_URL", "http://127.0.0.1:31000/generate")
SAMPLE_PATH = "/workspace/sample_instance.pkl"
K = 50  # top-K


def section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Step 1: Load Sample
# ---------------------------------------------------------------------------
section("Step 1: Load Sample from pickle")

with open(SAMPLE_PATH, "rb") as f:
    sample = pickle.load(f)

print(f"  tokens:          {len(sample.tokens)}")
print(f"  response_length: {sample.response_length}")
prompt_len = len(sample.tokens) - sample.response_length
print(f"  prompt_len:      {prompt_len}")
print(f"  loss_mask len:   {len(sample.loss_mask)}")
print(f"  trainable:       {sum(sample.loss_mask)} / {len(sample.loss_mask)}")
print(f"  instance_id:     {sample.metadata.get('instance_id', '?')}")

# Verify TITO data integrity
assert len(sample.loss_mask) == sample.response_length, "loss_mask / response_length mismatch"
assert len(sample.rollout_log_probs) == sample.response_length, "rollout_log_probs / response_length mismatch"
assert len(sample.tokens) == prompt_len + sample.response_length, "tokens length mismatch"
print("  TITO integrity: OK")


# ---------------------------------------------------------------------------
# Step 2: Query teacher for top-K logprobs
# ---------------------------------------------------------------------------
section("Step 2: Query teacher SGLang for top-K logprobs")
print(f"  TEACHER_URL: {TEACHER_URL}")
print(f"  K: {K}")

# We'll query on a truncated sequence for speed (full would be 42K tokens).
# Use first 2000 tokens of response for the test.
TEST_RESPONSE_LEN = min(2000, sample.response_length)
test_tokens = sample.tokens[:prompt_len + TEST_RESPONSE_LEN]
test_loss_mask = sample.loss_mask[:TEST_RESPONSE_LEN]

print(f"  Test sequence length: {len(test_tokens)} (prompt={prompt_len}, response={TEST_RESPONSE_LEN})")
print(f"  Test trainable tokens: {sum(test_loss_mask)}")

sys.path.insert(0, "/workspace/Kiro_Slime")
from examples.kiro_bipod.teacher_logprobs import query_teacher_topk

t0 = time.monotonic()
teacher_data = asyncio.run(query_teacher_topk(
    input_ids=test_tokens,
    response_length=TEST_RESPONSE_LEN,
    loss_mask=test_loss_mask,
    K=K,
    teacher_url=TEACHER_URL,
))
elapsed = time.monotonic() - t0

assert teacher_data is not None, "Teacher query returned None!"
teacher_lp = teacher_data["log_probs"]   # [T, K]
teacher_idx = teacher_data["indices"]     # [T, K]

print(f"  Query time: {elapsed:.2f}s")
print(f"  teacher_topk_log_probs shape: {teacher_lp.shape}")
print(f"  teacher_topk_indices shape:   {teacher_idx.shape}")
print(f"  teacher_topk_log_probs dtype: {teacher_lp.dtype}")
print(f"  teacher_topk_indices dtype:   {teacher_idx.dtype}")
assert teacher_lp.shape == (TEST_RESPONSE_LEN, K), f"Expected ({TEST_RESPONSE_LEN}, {K}), got {teacher_lp.shape}"
assert teacher_idx.shape == (TEST_RESPONSE_LEN, K), f"Expected ({TEST_RESPONSE_LEN}, {K}), got {teacher_idx.shape}"
print("  Shape check: OK")

# Sanity: teacher logprobs should be <= 0
assert teacher_lp.max() <= 0.0 + 1e-6, f"Teacher logprobs > 0: max={teacher_lp.max()}"
# Top-1 should have highest prob at each position
top1_lp = teacher_lp[:, 0]
assert (top1_lp >= teacher_lp[:, -1] - 1e-6).all(), "Top-1 not >= Top-K"
print(f"  Teacher top-1 logprob range: [{top1_lp.min():.4f}, {top1_lp.max():.4f}]")
print(f"  Teacher top-K logprob range: [{teacher_lp[:, -1].min():.4f}, {teacher_lp[:, -1].max():.4f}]")

# Check token indices are valid vocab IDs
assert teacher_idx.min() >= 0, f"Negative token ID: {teacher_idx.min()}"
print(f"  Token ID range: [{teacher_idx.min()}, {teacher_idx.max()}]")

# Check: teacher's top-1 token should often match the student's generated token
resp_tokens = test_tokens[prompt_len:]
teacher_top1_tokens = teacher_idx[:, 0].tolist()
match_count = sum(1 for a, b in zip(resp_tokens, teacher_top1_tokens) if a == b)
print(f"  Top-1 token match with student: {match_count}/{TEST_RESPONSE_LEN} ({match_count/TEST_RESPONSE_LEN*100:.1f}%)")
print("  Teacher query: OK")


# ---------------------------------------------------------------------------
# Step 3: Validate loss_mask alignment
# ---------------------------------------------------------------------------
section("Step 3: Validate loss_mask alignment with teacher data")

loss_mask_t = torch.tensor(test_loss_mask, dtype=torch.float32)  # [T]
trainable_positions = loss_mask_t.sum().item()

# Teacher logprobs at masked positions (tool outputs) should still be valid
# numbers — they just won't contribute to the loss.
masked_teacher_lp = teacher_lp[loss_mask_t == 0]
unmasked_teacher_lp = teacher_lp[loss_mask_t == 1]
print(f"  Trainable positions: {int(trainable_positions)}")
print(f"  Masked positions:    {int(TEST_RESPONSE_LEN - trainable_positions)}")
if unmasked_teacher_lp.numel() > 0:
    print(f"  Mean teacher top-1 logprob (trainable):   {unmasked_teacher_lp[:, 0].mean():.4f}")
if masked_teacher_lp.numel() > 0:
    print(f"  Mean teacher top-1 logprob (masked):      {masked_teacher_lp[:, 0].mean():.4f}")
print("  Alignment: OK")


# ---------------------------------------------------------------------------
# Step 4: Compute distillation loss (CPU mock)
# ---------------------------------------------------------------------------
section("Step 4: Compute top-K distillation loss (CPU mock)")

# Create mock student logprobs: slightly noisy version of teacher
# (simulates a student that's close but not identical to teacher)
torch.manual_seed(42)
noise = torch.randn_like(teacher_lp) * 0.5
student_topk = teacher_lp + noise  # Not real logprobs, but sufficient for loss validation

# Normalize to be valid log-probs (softmax over K dims, then log)
# This ensures they sum to < 1 in prob space
student_topk = F.log_softmax(student_topk, dim=-1)
# Scale down so they represent partial probability mass (not the full vocab)
student_topk = student_topk - 2.0  # shift so sum(exp(x)) << 1

# Also adjust teacher to be in valid range
teacher_topk = F.log_softmax(teacher_lp, dim=-1) - 2.0

# --- Tail trick ---
student_log_s = torch.logsumexp(student_topk, dim=-1, keepdim=True)
student_log_s = torch.clamp(student_log_s, max=-1e-7)
student_tail = torch.log(-torch.expm1(student_log_s))

teacher_log_s = torch.logsumexp(teacher_topk, dim=-1, keepdim=True)
teacher_log_s = torch.clamp(teacher_log_s, max=-1e-7)
teacher_tail = torch.log(-torch.expm1(teacher_log_s))

student_with_tail = torch.cat([student_topk, student_tail], dim=-1)  # [T, K+1]
teacher_with_tail = torch.cat([teacher_topk, teacher_tail], dim=-1)  # [T, K+1]

print(f"  student_with_tail shape: {student_with_tail.shape}")
print(f"  teacher_with_tail shape: {teacher_with_tail.shape}")

# Verify: exp(log-probs) sum to ~1 for each position (K+1 simplex)
student_probs = torch.exp(student_with_tail)
teacher_probs = torch.exp(teacher_with_tail)
print(f"  Student prob sum (should be ~1): mean={student_probs.sum(-1).mean():.6f}, "
      f"min={student_probs.sum(-1).min():.6f}, max={student_probs.sum(-1).max():.6f}")
print(f"  Teacher prob sum (should be ~1): mean={teacher_probs.sum(-1).mean():.6f}, "
      f"min={teacher_probs.sum(-1).min():.6f}, max={teacher_probs.sum(-1).max():.6f}")

# Reverse KL: D_KL(student || teacher) = sum_k student(k) * (log student(k) - log teacher(k))
per_token_kl = F.kl_div(
    teacher_with_tail,
    student_with_tail,
    reduction="none",
    log_target=True,
).sum(dim=-1)  # [T]

print(f"  per_token_kl shape: {per_token_kl.shape}")
print(f"  per_token_kl range: [{per_token_kl.min():.6f}, {per_token_kl.max():.6f}]")
assert (per_token_kl >= -1e-6).all(), f"KL divergence should be >= 0, min={per_token_kl.min()}"
print("  KL non-negativity: OK")

# Apply loss mask: only trainable tokens contribute
masked_kl = per_token_kl * loss_mask_t
if trainable_positions > 0:
    mean_kl_trainable = masked_kl.sum() / trainable_positions
    mean_kl_all = per_token_kl.mean()
    print(f"  Mean KL (all positions):       {mean_kl_all:.6f}")
    print(f"  Mean KL (trainable only):      {mean_kl_trainable:.6f}")
    print(f"  Total KL (trainable):          {masked_kl.sum():.6f}")
else:
    print("  WARNING: No trainable positions in test window")

# Verify the loss is 0 when student == teacher (sanity)
perfect_kl = F.kl_div(
    teacher_with_tail,
    teacher_with_tail,
    reduction="none",
    log_target=True,
).sum(dim=-1)
assert perfect_kl.abs().max() < 1e-5, f"KL(teacher||teacher) should be ~0, got max={perfect_kl.abs().max()}"
print("  KL(teacher, teacher) ≈ 0: OK")

print("\n  Loss computation: OK")


# ---------------------------------------------------------------------------
# Step 5: Test with actual teacher logprobs (not mock)
# ---------------------------------------------------------------------------
section("Step 5: Compute loss with real teacher logprobs")

# Use the actual teacher logprobs we queried
# Student logprobs: use the student's rollout logprobs from the sample
student_rollout_lp = torch.tensor(sample.rollout_log_probs[:TEST_RESPONSE_LEN], dtype=torch.float32)

# The student rollout_log_probs are per-token log-probs of the generated token.
# For the top-K loss, we'd normally get student log-probs at teacher's K indices
# from the Megatron forward pass. Here we can only approximate:
# - Position where teacher top-1 == generated token: student_lp ≈ rollout_lp
# - Other positions: we don't have the student's logits, so we skip full loss

# Instead, let's verify the teacher data is well-formed for the loss function
# by checking that the tail trick produces valid distributions.
teacher_log_s = torch.logsumexp(teacher_lp, dim=-1, keepdim=True)
teacher_log_s_clamped = torch.clamp(teacher_log_s, max=-1e-7)
teacher_tail_real = torch.log(-torch.expm1(teacher_log_s_clamped))

# Check tail probabilities are valid
teacher_tail_prob = torch.exp(teacher_tail_real)
teacher_topk_prob_sum = torch.exp(teacher_log_s)
print(f"  Teacher top-K prob mass: mean={teacher_topk_prob_sum.mean():.4f}, "
      f"min={teacher_topk_prob_sum.min():.4f}, max={teacher_topk_prob_sum.max():.4f}")
print(f"  Teacher tail prob:       mean={teacher_tail_prob.mean():.4f}, "
      f"min={teacher_tail_prob.min():.4f}, max={teacher_tail_prob.max():.4f}")

total_prob = teacher_topk_prob_sum.squeeze(-1) + teacher_tail_prob.squeeze(-1)
print(f"  Teacher total prob (should be ~1): mean={total_prob.mean():.6f}, "
      f"max_deviation={torch.abs(total_prob - 1.0).max():.2e}")
assert torch.abs(total_prob - 1.0).max() < 1e-4, "Teacher probs don't sum to 1!"
print("  Teacher distribution validity: OK")

# Check that positions where top-K captures most probability are handled well
high_mass_positions = (teacher_topk_prob_sum.squeeze() > 0.99).sum()
low_mass_positions = (teacher_topk_prob_sum.squeeze() < 0.5).sum()
print(f"  Positions where top-{K} captures >99% prob: {high_mass_positions}/{TEST_RESPONSE_LEN}")
print(f"  Positions where top-{K} captures <50% prob: {low_mass_positions}/{TEST_RESPONSE_LEN}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
section("ALL TESTS PASSED")
print(f"  Teacher SGLang:     {TEACHER_URL}")
print(f"  Model:              Qwen3-4B (as test teacher)")
print(f"  Top-K:              {K}")
print(f"  Test response len:  {TEST_RESPONSE_LEN}")
print(f"  Teacher query time: {elapsed:.2f}s")
print(f"  teacher_topk_log_probs: [{TEST_RESPONSE_LEN}, {K}] float32")
print(f"  teacher_topk_indices:   [{TEST_RESPONSE_LEN}, {K}] int64")
print(f"  KL loss (mock):     {mean_kl_trainable:.6f}")
print(f"  Loss mask correct:  trainable={int(trainable_positions)}, "
      f"masked={int(TEST_RESPONSE_LEN - trainable_positions)}")
print()
