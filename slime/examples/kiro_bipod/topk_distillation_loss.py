"""Top-K logits-based distillation loss with tail trick.

Computes reverse KL divergence D_KL(student || teacher) over the teacher's
top-K vocabulary tokens plus a "tail" bin capturing remaining probability
mass, following the SDFT/SDPO approach.

The loss is applied only to positions where loss_mask == 1 (model-generated
tokens in the multi-turn trajectory).  Tool outputs and injected prompts
within the response are masked out.

Usage:
    --loss-type custom_loss
    --custom-loss-function-path examples.kiro_bipod.topk_distillation_loss.topk_distillation_loss_function
    --distill-topk 50
    --disable-compute-advantages-and-returns

Reference: SDFT (arXiv 2601.19897), SDPO (arXiv 2601.20802)
"""

from __future__ import annotations

from argparse import Namespace
from typing import Callable

import torch
import torch.nn.functional as F

from megatron.core import mpu

from slime.backends.megatron_utils.loss import get_responses
from slime.utils.ppo_utils import compute_log_probs


def _compute_one_k(logits, indices_k, tp_group):
    """Compute log-prob for one column of top-K indices.

    Clones logits because ``fused_vocab_parallel_cross_entropy`` modifies
    them in-place.  Wrapped by ``torch.utils.checkpoint`` in the caller
    so that only one clone + intermediate set exists at a time instead of
    K copies.
    """
    return compute_log_probs(logits.clone(), indices_k, tp_group).squeeze(-1)


def topk_distillation_loss_function(
    args: Namespace,
    batch: dict,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute top-K logits-based distillation loss with tail trick.

    For each response token position, we compute the student's log-probability
    at each of the teacher's top-K token indices, append a tail bin for the
    remaining probability mass (both student and teacher), then compute
    D_KL(student || teacher) over this (K+1)-simplex.

    The ``sum_of_sample_mean`` reducer (provided by Slime) correctly handles
    per-sample averaging with the multi-turn loss_mask: only positions where
    loss_mask == 1 contribute to the loss.

    Reads from ``teacher_topk_log_probs`` ([T, K]) and ``teacher_topk_indices``
    ([T, K]) — separate fields that do not interfere with the legacy 1D
    ``teacher_log_probs`` used by the token-level OPD path.
    """
    teacher_topk_logprobs = batch["teacher_topk_log_probs"]
    teacher_topk_indices = batch["teacher_topk_indices"]
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)
    tp_group = mpu.get_tensor_model_parallel_group()

    K = args.distill_topk

    all_student_topk_logps = []
    all_teacher_topk_logps = []

    for i, (logits_chunk, tokens_chunk) in enumerate(get_responses(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    )):
        t_logps = teacher_topk_logprobs[i]
        t_indices = teacher_topk_indices[i]

        if not t_logps.is_cuda:
            t_logps = t_logps.to(device=logits_chunk.device)
        if not t_indices.is_cuda:
            t_indices = t_indices.to(device=logits_chunk.device)

        # With context parallelism, some CP ranks may receive zero response
        # tokens for a given sample.  fused_vocab_parallel_cross_entropy
        # cannot handle empty tensors, so skip them.
        if logits_chunk.shape[0] == 0:
            all_student_topk_logps.append(torch.zeros(0, K, device=logits_chunk.device))
            all_teacher_topk_logps.append(torch.zeros(0, K, device=logits_chunk.device))
            continue

        # Compute student log-prob at each of the K teacher-selected tokens.
        # Each call to compute_log_probs clones the logits (~1 GiB) and
        # fused_cross_entropy keeps intermediates for backward.  Without
        # checkpointing, K=50 calls hold 50 copies simultaneously (~54 GiB).
        # Gradient checkpointing discards intermediates after each forward
        # and recomputes them one at a time during backward.
        student_logps_k = []
        for k in range(K):
            if torch.is_grad_enabled():
                lp_k = torch.utils.checkpoint.checkpoint(
                    _compute_one_k,
                    logits_chunk,
                    t_indices[:, k],
                    tp_group,
                    use_reentrant=False,
                )
            else:
                lp_k = compute_log_probs(
                    logits_chunk, t_indices[:, k], tp_group,
                ).squeeze(-1)
            student_logps_k.append(lp_k)
        student_topk_logps = torch.stack(student_logps_k, dim=-1)  # [T_i, K]

        all_student_topk_logps.append(student_topk_logps)
        all_teacher_topk_logps.append(t_logps)

    # Concatenate across all samples in the micro-batch: [sum(T_i), K]
    student_topk = torch.cat(all_student_topk_logps, dim=0)
    teacher_topk = torch.cat(all_teacher_topk_logps, dim=0)

    # --- Tail trick ---
    # Compute log-probability of the "tail" bin = log(1 - sum(top-K probs))
    # Using numerically stable: log(1 - exp(x)) = log(-expm1(x))
    student_log_s = torch.logsumexp(student_topk, dim=-1, keepdim=True)
    student_log_s = torch.clamp(student_log_s, max=-1e-7)
    student_tail = torch.log(-torch.expm1(student_log_s))

    teacher_log_s = torch.logsumexp(teacher_topk, dim=-1, keepdim=True)
    teacher_log_s = torch.clamp(teacher_log_s, max=-1e-7)
    teacher_tail = torch.log(-torch.expm1(teacher_log_s))

    # Concatenate top-K + tail: [sum(T_i), K+1]
    student_with_tail = torch.cat([student_topk, student_tail], dim=-1)
    teacher_with_tail = torch.cat([teacher_topk, teacher_tail], dim=-1)

    # Reverse KL: D_KL(student || teacher)
    # = sum_k student(k) * (log student(k) - log teacher(k))
    # PyTorch kl_div(input, target, log_target=True) = target * (log_target - input)
    # So: input=teacher, target=student => student * (log_student - log_teacher)
    per_token_kl = F.kl_div(
        teacher_with_tail,
        student_with_tail,
        reduction="none",
        log_target=True,
    ).sum(dim=-1)  # [sum(T_i)]

    # sum_of_sample_mean applies per-sample (loss_mask * kl).sum() / max(mask.sum(), 1)
    kl_loss = sum_of_sample_mean(per_token_kl)

    loss = kl_loss

    # Optional entropy bonus
    entropy_loss = torch.tensor(0.0, device=logits.device)
    if args.entropy_coef != 0.0:
        student_probs = torch.exp(student_with_tail)
        entropy = -(student_probs * student_with_tail).sum(dim=-1)
        entropy_loss = sum_of_sample_mean(entropy)
        loss = loss - args.entropy_coef * entropy_loss

    # Ensure gradient flows even with empty micro-batches
    if per_token_kl.numel() == 0:
        loss = loss + 0 * logits.sum()

    reported_loss = {
        "loss": loss.clone().detach(),
        "kl_loss": kl_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
    }

    return loss, reported_loss
