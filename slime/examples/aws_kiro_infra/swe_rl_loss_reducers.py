"""Custom PG loss reducers for swe-rl.

Hook in via slime's ``--custom-pg-loss-reducer-function-path`` flag. The
factory signature (see ``slime/backends/megatron_utils/loss.py:704-710``):

    def factory(total_lengths, response_lengths, loss_masks, calculate_per_token_loss)
        -> Callable[[torch.Tensor], torch.Tensor]

The returned reducer takes a per-response-token tensor (pg_loss) and returns
a scalar loss contribution.
"""

import os
from typing import Callable, List

import torch

from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean


def dr_grpo_length_norm_reducer(
    total_lengths: List[int],
    response_lengths: List[int],
    loss_masks: List[torch.Tensor],
    calculate_per_token_loss: bool,          # slime passes this but we override
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Dr.GRPO / DeepSWE ``seq-mean-token-sum`` length normalization.

    Per-sample loss = (Σ_t mask_t · x_t) / MAX_RESP   (MAX_RESP is a constant)
    Batch  loss     = Σ_i per_sample_loss_i

    The denominator is a *constant* (``SWE_RL_MAX_RESPONSE_LENGTH`` env var,
    default 32768), not the sample's own length. This eliminates GRPO's
    length bias ("errors get rewarded for being longer") — see Dr.GRPO
    (https://arxiv.org/abs/2503.20783) and DeepSWE blog §2.3.

    Implementation note: we delegate the masked-token summation to slime's
    built-in ``sum_of_token`` reducer (via ``calculate_per_token_loss=True``
    path of ``get_sum_of_sample_mean``). That reducer handles Context Parallel
    chunking correctly for both CP=1 and CP>1 cases. We then just divide the
    aggregate by the constant ``MAX_RESP`` — this is numerically identical to
    ``Σ_i (Σ_t mask_t · x_t) / MAX_RESP`` since the constant factors out.
    """
    max_resp = int(os.environ.get("SWE_RL_MAX_RESPONSE_LENGTH", "32768"))

    # sum_of_token returns Σ_i Σ_t mask_it · x_it across all samples in the
    # local batch. Dividing by MAX_RESP yields the Dr.GRPO per-sample-summed
    # loss (equivalent to Σ_i Σ_t mask_it · x_it / MAX_RESP).
    sum_of_token = get_sum_of_sample_mean(
        total_lengths,
        response_lengths,
        loss_masks,
        calculate_per_token_loss=True,
    )

    def reducer(x: torch.Tensor) -> torch.Tensor:
        return sum_of_token(x) / max_resp

    return reducer
