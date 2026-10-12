"""Stable grading interface for the SFT and DPO implementation.

Keep the function signatures below unchanged. Each function should be a thin
wrapper around your own implementation; you do not need to place the training
code itself in this file.
"""

from __future__ import annotations


def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Return input_ids, attention_mask, and response-only labels."""

    from src.sft import build_sft_batch as make_batch

    return make_batch(tokenizer, prompts, responses, max_length)

def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    """Return one summed response-token log-probability per input row."""

    from src.sft import compute_response_logprobs as get_logps

    return get_logps(model, tokenizer, prompts, responses, max_length)


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Return a scalar response-only SFT loss."""

    from src.sft import compute_sft_loss as get_sft_loss

    return get_sft_loss(model, tokenizer, prompts, responses, max_length)

def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Return separate response-masked LM batches for chosen and rejected rows."""

    raise NotImplementedError("Connect this adapter to your DPO batch builder")



def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Return separate response-masked batches for chosen and rejected rows."""

    from src.dpo import build_dpo_batch as make_dpo_batch

    return make_dpo_batch(
        tokenizer, prompts, chosen_responses, rejected_responses, max_length
    )


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Return the mean DPO loss from the four log probability tensors."""

    from src.dpo import compute_dpo_loss_from_logps as get_dpo_loss

    return get_dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta,
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
    """Calculate DPO loss using the policy and reference models."""

    from src.dpo import compute_dpo_loss as get_dpo_loss

    return get_dpo_loss(
        policy_model,
        reference_model,
        tokenizer,
        prompts,
        chosen_responses,
        rejected_responses,
        beta,
        max_length,
    )

