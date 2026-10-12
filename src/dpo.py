
import torch
import os
import json
import math
import torch.nn.functional as F

from src.sft import build_sft_batch, compute_response_logprobs



from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """
    Prepare the chosen and rejected responses for DPO.

    Each batch contains input_ids, attention_mask, and labels,
    just like our SFT batches.
    """

    # The chosen and rejected responses use the same prompts.
    # Our SFT batching function already handles padding, EOS tokens, and response-only labels.
    # so we make two separate SFT batches for the chosen and rejected responses.
    chosen_batch = build_sft_batch(
        tokenizer, prompts, chosen_responses, max_length
    )

    rejected_batch = build_sft_batch(
        tokenizer, prompts, rejected_responses, max_length
    )

    return {
        "chosen": chosen_batch,
        "rejected": rejected_batch
    }

def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """
    Calculate the DPO loss from the four response log probabilities.

    We want the policy to prefer chosen responses over rejected ones,
    MORE than the original SFT reference model already did.
    """

    # STEP 1: Calculate the policy preference gap (delta_theta).
    #
    # delta_theta = log pi_theta(y_w | x) - log pi_theta(y_l | x)
    #
    # pi_theta is our trainable policy model.
    # y_w is the chosen (winning) response.
    # y_l is the rejected (losing) response.
    #
    # A larger gap means the policy prefers the chosen response more.
    policy_gap = policy_chosen_logps - policy_rejected_logps

    # STEP 2: Calculate the reference preference gap (delta_ref).
    #
    # delta_ref = log pi_ref(y_w | x) - log pi_ref(y_l | x)
    #
    # pi_ref is the frozen SFT model.
    # This tells us how much the original model preferred the chosen response before DPO training.
    reference_gap = reference_chosen_logps - reference_rejected_logps

    # STEP 3: Compare the two preference gaps.
    #
    # delta_theta - delta_ref
    #
    # We want this difference to increase during training.
    # A positive value means the policy has improved its relative preference for the chosen response compared with the reference
    preference_difference = policy_gap - reference_gap

    # STEP 4: Multiply the difference by beta
    #
    # beta * (delta_theta - delta_ref)
    #
    # beta controls the scale of the preference comparison
    # Our assignment recommends beta = 0.1
    logits = beta * preference_difference

    # STEP 5: Apply the DPO loss.
    #
    # L_DPO = -log(sigmoid(beta * (delta_theta - delta_ref)))
    #
    # F.logsigmoid(logits) computes log(sigmoid(logits)).
    # The negative sign converts it into a loss that we minimize 
    #
    # .mean() takes the average loss across all preference pairs
    # Each pair contributes one DPO loss value
    loss = -F.logsigmoid(logits).mean()

    return loss



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
    """
    Compute DPO loss using the trainable policy and frozen reference.
    """

    # Get the policy's log probabilities for both responses.
    # We need gradients here to update the policy during training.
    policy_chosen_logps = compute_response_logprobs(
        policy_model, tokenizer, prompts, chosen_responses, max_length
    )

    policy_rejected_logps = compute_response_logprobs(
        policy_model, tokenizer, prompts, rejected_responses, max_length
    )

    # The reference model is only used for comparison, so no gradients.
    with torch.no_grad(): # this command so gradients are not computed for the reference model and hence their gradient graphs are not stored in memory.
        reference_chosen_logps = compute_response_logprobs(
            reference_model, tokenizer, prompts, chosen_responses, max_length
        )

        reference_rejected_logps = compute_response_logprobs(
            reference_model, tokenizer, prompts, rejected_responses, max_length
        )

    # Move reference scores to the policy's device in case
    # the two models are running on different GPUs.
    policy_device = policy_chosen_logps.device
    reference_chosen_logps = reference_chosen_logps.to(policy_device)
    reference_rejected_logps = reference_rejected_logps.to(policy_device)

    # Calculate the DPO loss from the policy and reference preference gaps.
    loss = compute_dpo_loss_from_logps(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta,
    )

    return loss




def evaluate_dpo(policy_model, reference_model, tokenizer, val_loader, beta, max_length):
    """
    Evaluate the DPO model using validation loss,
    reward accuracy, and reward margin.
    """

    policy_model.eval()
    reference_model.eval()

    total_loss = 0.0
    correct = 0
    total_margin = 0.0
    total_pairs = 0

    # Validation doesn't require gradients since we're not updating weights.
    with torch.no_grad():
        for batch in val_loader:
            prompts = batch["prompt"]
            chosen = batch["chosen"]
            rejected = batch["rejected"]

            # Get the policy's log probabilities for both responses.
            policy_chosen = compute_response_logprobs(
                policy_model, tokenizer, prompts, chosen, max_length
            )
            policy_rejected = compute_response_logprobs(
                policy_model, tokenizer, prompts, rejected, max_length
            )

            # The reference scores are moved to the policy's device
            # in case the models are on different GPUs.
            ref_chosen = compute_response_logprobs(
                reference_model, tokenizer, prompts, chosen, max_length
            ).to(policy_chosen.device)

            ref_rejected = compute_response_logprobs(
                reference_model, tokenizer, prompts, rejected, max_length
            ).to(policy_chosen.device)

            # Calculate DPO loss from the four log probabilities.
            loss = compute_dpo_loss_from_logps(
                policy_chosen,
                policy_rejected,
                ref_chosen,
                ref_rejected,
                beta
            )

            # Implicit reward = beta * (policy logp - reference logp).
            # This measures how much the policy has changed from the reference.
            chosen_reward = beta * (policy_chosen - ref_chosen)
            rejected_reward = beta * (policy_rejected - ref_rejected)

            # Reward margin = chosen reward - rejected reward.
            # A positive margin means the policy favors the chosen response
            # more strongly relative to the reference.
            margins = chosen_reward - rejected_reward

            # Compare each margin with zero
            # This returns a boolean tensor containing True or False
            correct_predictions = margins > 0

            # Convert True/False to 1/0 and count the correct pairs
            num_correct = correct_predictions.int().sum()
            correct += num_correct.item()

            # Add the margins across all pairs in this batch.
            total_margin += margins.sum().item()

            # Weight batch loss by its size so the final average is calculated over all pairs, not over batches.
            batch_size = len(prompts)
            total_pairs += batch_size
            total_loss += loss.item() * batch_size

    return {
        "val_loss": total_loss / total_pairs,
        "reward_accuracy": correct / total_pairs,
        "reward_margin": total_margin / total_pairs
    }



def train_dpo(
    policy_model,
    reference_model,
    tokenizer,
    train_data,
    val_data,
    output_dir,
    beta=0.1,
    max_length=512,
    batch_size=1,
    accumulation_steps=32,
    learning_rate=1e-6,
    epochs=1,
):
    """
    Train the policy model using DPO and evaluate it after each epoch.
    """

    device = next(policy_model.parameters()).device

    # The reference is only used for comparison.
    reference_model.requires_grad_(False)
    reference_model.eval()

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )

    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False
    )

    optimizer = AdamW(
        policy_model.parameters(),
        lr=learning_rate,
        weight_decay=0.1
    )

    steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_steps = steps_per_epoch * epochs

    # Use bfloat16 on A100, otherwise float16 on supported CUDA GPUs.
    use_cuda = device.type == "cuda"
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    precision = torch.bfloat16 if use_bf16 else torch.float16

    scaler = GradScaler("cuda", enabled=use_cuda and not use_bf16)

    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.jsonl")

    def save_metric(metric):
        with open(metrics_path, "a") as file:
            file.write(json.dumps(metric) + "\n")

    # Clear metrics from any previous run in this directory.
    open(metrics_path, "w").close()

    best_accuracy = -1.0
    global_step = 0

    for epoch in range(epochs):

        policy_model.train()
        total_train_loss = 0.0
        total_train_pairs = 0

        optimizer.zero_grad()

        window_loss = 0.0

        for step, batch in enumerate(train_loader):

            prompts = batch["prompt"]
            chosen = batch["chosen"]
            rejected = batch["rejected"]

            with autocast("cuda", dtype=precision, enabled=use_cuda):
                loss = compute_dpo_loss(
                    policy_model,
                    reference_model,
                    tokenizer,
                    prompts,
                    chosen,
                    rejected,
                    beta,
                    max_length
                )

            # Accumulate gradients before updating the policy.
            # The last window may have fewer than 32 microbatches.
            # step // accumulation_steps does integer division to find the start of the current window
            window_start = (step // accumulation_steps) * accumulation_steps
            window_size = min(
                accumulation_steps,
                len(train_loader) - window_start
            )

            # Keep track of the average loss across this accumulation window.
            window_loss += loss.item() / window_size

            scaled_loss = loss / window_size
            scaler.scale(scaled_loss).backward()

            total_train_loss += loss.item() * len(prompts)
            total_train_pairs += len(prompts)

            should_update = ( # we update when we reach the end of a window or the end of the epoch
                (step + 1) % accumulation_steps == 0
                or (step + 1) == len(train_loader)
            )

            if should_update:
                scaler.unscale_(optimizer) # unscale restores the original gradient values before clipping
                clip_grad_norm_(policy_model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                global_step += 1

                if global_step % 10 == 0:
                    print(
                        f"Step {global_step}/{total_steps} "
                        f"| DPO loss: {window_loss:.4f}",
                        flush=True
                    )

                save_metric({
                    "type": "train",
                    "epoch": epoch + 1,
                    "step": global_step,
                    "loss": window_loss
                })

                # Start calculating a fresh average for the next window.
                window_loss = 0.0

        avg_train_loss = total_train_loss / total_train_pairs

        # Validate after the epoch and choose the best checkpoint
        # using reward accuracy.
        metrics = evaluate_dpo(
            policy_model,
            reference_model,
            tokenizer,
            val_loader,
            beta,
            max_length
        )

        save_metric({
            "type": "validation",
            "epoch": epoch + 1,
            "step": global_step,
            "train_loss": avg_train_loss,
            **metrics
        })

        print(f"\nEpoch {epoch + 1} completed")
        print(f"Average training loss: {avg_train_loss:.4f}")
        print(f"Validation DPO loss: {metrics['val_loss']:.4f}")
        print(f"Reward accuracy: {metrics['reward_accuracy']:.4f}")
        print(f"Reward margin: {metrics['reward_margin']:.4f}")

        if metrics["reward_accuracy"] > best_accuracy:
            best_accuracy = metrics["reward_accuracy"]

            best_path = os.path.join(output_dir, "best")
            policy_model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)

            print("Best DPO checkpoint saved.")

    final_path = os.path.join(output_dir, "final")
    policy_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    print("\nDPO training completed.")
    print("Metrics:", metrics_path)
    print("Best checkpoint:", os.path.join(output_dir, "best"))
    print("Final checkpoint:", final_path)
