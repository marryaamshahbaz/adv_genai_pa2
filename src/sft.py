
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from transformers import get_cosine_schedule_with_warmup
import os


def build_sft_batch(tokenizer, prompts, responses, max_length):
    """
    Convert prompt-response pairs into tensors for SFT.

    Returns input_ids, attention_mask, and labels.
    Only response tokens contribute to the loss.
    """

    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    # Use the tokenizer's padding token.
    # If it doesn't have one, we can use EOS for padding.
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    for prompt, response in zip(prompts, responses):

        # Convert the prompt and response into token IDs.
        # We add EOS ourselves instead of automatically adding special tokens.
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(response, add_special_tokens=False)

        # EOS teaches the model when to stop generating an answer.
        if tokenizer.eos_token_id is not None:
            response_ids.append(tokenizer.eos_token_id)

        # The model receives both the prompt and the correct response.
        input_ids = prompt_ids + response_ids

        # We only want to calculate loss on the response.
        # -100 means PyTorch will ignore that position in cross-entropy.
        labels = [-100] * len(prompt_ids) + response_ids

        # Limit the sequence length.
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

        # 1 means a real token that the model can attend to.
        attention_mask = [1] * len(input_ids)

        # Add padding so every example has the same length.
        padding_needed = max_length - len(input_ids)

        input_ids += [pad_id] * padding_needed
        attention_mask += [0] * padding_needed

        # Padding should not contribute to the loss either.
        labels += [-100] * padding_needed

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(labels)

    # Convert Python lists into tensors.
    # torch.long is the integer type expected for token IDs and labels.
    return {
        "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(all_attention_masks, dtype=torch.long),
        "labels": torch.tensor(all_labels, dtype=torch.long),
    }

def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    """
    Calculate the summed log probability of each response.

    We calculate probabilities only for response tokens,
    ignoring the prompt and padding.

    Returns one log probability per example.
    """

    # Reuse our SFT batching function.
    # This already creates the response-only labels.
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)

    # The tensors must be on the same device as the model.
    # This could be CPU or GPU.
    device = next(model.parameters()).device

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    # Forward pass: the model predicts the next token at every position.
    # logits shape: (batch_size, sequence_length, vocab_size)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )
    logits = outputs.logits

    # Logits at position t predict the token at position t+1.
    # Therefore, shift the predictions and labels to align them.
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    # Convert raw logits into log probabilities over the vocabulary.
    # dim=-1 means softmax is applied across the vocabulary dimension.
    log_probs = torch.log_softmax(shift_logits, dim=-1)

    # -100 is not a valid vocabulary index.
    # Replace ignored labels with 0 temporarily for gathering.
    safe_labels = shift_labels.clone()
    safe_labels[safe_labels == -100] = 0

    # Pick the log probability assigned to the correct token.
    # unsqueeze(-1) adds a final dimension for gather().
    # squeeze(-1) removes that dimension afterward.
    token_logps = torch.gather(
        log_probs,
        dim=-1,
        index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)

    # Mask out prompt and padding positions.
    # Multiplying by 0 removes their contribution.
    response_mask = (shift_labels != -100)
    token_logps = token_logps * response_mask

    # Sum over response tokens.
    # dim=1 is the sequence-length dimension.
    # Result shape: (batch_size,)
    response_logps = token_logps.sum(dim=1)

    return response_logps

def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """
    Calculate the mean negative log probability of response tokens.

    This is our SFT loss, which the optimizer will try to minimize.
    """

    # Get the summed log probability of each response.
    # Shape: (batch_size,)
    response_logps = compute_response_logprobs(
        model, tokenizer, prompts, responses, max_length
    )

    # Get the labels to count how many response tokens we have.
    # Prompt and padding positions have label -100.
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    labels = batch["labels"]

    # Count all response tokens across the batch.
    # (labels != -100) creates a boolean mask:
    # True for response tokens, False for prompt/padding.
    # .sum() counts the True values.
    num_response_tokens = (labels[:, 1:] != -100).sum()

    # Negative sign turns log probability maximization into loss minimization.
    # Dividing by the token count gives mean loss per response token.
    loss = -response_logps.sum() / num_response_tokens

    return loss



def train_sft(
    model,
    tokenizer,
    train_data,
    val_data,
    output_dir,
    max_length=512,
    batch_size=2,
    accumulation_steps=16,
    learning_rate=2e-5,
    epochs=1,
):
    """
    Train the model using SFT.

    Saves training/validation losses and the best/final checkpoints.
    """

    import os
    import json
    import math
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from torch.amp import autocast, GradScaler
    from torch.nn.utils import clip_grad_norm_
    from transformers import get_cosine_schedule_with_warmup

    device = next(model.parameters()).device

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False
    )

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.1
    )

    # One optimizer update happens after accumulation_steps microbatches.
    steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_steps = steps_per_epoch * epochs

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.03 * total_steps),
        num_training_steps=total_steps
    )

    # A100 supports BF16; use FP16 on GPUs without BF16 support.
    use_cuda = device.type == "cuda"
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    precision = torch.bfloat16 if use_bf16 else torch.float16

    scaler = GradScaler("cuda", enabled=use_cuda and not use_bf16)

    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.jsonl")

    best_val_loss = float("inf")
    global_step = 0

    # Save each training or validation result as one JSON line.
    def save_metric(metric):
        with open(metrics_path, "a") as file:
            file.write(json.dumps(metric) + "\n")

    # Clear previous metrics if this is a new run.
    open(metrics_path, "w").close()

    for epoch in range(epochs):

        model.train()

        total_train_loss = 0.0
        total_train_tokens = 0

        train_iterator = iter(train_loader)

        # Each window contains the microbatches for one optimizer update.
        for start in range(0, len(train_loader), accumulation_steps):

            window_size = min(
                accumulation_steps,
                len(train_loader) - start
            )

            window_batches = [
                next(train_iterator) for _ in range(window_size)
            ]

            # Count response tokens across the entire accumulation window.
            # We need this count to calculate the correct mean loss.
            window_tokens = 0

            for batch in window_batches:
                labels = build_sft_batch(
                    tokenizer,
                    batch["prompt"],
                    batch["response"],
                    max_length
                )["labels"]

                window_tokens += (labels[:, 1:] != -100).sum().item()

            window_loss = 0.0
            optimizer.zero_grad()

            for batch in window_batches:

                prompts = batch["prompt"]
                responses = batch["response"]

                labels = build_sft_batch(
                    tokenizer, prompts, responses, max_length
                )["labels"]

                token_count = (labels[:, 1:] != -100).sum().item()

                with autocast("cuda", dtype=precision, enabled=use_cuda):
                    loss = compute_sft_loss(
                        model, tokenizer, prompts, responses, max_length
                    )

                # Our SFT loss is averaged over response tokens.
                #
                # Multiplying by token_count gives the total token loss.
                # Dividing by window_tokens averages over all response
                # tokens in the effective batch, not each microbatch.
                scaled_loss = loss * (token_count / window_tokens)

                # Accumulate gradients; do not update weights yet.
                scaler.scale(scaled_loss).backward()

                window_loss += loss.detach().item() * token_count

            # Unscale gradients before applying gradient clipping.
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update model weights after all microbatches.
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()

            # Don't advance the scheduler if an FP16 update was skipped.
            if not scaler.is_enabled() or scaler.get_scale() >= old_scale:
                scheduler.step()

            global_step += 1
            total_train_loss += window_loss
            total_train_tokens += window_tokens

            # Record the mean loss for this optimizer step.
            step_loss = window_loss / window_tokens

            save_metric({
                "type": "train",
                "epoch": epoch + 1,
                "step": global_step,
                "loss": step_loss
            })

            if global_step % 50 == 0:
                print(
                    f"Step {global_step}/{total_steps} "
                    f"| Loss: {step_loss:.4f}",
                    flush=True
                )

        avg_train_loss = total_train_loss / total_train_tokens

        # Validation: measure loss without updating model weights.
        model.eval()

        total_val_loss = 0.0
        total_val_tokens = 0

        with torch.no_grad():
            for batch in val_loader:

                prompts = batch["prompt"]
                responses = batch["response"]

                labels = build_sft_batch(
                    tokenizer, prompts, responses, max_length
                )["labels"]

                token_count = (labels[:, 1:] != -100).sum().item()

                with autocast("cuda", dtype=precision, enabled=use_cuda):
                    loss = compute_sft_loss(
                        model, tokenizer, prompts, responses, max_length
                    )

                total_val_loss += loss.item() * token_count
                total_val_tokens += token_count

        avg_val_loss = total_val_loss / total_val_tokens

        # Save the end-of-epoch training and validation results.
        save_metric({
            "type": "validation",
            "epoch": epoch + 1,
            "step": global_step,
            "train_loss": avg_train_loss,
            "loss": avg_val_loss
        })

        print(f"\nEpoch {epoch + 1} completed")
        print(f"Average training loss: {avg_train_loss:.4f}")
        print(f"Validation loss: {avg_val_loss:.4f}")

        # Save the checkpoint with the lowest validation loss.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            best_path = os.path.join(output_dir, "best")
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)

            print("Best checkpoint saved.")

    # Save the model after the last training update.
    final_path = os.path.join(output_dir, "final")

    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    print("\nSFT training completed.")
    print("Metrics:", metrics_path)
    print("Best checkpoint:", os.path.join(output_dir, "best"))
    print("Final checkpoint:", final_path)
