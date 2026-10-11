
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
    Train the model using supervised fine-tuning.

    train_data and val_data contain dictionaries with:
        {"prompt": ..., "response": ...}

    Save the best and final model checkpoints.
    """

    device = next(model.parameters()).device

    # AdamW updates the model weights using the gradients.
    # weight_decay discourages excessively large weights.
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.1
    )

    # DataLoader groups training examples into batches.
    # shuffle=True changes their order each epoch.
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False
    )

    # One optimizer step happens after accumulation_steps batches.
    steps_per_epoch = (
        len(train_loader) + accumulation_steps - 1
    ) // accumulation_steps

    total_steps = steps_per_epoch * epochs
    warmup_steps = int(0.03 * total_steps)

    # Start with a small learning rate, then gradually reduce it.
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # GradScaler helps avoid underflow when training with fp16.
    use_cuda = device.type == "cuda"
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    precision = torch.bfloat16 if use_bf16 else torch.float16

    scaler = GradScaler("cuda", enabled=use_cuda and not use_bf16)

    best_val_loss = float("inf")
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(epochs):

        model.train()
        optimizer.zero_grad()

        total_train_loss = 0.0

        for step, batch in enumerate(train_loader):

            # DataLoader gives lists of prompts and responses.
            prompts = batch["prompt"]
            responses = batch["response"]

            # Autocast performs suitable operations in lower precision.
            # This reduces GPU memory usage.
            with autocast("cuda", dtype=precision, enabled=use_cuda):

                loss = compute_sft_loss(
                    model, tokenizer, prompts, responses, max_length
                )

            total_train_loss += loss.item()

            # Divide by the number of microbatches in this
            # accumulation window so gradients remain averaged.
            window_start = (step // accumulation_steps) * accumulation_steps
            window_size = min(
                accumulation_steps,
                len(train_loader) - window_start
            )

            scaled_loss = loss / window_size

            # Backward calculates gradients for all trainable weights.
            # Gradients accumulate until optimizer.zero_grad() is called.
            scaler.scale(scaled_loss).backward()

            # Update after enough batches, or at the last batch.
            should_update = (
                (step + 1) % accumulation_steps == 0
                or (step + 1) == len(train_loader)
            )

            if should_update:

                # Restore gradients to their actual scale before clipping.
                scaler.unscale_(optimizer)

                # Limit the overall gradient norm to 1.0.
                clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Update the model parameters.
                scaler.step(optimizer)
                scaler.update()

                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % 50 == 0:
                print(
                    f"Epoch {epoch + 1}, Batch {step + 1}, "
                    f"Loss: {loss.item():.4f}"
                )

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation measures how well the model predicts
        # responses it was not trained on.
        model.eval()
        total_val_loss = 0.0
        total_val_tokens = 0

        # No gradients are needed during validation.
        with torch.no_grad():

            for batch in val_loader:

                prompts = batch["prompt"]
                responses = batch["response"]

                with autocast("cuda", dtype=precision, enabled=use_cuda):

                    loss = compute_sft_loss(
                        model, tokenizer, prompts, responses, max_length
                    )

                # Weight each batch by its number of response tokens
                # so validation loss is averaged fairly.
                labels = build_sft_batch(
                    tokenizer, prompts, responses, max_length
                )["labels"]

                token_count = (labels[:, 1:] != -100).sum().item()

                total_val_loss += loss.item() * token_count
                total_val_tokens += token_count

        avg_val_loss = total_val_loss / total_val_tokens

        print(f"\nEpoch {epoch + 1} finished")
        print(f"Training loss: {avg_train_loss:.4f}")
        print(f"Validation loss: {avg_val_loss:.4f}")

        # Save the model that achieves the lowest validation loss.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            model.save_pretrained(os.path.join(output_dir, "best"))
            tokenizer.save_pretrained(os.path.join(output_dir, "best"))

            print("Saved new best checkpoint.")

    # Also save the model after the final epoch.
    model.save_pretrained(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))

    print("\nSFT training completed.")
