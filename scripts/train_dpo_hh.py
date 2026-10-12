
import argparse
import random
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.dpo import train_dpo


MODEL_DATASET = "Anthropic/hh-rlhf"


def load_hh_data(tokenizer, split, num_examples, max_length, seed):
    """
    Load HH-RLHF preference pairs.
    """

    dataset = load_dataset(
        MODEL_DATASET,
        split=split,
        streaming=True
    ).shuffle(seed=seed, buffer_size=1000)

    examples = []

    for row in dataset:
        chosen_text = row["chosen"]
        rejected_text = row["rejected"]

        # Both strings contain the conversation and the final answer.
        # rsplit(..., 1) splits at the last Assistant marker.
        marker = "\n\nAssistant:"

        if marker not in chosen_text or marker not in rejected_text:
            continue

        chosen_prompt, chosen = chosen_text.rsplit(marker, 1)
        rejected_prompt, rejected = rejected_text.rsplit(marker, 1)

        # Both responses must belong to the same conversation.
        if chosen_prompt != rejected_prompt:
            continue

        instruction = chosen_prompt.strip()
        chosen = chosen.strip()
        rejected = rejected.strip()

        if not instruction or not chosen or not rejected:
            continue

        if chosen == rejected:
            continue

        # Use the same prompt format as our SFT training.
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

        prompt_ids = tokenizer.encode(
            prompt, add_special_tokens=False
        )

        # Leave some room for response tokens.
        if len(prompt_ids) >= max_length - 1:
            continue

        examples.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })

        if len(examples) == num_examples:
            break

    print(f"Loaded {len(examples)} HH-RLHF examples from {split}")
    return examples


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sft_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--train_size", type=int, default=15000)
    parser.add_argument("--val_size", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load the tokenizer from our trained SFT checkpoint.
    tokenizer = AutoTokenizer.from_pretrained(args.sft_checkpoint)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Both models start with exactly the same SFT weights.
    print("Loading policy model...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.sft_checkpoint,
        dtype=torch.float32
    )

    print("Loading reference model...")
    reference_model = AutoModelForCausalLM.from_pretrained(
        args.sft_checkpoint,
        dtype=torch.float32
    )

    # Use separate GPUs if available, otherwise share one GPU.
    if torch.cuda.device_count() >= 2:
        policy_device = torch.device("cuda:0")
        reference_device = torch.device("cuda:1")
    else:
        policy_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        reference_device = policy_device

    policy_model = policy_model.to(policy_device)
    reference_model = reference_model.to(reference_device)

    reference_model.requires_grad_(False)
    reference_model.eval()

    print("Policy device:", policy_device)
    print("Reference device:", reference_device)

    train_data = load_hh_data(
        tokenizer,
        "train",
        args.train_size,
        args.max_length,
        args.seed
    )

    val_data = load_hh_data(
        tokenizer,
        "test",
        args.val_size,
        args.max_length,
        args.seed
    )

    if len(train_data) != args.train_size:
        raise ValueError("Not enough HH-RLHF training pairs were loaded.")

    if len(val_data) != args.val_size:
        raise ValueError("Not enough HH-RLHF validation pairs were loaded.")

    print("\nStarting HH-RLHF DPO training...")

    train_dpo(
        policy_model,
        reference_model,
        tokenizer,
        train_data,
        val_data,
        output_dir=args.output_dir,
        beta=args.beta,
        max_length=args.max_length,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        learning_rate=args.learning_rate,
        epochs=args.epochs
    )


if __name__ == "__main__":
    main()
