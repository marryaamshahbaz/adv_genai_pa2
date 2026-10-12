
import argparse
import random
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.dpo import train_dpo


MODEL_DATASET = "HuggingFaceH4/ultrafeedback_binarized"


def load_ultrafeedback_data(tokenizer, split, num_examples, max_length, seed):
    """
    Load UltraFeedback chosen/rejected preference pairs.
    """

    dataset = load_dataset(
        MODEL_DATASET,
        split=split,
        streaming=True
    ).shuffle(seed=seed, buffer_size=1000)

    examples = []

    for row in dataset:
        instruction = row["prompt"]

        chosen_messages = row["chosen"]
        rejected_messages = row["rejected"]

        if not chosen_messages or not rejected_messages:
            continue

        # Both responses should be assistant messages.
        if chosen_messages[-1]["role"] != "assistant":
            continue

        if rejected_messages[-1]["role"] != "assistant":
            continue

        chosen = chosen_messages[-1]["content"]
        rejected = rejected_messages[-1]["content"]

        if not instruction.strip() or not chosen.strip() or not rejected.strip():
            continue

        if chosen.strip() == rejected.strip():
            continue

        # Keep the same prompt format we used for SFT.
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

        prompt_ids = tokenizer.encode(
            prompt, add_special_tokens=False
        )

        if len(prompt_ids) >= max_length - 1:
            continue

        examples.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })

        if len(examples) == num_examples:
            break

    print(f"Loaded {len(examples)} UltraFeedback examples from {split}")
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

    tokenizer = AutoTokenizer.from_pretrained(args.sft_checkpoint)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load two independent copies of the SFT checkpoint.
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

    train_data = load_ultrafeedback_data(
        tokenizer,
        "train_prefs",
        args.train_size,
        args.max_length,
        args.seed
    )

    val_data = load_ultrafeedback_data(
        tokenizer,
        "test_prefs",
        args.val_size,
        args.max_length,
        args.seed
    )

    if len(train_data) != args.train_size:
        raise ValueError("Not enough UltraFeedback training pairs were loaded.")

    if len(val_data) != args.val_size:
        raise ValueError("Not enough UltraFeedback validation pairs were loaded.")

    print("\nStarting UltraFeedback DPO training...")

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
