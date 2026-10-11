
import argparse
import random
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.sft import train_sft


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATASET_NAME = "HuggingFaceH4/ultrachat_200k"


def load_sft_data(tokenizer, split, num_examples, max_length, seed):
    """
    Load UltraChat examples and convert them into single-turn
    prompt-response pairs for our SFT training function.
    """

    # streaming=True lets us read examples without first
    # downloading the entire dataset.
    dataset = load_dataset(
        DATASET_NAME,
        split=split,
        streaming=True
    )

    # Shuffle the examples reproducibly using our chosen seed.
    dataset = dataset.shuffle(seed=seed, buffer_size=1000)

    examples = []

    for row in dataset:

        # UltraChat stores conversations as a list of messages.
        # We only use the first user instruction and assistant reply.
        messages = row["messages"]

        if len(messages) < 2:
            continue

        if messages[0]["role"] != "user":
            continue

        if messages[1]["role"] != "assistant":
            continue

        instruction = messages[0]["content"]
        response = messages[1]["content"]

        if not instruction.strip() or not response.strip():
            continue

        # Use the same prompt format for SFT and later DPO.
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

        # Keep examples where at least one response token and
        # its preceding context fit within max_length.
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

        if len(prompt_ids) >= max_length - 1:
            continue

        examples.append({
            "prompt": prompt,
            "response": response
        })

        if len(examples) == num_examples:
            break

    print(f"Loaded {len(examples)} examples from {split}")

    return examples


def main():

    # Command-line arguments let us change training sizes
    # without editing the script every time.
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_size", type=int, default=30000)
    parser.add_argument("--val_size", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accumulation_steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/sft")

    args = parser.parse_args()

    # Set seeds so random operations are reproducible.
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load the tokenizer associated with Qwen.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Our batch builder needs a padding token.
    # Qwen has an EOS token which can also be used for padding.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Download the pretrained model with FP32 parameters.
    # Our training loop uses autocast for lower-precision operations.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # .to(device) moves the model's parameters onto the GPU or CPU.
    model = model.to(device)

    print("Training device:", device)

    # Build our training and validation datasets.
    train_data = load_sft_data(
        tokenizer,
        split="train_sft",
        num_examples=args.train_size,
        max_length=args.max_length,
        seed=args.seed
    )

    val_data = load_sft_data(
        tokenizer,
        split="test_sft",
        num_examples=args.val_size,
        max_length=args.max_length,
        seed=args.seed
    )

    # Make sure we obtained the requested number of examples.
    if len(train_data) != args.train_size or len(val_data) != args.val_size:
        raise ValueError("Could not load the requested number of SFT examples.")

    print("\nStarting supervised fine-tuning...")

    # This is the training loop we already wrote in src/sft.py.
    train_sft(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        val_data=val_data,
        output_dir=args.output_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        learning_rate=args.learning_rate,
        epochs=args.epochs
    )


if __name__ == "__main__":
    main()
