
import argparse
import json
import random
import os
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATASET_NAME = "tatsu-lab/alpaca_eval"


def load_eval_prompts(num_examples, seed):
    # Load AlpacaEval and randomly select 300 prompts using seed 42.
    dataset = load_dataset(DATASET_NAME, "alpaca_eval", split="eval")

    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)

    prompts = []

    for index in indices[:num_examples]:
        prompts.append({
            "id": str(index),
            "instruction": dataset[index]["instruction"]
        })

    return prompts


def generate_baseline(model, tokenizer, prompts, max_new_tokens):
    model.eval()
    device = next(model.parameters()).device

    outputs = []

    for i, example in enumerate(prompts):
        instruction = example["instruction"]

        # Use the same prompt format as SFT.
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate an answer without updating model weights.
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        # Remove the input tokens so we only keep the generated response.
        prompt_length = inputs["input_ids"].shape[1]
        response_ids = generated_ids[0][prompt_length:]

        response = tokenizer.decode(
            response_ids,
            skip_special_tokens=True
        )

        outputs.append({
            "id": example["id"],
            "instruction": instruction,
            "output": response,
            "generator": "qwen2.5-0.5b-base",
            "dataset": "alpaca_eval"
        })

        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{len(prompts)} responses")

    return outputs


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default="outputs/baseline")
    parser.add_argument("--num_examples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=256)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    prompts = load_eval_prompts(args.num_examples, args.seed)

    # Save the exact prompt subset for later SFT/DPO evaluation.
    with open(os.path.join(args.output_dir, "alpaca_eval_prompts.json"), "w") as file:
        json.dump(prompts, file, indent=2)

    outputs = generate_baseline(
        model, tokenizer, prompts, args.max_new_tokens
    )

    with open(os.path.join(args.output_dir, "baseline_outputs.json"), "w") as file:
        json.dump(outputs, file, indent=2)

    print("Baseline generation completed.")


if __name__ == "__main__":
    main()
