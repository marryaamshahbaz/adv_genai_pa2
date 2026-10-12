
import argparse
import json
import os
import random

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer


MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
DATA_PATH = "data/persona_lora/ramsay_style_preferences.json"


def load_persona_data(seed):
    # Load the provided prompt, chosen, and rejected preference pairs.
    with open(DATA_PATH, "r") as file:
        data = json.load(file)

    random.Random(seed).shuffle(data)

    # Hold out 10% of the dataset for validation.
    val_size = int(len(data) * 0.1)

    val_data = data[:val_size]
    train_data = data[val_size:]

    # TRL expects conversational messages for instruction-tuned models.
    def format_example(example):
        return {
            "prompt": [
                {"role": "user", "content": example["prompt"]}
            ],
            "chosen": [
                {"role": "assistant", "content": example["chosen"]}
            ],
            "rejected": [
                {"role": "assistant", "content": example["rejected"]}
            ]
        }

    train_dataset = Dataset.from_list(
        [format_example(row) for row in train_data]
    )

    val_dataset = Dataset.from_list(
        [format_example(row) for row in val_data]
    )

    print("Training pairs:", len(train_dataset))
    print("Validation pairs:", len(val_dataset))

    return train_dataset, val_dataset


def train_persona(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    train_dataset, val_dataset = load_persona_data(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    # Load the original instruction-tuned model.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    )

    # Only the LoRA adapter weights will be trainable.
    lora_config = LoraConfig(
        r=12,
        lora_alpha=24,
        lora_dropout=0.08,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    training_args = DPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        beta=0.1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=3,
        num_train_epochs=2,
        max_length=512,
        fp16=torch.cuda.is_available(),
        bf16=False,
        disable_dropout=False,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=5,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        peft_config=lora_config
    )

    # Confirm LoRA is training only a small subset of parameters.
    trainer.model.print_trainable_parameters()

    trainer.train()

    # Evaluate on the held-out preference pairs.
    metrics = trainer.evaluate()

    trainer.save_model(os.path.join(args.output_dir, "adapter"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "adapter"))

    trainer.save_state()

    with open(os.path.join(args.output_dir, "eval_metrics.json"), "w") as file:
        json.dump(metrics, file, indent=2)

    with open(os.path.join(args.output_dir, "training_history.json"), "w") as file:
        json.dump(trainer.state.log_history, file, indent=2)

    print("Validation metrics:", metrics)
    print("Saved adapter to:", os.path.join(args.output_dir, "adapter"))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train_persona(args)


if __name__ == "__main__":
    main()
