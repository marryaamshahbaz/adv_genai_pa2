
import json
from pathlib import Path
import matplotlib.pyplot as plt

Path("plots").mkdir(exist_ok=True)


# sft training and val loss

with open("outputs/sft/metrics.jsonl") as f:
    sft_data = [json.loads(line) for line in f if line.strip()]

sft_train_steps = []
sft_train_loss = []
sft_val_steps = []
sft_val_loss = []

for i, row in enumerate(sft_data):
    step = row.get("step", i)
    record_type = row.get("type", row.get("split", "")).lower()

    if record_type == "train" and "loss" in row:
        sft_train_steps.append(step)
        sft_train_loss.append(row["loss"])

    elif record_type == "train" and "train_loss" in row:
        sft_train_steps.append(step)
        sft_train_loss.append(row["train_loss"])

    if record_type in ("validation", "val", "eval"):
        if "val_loss" in row:
            sft_val_steps.append(step)
            sft_val_loss.append(row["val_loss"])

        elif "eval_loss" in row:
            sft_val_steps.append(step)
            sft_val_loss.append(row["eval_loss"])

        elif "loss" in row:
            sft_val_steps.append(step)
            sft_val_loss.append(row["loss"])

print("SFT training records:", len(sft_train_loss))
print("SFT validation records:", len(sft_val_loss))

if sft_train_loss or sft_val_loss:
    plt.figure(figsize=(9, 5))

    if sft_train_loss:
        plt.plot(
            sft_train_steps,
            sft_train_loss,
            alpha=0.25,
            label="Training loss (raw)"
        )

        # Average every 20 steps to show the general trend.
        if len(sft_train_loss) >= 20:
            smooth_loss = []
            smooth_steps = []

            for i in range(19, len(sft_train_loss)):
                average = sum(sft_train_loss[i-19:i+1]) / 20
                smooth_loss.append(average)
                smooth_steps.append(sft_train_steps[i])

            plt.plot(
                smooth_steps,
                smooth_loss,
                linewidth=2,
                label="Training loss (20-step average)"
            )

    if sft_val_loss:
        plt.scatter(
            sft_val_steps,
            sft_val_loss,
            s=70,
            marker="o",
            color="red",
            label="Validation loss",
            zorder=5
        )

        if len(sft_val_loss) > 1:
            plt.plot(
                sft_val_steps,
                sft_val_loss,
                color="red",
                alpha=0.6
            )

    plt.title("SFT: Training and Validation Loss")
    plt.xlabel("Optimizer step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/sft_loss.png", dpi=200)
    plt.close()

    print("Saved plots/sft_loss.png")


# hh dpo training and val loss

with open("outputs/dpo_Hh/metrics.jsonl") as f:
    hh_data = [json.loads(line) for line in f if line.strip()]

hh_train_steps = []
hh_train_loss = []
hh_val_steps = []
hh_val_loss = []

for i, row in enumerate(hh_data):
    step = row.get("step", i)
    record_type = row.get("type", row.get("split", "")).lower()

    if record_type == "train" and "loss" in row:
        hh_train_steps.append(step)
        hh_train_loss.append(row["loss"])

    elif record_type == "train" and "train_loss" in row:
        hh_train_steps.append(step)
        hh_train_loss.append(row["train_loss"])

    if record_type in ("validation", "val", "eval"):
        if "val_loss" in row:
            hh_val_steps.append(step)
            hh_val_loss.append(row["val_loss"])

        elif "eval_loss" in row:
            hh_val_steps.append(step)
            hh_val_loss.append(row["eval_loss"])

        elif "loss" in row:
            hh_val_steps.append(step)
            hh_val_loss.append(row["loss"])

print("HH-DPO training records:", len(hh_train_loss))
print("HH-DPO validation records:", len(hh_val_loss))

if hh_train_loss or hh_val_loss:
    plt.figure(figsize=(9, 5))

    if hh_train_loss:
        plt.plot(
            hh_train_steps,
            hh_train_loss,
            alpha=0.25,
            label="Training loss (raw)"
        )

        if len(hh_train_loss) >= 20:
            smooth_loss = []
            smooth_steps = []

            for i in range(19, len(hh_train_loss)):
                average = sum(hh_train_loss[i-19:i+1]) / 20
                smooth_loss.append(average)
                smooth_steps.append(hh_train_steps[i])

            plt.plot(
                smooth_steps,
                smooth_loss,
                linewidth=2,
                label="Training loss (20-step average)"
            )

    if hh_val_loss:
        plt.scatter(
            hh_val_steps,
            hh_val_loss,
            s=70,
            color="red",
            label="Validation loss",
            zorder=5
        )

        if len(hh_val_loss) > 1:
            plt.plot(
                hh_val_steps,
                hh_val_loss,
                color="red",
                alpha=0.6
            )

    plt.title("HH-RLHF DPO: Training and Validation Loss")
    plt.xlabel("Optimizer step")
    plt.ylabel("DPO loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/hh_dpo_loss.png", dpi=200)
    plt.close()

    print("Saved plots/hh_dpo_loss.png")


# hh dpo reward accuracy 


hh_accuracy_steps = []
hh_accuracy = []

for i, row in enumerate(hh_data):
    step = row.get("step", i)

    for key in (
        "reward_accuracy",
        "val_reward_accuracy",
        "eval_reward_accuracy",
        "eval_rewards/accuracies"
    ):
        if key in row:
            hh_accuracy_steps.append(step)
            hh_accuracy.append(row[key])
            break

if hh_accuracy:
    plt.figure(figsize=(7, 5))

    if len(hh_accuracy) == 1:
        plt.bar(
            ["Final validation"],
            [hh_accuracy[0]],
            width=0.45
        )
        plt.text(
            0,
            hh_accuracy[0] + 0.02,
            f"{hh_accuracy[0]:.3f}",
            ha="center"
        )
    else:
        plt.plot(hh_accuracy_steps, hh_accuracy, marker="o")

    plt.title("HH-RLHF DPO: Validation Reward Accuracy")
    plt.ylabel("Reward accuracy")
    plt.ylim(0, 1.1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/hh_reward_accuracy.png", dpi=200)
    plt.close()

    print("Saved plots/hh_reward_accuracy.png")

# hh dpo reward margin

hh_margin_steps = []
hh_margin = []

for i, row in enumerate(hh_data):
    step = row.get("step", i)

    for key in (
        "reward_margin",
        "val_reward_margin",
        "eval_reward_margin",
        "eval_rewards/margins"
    ):
        if key in row:
            hh_margin_steps.append(step)
            hh_margin.append(row[key])
            break

if hh_margin:
    plt.figure(figsize=(7, 5))

    if len(hh_margin) == 1:
        plt.bar(["Final validation"], [hh_margin[0]], width=0.45)
        plt.text(
            0,
            hh_margin[0],
            f"{hh_margin[0]:.3f}",
            ha="center",
            va="bottom"
        )
    else:
        plt.plot(hh_margin_steps, hh_margin, marker="o")

    plt.title("HH-RLHF DPO: Validation Reward Margin")
    plt.ylabel("Reward margin")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/hh_reward_margin.png", dpi=200)
    plt.close()

    print("Saved plots/hh_reward_margin.png")


# dpo ultra feedback training and val loss


with open("outputs/dpo_uf/metrics.jsonl") as f:
    uf_data = [json.loads(line) for line in f if line.strip()]

uf_train_steps = []
uf_train_loss = []
uf_val_steps = []
uf_val_loss = []

for i, row in enumerate(uf_data):
    step = row.get("step", i)
    record_type = row.get("type", row.get("split", "")).lower()

    if record_type == "train" and "loss" in row:
        uf_train_steps.append(step)
        uf_train_loss.append(row["loss"])

    elif record_type == "train" and "train_loss" in row:
        uf_train_steps.append(step)
        uf_train_loss.append(row["train_loss"])

    if record_type in ("validation", "val", "eval"):
        if "val_loss" in row:
            uf_val_steps.append(step)
            uf_val_loss.append(row["val_loss"])

        elif "eval_loss" in row:
            uf_val_steps.append(step)
            uf_val_loss.append(row["eval_loss"])

        elif "loss" in row:
            uf_val_steps.append(step)
            uf_val_loss.append(row["loss"])

print("UltraFeedback training records:", len(uf_train_loss))
print("UltraFeedback validation records:", len(uf_val_loss))

if uf_train_loss or uf_val_loss:
    plt.figure(figsize=(9, 5))

    if uf_train_loss:
        plt.plot(
            uf_train_steps,
            uf_train_loss,
            alpha=0.25,
            label="Training loss (raw)"
        )

        if len(uf_train_loss) >= 20:
            smooth_loss = []
            smooth_steps = []

            for i in range(19, len(uf_train_loss)):
                average = sum(uf_train_loss[i-19:i+1]) / 20
                smooth_loss.append(average)
                smooth_steps.append(uf_train_steps[i])

            plt.plot(
                smooth_steps,
                smooth_loss,
                linewidth=2,
                label="Training loss (20-step average)"
            )

    if uf_val_loss:
        plt.scatter(
            uf_val_steps,
            uf_val_loss,
            s=70,
            color="red",
            label="Validation loss",
            zorder=5
        )

        if len(uf_val_loss) > 1:
            plt.plot(
                uf_val_steps,
                uf_val_loss,
                color="red",
                alpha=0.6
            )

    plt.title("UltraFeedback DPO: Training and Validation Loss")
    plt.xlabel("Optimizer step")
    plt.ylabel("DPO loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/ultrafeedback_dpo_loss.png", dpi=200)
    plt.close()

    print("Saved plots/ultrafeedback_dpo_loss.png")


# dpo ultra feedback reward accuracy

uf_accuracy_steps = []
uf_accuracy = []

for i, row in enumerate(uf_data):
    step = row.get("step", i)

    for key in (
        "reward_accuracy",
        "val_reward_accuracy",
        "eval_reward_accuracy",
        "eval_rewards/accuracies"
    ):
        if key in row:
            uf_accuracy_steps.append(step)
            uf_accuracy.append(row[key])
            break

if uf_accuracy:
    plt.figure(figsize=(7, 5))

    if len(uf_accuracy) == 1:
        plt.bar(["Final validation"], [uf_accuracy[0]], width=0.45)
        plt.text(
            0,
            uf_accuracy[0] + 0.02,
            f"{uf_accuracy[0]:.3f}",
            ha="center"
        )
    else:
        plt.plot(uf_accuracy_steps, uf_accuracy, marker="o")

    plt.title("UltraFeedback DPO: Validation Reward Accuracy")
    plt.ylabel("Reward accuracy")
    plt.ylim(0, 1.1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/ultrafeedback_reward_accuracy.png", dpi=200)
    plt.close()

    print("Saved plots/ultrafeedback_reward_accuracy.png")


# dpo ultra feedback reward margin

uf_margin_steps = []
uf_margin = []

for i, row in enumerate(uf_data):
    step = row.get("step", i)

    for key in (
        "reward_margin",
        "val_reward_margin",
        "eval_reward_margin",
        "eval_rewards/margins"
    ):
        if key in row:
            uf_margin_steps.append(step)
            uf_margin.append(row[key])
            break

if uf_margin:
    plt.figure(figsize=(7, 5))

    if len(uf_margin) == 1:
        plt.bar(["Final validation"], [uf_margin[0]], width=0.45)
        plt.text(
            0,
            uf_margin[0],
            f"{uf_margin[0]:.3f}",
            ha="center",
            va="bottom"
        )
    else:
        plt.plot(uf_margin_steps, uf_margin, marker="o")

    plt.title("UltraFeedback DPO: Validation Reward Margin")
    plt.ylabel("Reward margin")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/ultrafeedback_reward_margin.png", dpi=200)
    plt.close()

    print("Saved plots/ultrafeedback_reward_margin.png")
