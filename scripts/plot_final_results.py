import json
from pathlib import Path
import matplotlib.pyplot as plt

Path("plots").mkdir(exist_ok=True)

# ==========================================================
# TASK 5: PERSONA LORA COMPARISON
# ==========================================================
# Using your actual reported metrics:
# Run A
run_a_val_loss = 0.5442184805870056
run_a_reward_margin = 0.3265659511089325

# Run B
run_b_val_loss = 0.36434122920036316
run_b_reward_margin = 0.8420848250389099

# Run C
run_c_val_loss = 0.10795331746339798
run_c_reward_margin = 2.325434923171997

runs = ["Run A", "Run B", "Run C"]

val_losses = [
    run_a_val_loss,
    run_b_val_loss,
    run_c_val_loss
]

reward_margins = [
    run_a_reward_margin,
    run_b_reward_margin,
    run_c_reward_margin
]

# ----- Task 5 chart 1: validation loss -----
plt.figure(figsize=(7, 5))
plt.bar(runs, val_losses)
plt.title("Task 5: Persona LoRA Validation Loss")
plt.ylabel("Validation loss")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("plots/task5_persona_val_loss.png", dpi=200)
plt.close()

print("Saved plots/task5_persona_val_loss.png")

# ----- Task 5 chart 2: reward margin -----
plt.figure(figsize=(7, 5))
plt.bar(runs, reward_margins)
plt.title("Task 5: Persona LoRA Reward Margin")
plt.ylabel("Reward margin")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("plots/task5_persona_reward_margin.png", dpi=200)
plt.close()

print("Saved plots/task5_persona_reward_margin.png")


# ==========================================================
# TASK 7: GEMINI JUDGE RESULTS
# ==========================================================
# This part reads the summary JSON files you saved from Task 7.
# Adjust the paths if your files are in a slightly different folder.

with open("outputs/task7/hh_summary.json") as f:
    hh_summary = json.load(f)

with open("outputs/task7/alpaca_summary.json") as f:
    alpaca_summary = json.load(f)

# We use counts if available. If counts are missing, you can switch to percentages.
hh_counts = hh_summary["counts"]
alpaca_counts = alpaca_summary["counts"]

hh_dpo = hh_counts.get("dpo_wins", 0)
hh_sft = hh_counts.get("sft_wins", 0)
hh_ties = hh_counts.get("ties", 0)

alpaca_dpo = alpaca_counts.get("dpo_wins", 0)
alpaca_sft = alpaca_counts.get("sft_wins", 0)
alpaca_ties = alpaca_counts.get("ties", 0)

categories = ["DPO wins", "SFT wins", "Ties"]

hh_values = [hh_dpo, hh_sft, hh_ties]
alpaca_values = [alpaca_dpo, alpaca_sft, alpaca_ties]

x = [0, 1, 2]
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar([i - width/2 for i in x], hh_values, width=width, label="HH-RLHF")
plt.bar([i + width/2 for i in x], alpaca_values, width=width, label="AlpacaEval")

plt.xticks(x, categories)
plt.ylabel("Number of prompts")
plt.title("Task 7: Gemini Pairwise Judge Results")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("plots/task7_judge_results.png", dpi=200)
plt.close()

print("Saved plots/task7_judge_results.png")