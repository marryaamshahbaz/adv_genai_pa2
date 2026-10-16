import json
from pathlib import Path

Path("outputs/task7").mkdir(exist_ok=True)

for name in ["hh", "alpaca"]:

    with open(f"outputs/task6/{name}_mapping.jsonl") as f:
        mappings = {
            row["id"]: row
            for line in f if line.strip()
            for row in [json.loads(line)]
        }

    with open(f"outputs/task7/{name}_judgments.jsonl") as f:
        judgments = [json.loads(line) for line in f if line.strip()]

    dpo_wins = 0
    sft_wins = 0
    ties = 0
    unknown = 0

    for row in judgments:
        winner = row["winner"]

        if winner == "TIE":
            ties += 1
        elif winner == "UNKNOWN":
            unknown += 1
        elif winner in ("A", "B"):
            key = "model_a" if winner == "A" else "model_b"
            model = mappings[row["id"]][key]

            if model == "dpo":
                dpo_wins += 1
            elif model == "sft":
                sft_wins += 1

    total = len(mappings)
    failures = total - len({row["id"] for row in judgments})

    summary = {
        "evaluation": name,
        "total_pairs": total,
        "counts": {
            "dpo_wins": dpo_wins,
            "sft_wins": sft_wins,
            "ties": ties,
            "unknown": unknown,
            "failures": failures
        }
    }

    with open(f"outputs/task7/{name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(name, summary["counts"])