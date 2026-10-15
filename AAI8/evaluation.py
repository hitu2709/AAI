
import csv
import os
from statistics import mean

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_rows(rows, filename):
    if not rows:
        return

    path = os.path.join(RESULTS_DIR, filename)

    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {path}")


def summarize(rows, version):
    total = len(rows)

    success_rate = (
        100 * sum(int(row["task_success"]) for row in rows) / total
        if total else 0
    )

    tool_accuracy = (
        100 * sum(int(row["tool_call_correct"]) for row in rows) / total
        if total else 0
    )

    avg_latency = (
        mean(float(row["latency_seconds"]) for row in rows)
        if total else 0
    )

    known_tokens = [
        int(row["tokens"])
        for row in rows
        if row.get("tokens") not in ("", None)
    ]

    total_tokens = sum(known_tokens) if known_tokens else "Unavailable"
    token_coverage = f"{len(known_tokens)}/{total}"

    failed_tasks = sum(
        1 for row in rows if not int(row["task_success"])
    )

    return {
        "prompt_version": version,
        "task_success_rate_percent": round(success_rate, 2),
        "tool_calling_accuracy_percent": round(tool_accuracy, 2),
        "average_latency_seconds": round(avg_latency, 3),
        "total_tokens": total_tokens,
        "token_measurements_available": token_coverage,
        "failed_tasks": failed_tasks,
    }


def save_comparison(v1, v2):
    metrics = [
        ("Task Success Rate (%)", "task_success_rate_percent"),
        ("Tool-Calling Accuracy (%)", "tool_calling_accuracy_percent"),
        ("Average Latency (s)", "average_latency_seconds"),
        ("Total Token Consumption", "total_tokens"),
        ("Token Measurements Available", "token_measurements_available"),
        ("Failed Tasks", "failed_tasks"),
    ]

    rows = [
        {
            "metric": label,
            "prompt_v1": v1[key],
            "prompt_v2": v2[key],
        }
        for label, key in metrics
    ]

    save_rows(rows, "comparison.csv")

    print("\nComparison")
    for row in rows:
        print(
            f"{row['metric']}: "
            f"V1={row['prompt_v1']} | V2={row['prompt_v2']}"
        )
