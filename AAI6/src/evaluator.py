import csv
import os


def save_observation(results, filename="results/observation.csv"):

    os.makedirs("results", exist_ok=True)

    fieldnames = [
        "Query",
        "Traditional RAG Result",
        "Agentic RAG Result",
        "Retrieval Attempts",
        "Relevant Evidence",
        "Grounded",
        "Latency"
    ]

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:
            writer.writerow(result)

    print(
        f"\nObservation table saved to: {filename}"
    )