from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import matplotlib.pyplot as plt
import seaborn as sns
import os


def evaluate_results(actual, predicted):

    accuracy = accuracy_score(actual, predicted)

    precision = precision_score(
        actual,
        predicted,
        pos_label="UNSAFE",
        zero_division=0
    )

    recall = recall_score(
        actual,
        predicted,
        pos_label="UNSAFE",
        zero_division=0
    )

    f1 = f1_score(
        actual,
        predicted,
        pos_label="UNSAFE",
        zero_division=0
    )

    cm = confusion_matrix(
        actual,
        predicted,
        labels=["SAFE", "UNSAFE"]
    )

    print("\n========== EVALUATION RESULTS ==========")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    os.makedirs("outputs", exist_ok=True)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=["SAFE", "UNSAFE"],
        yticklabels=["SAFE", "UNSAFE"]
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("Confusion Matrix")

    plt.savefig("outputs/confusion_matrix.png")

    plt.show()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }