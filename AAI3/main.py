import pandas as pd
import os

from src.deterministic_guardrail import deterministic_guardrail
from src.llm_guardrail import llm_guardrail
from src.evaluator import evaluate_results


def main():

    print("====================================")
    print(" MULTI-LAYER GUARDRAIL FRAMEWORK")
    print("====================================")

    # Create output directory
    os.makedirs("outputs", exist_ok=True)

    # Load dataset
    dataset_path = "data/guardrail_dataset.xlsx"

    print("\nLoading dataset...")

    df = pd.read_excel(dataset_path)

    print(f"Total Prompts: {len(df)}")

    predicted_labels = []
    guardrail_used = []

    # Process every prompt
    for index, row in df.iterrows():

        prompt = str(row["Prompt"])

        print(f"\nProcessing Prompt {index + 1}/{len(df)}")

        # First Layer:
        # Deterministic Guardrail
        deterministic_result = deterministic_guardrail(prompt)

        if deterministic_result == "UNSAFE":

            predicted_label = "UNSAFE"
            layer = "Deterministic Guardrail"

            print("Blocked by Deterministic Guardrail")

        else:

            # Second Layer:
            # LLM-based Guardrail
            predicted_label = llm_guardrail(prompt)

            layer = "LLM Guardrail"

            print(
                f"Processed by LLM Guardrail -> "
                f"{predicted_label}"
            )

        predicted_labels.append(predicted_label)

        guardrail_used.append(layer)

    # Add results to dataframe
    df["Predicted_Label"] = predicted_labels

    df["Guardrail_Layer"] = guardrail_used

    # Save predictions
    output_file = "outputs/predictions.csv"

    df.to_csv(
        output_file,
        index=False
    )

    print("\nPredictions saved successfully!")

    # Evaluation
    actual_labels = df["Expected_Label"].str.upper()

    predicted_labels = df["Predicted_Label"].str.upper()

    evaluate_results(
        actual_labels,
        predicted_labels
    )

    print("\n====================================")
    print(" EXPERIMENT COMPLETED SUCCESSFULLY")
    print("====================================")


if __name__ == "__main__":
    main()