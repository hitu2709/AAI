import os
import pandas as pd
import dspy

from dspy_optimizer.signature import CustomerSupport
from dspy_optimizer.train_examples import trainset
from dspy_optimizer.metric import metric

from config.config import GROQ_API_KEY

# Import evaluation functions
from evaluation.rouge_eval import rouge
from evaluation.bertscore_eval import bert
from evaluation.cosine_eval import cosine
from evaluation.llm_judge import judge

# Configure DSPy
lm = dspy.LM(
    "groq/llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY
)

dspy.settings.configure(lm=lm)

# DSPy Program
program = dspy.Predict(CustomerSupport)

optimizer = dspy.BootstrapFewShot(metric=metric)

optimized = optimizer.compile(
    program,
    trainset=trainset
)

# Generate Optimized Response
result = optimized(
    query="I received the wrong product yesterday and I want a replacement."
)

print("\nOptimized Response:\n")
print(result.answer)

# Create outputs folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Save optimized response
optimized_df = pd.DataFrame({
    "Prompt": ["DSPy Optimized"],
    "Response": [result.answer]
})

optimized_df.to_csv("outputs/optimized_response.csv", index=False)

# Evaluate optimized response
rouge_score = rouge(result.answer)
bert_score = bert(result.answer)
cosine_score = cosine(result.answer)
judge_score = judge(result.answer)

comparison = pd.DataFrame({
    "Prompt": ["DSPy Optimized"],
    "Response": [result.answer],
    "ROUGE": [rouge_score],
    "BERTScore": [bert_score],
    "Cosine": [cosine_score],
    "Judge": [judge_score]
})

comparison.to_csv("outputs/optimized_evaluation.csv", index=False)

# Append to evaluation.csv
evaluation_path = "outputs/evaluation.csv"

if os.path.exists(evaluation_path):
    old_df = pd.read_csv(evaluation_path)
    final_df = pd.concat([old_df, comparison], ignore_index=True)
else:
    final_df = comparison

final_df.to_csv("outputs/final_comparison.csv", index=False)

print("\nComparison saved successfully!")

print("\nFinal Comparison:\n")
print(final_df)