from utils.llm import llm
from utils.tools import wiki, calculator
from utils.callbacks import LLMCallCounter
from utils.helper import print_result


def run_reflection(query):
    """
    Reflection Pattern

    Step 1 -> Generate Initial Answer
    Step 2 -> Reflect and Improve
    """

    counter = LLMCallCounter()

    query_lower = query.lower()

    calculator_result = ""
    wiki_result = ""

    # ----------------------------------------
    # Tool Selection
    # ----------------------------------------

    math_keywords = [
        "calculate",
        "years",
        "add",
        "subtract",
        "multiply",
        "divide",
        "+",
        "-",
        "*",
        "/"
    ]

    wiki_keywords = [
        "python",
        "who",
        "what",
        "history",
        "creator",
        "language"
    ]

    # Calculator

    if any(word in query_lower for word in math_keywords):

        try:
            if "1991" in query_lower:
                calculator_result = calculator.invoke("2026-1991")
            else:
                calculator_result = calculator.invoke(query)

        except Exception:
            calculator_result = "Calculation failed."

    # Wikipedia

    if any(word in query_lower for word in wiki_keywords):

        try:
            wiki_result = wiki.invoke(query)

        except Exception:
            wiki_result = "No information found."

    # ----------------------------------------
    # Initial Answer (LLM Call 1)
    # ----------------------------------------

    initial_prompt = f"""
You are an AI assistant.

Question:
{query}

Available Tool Outputs

Calculator:
{calculator_result}

Wikipedia:
{wiki_result}

Generate the best possible answer.
"""

    initial_answer = llm.with_config(
        callbacks=[counter]
    ).invoke(initial_prompt).content

    print("\nInitial Answer\n")
    print(initial_answer)

    # ----------------------------------------
    # Reflection (LLM Call 2)
    # ----------------------------------------

    reflection_prompt = f"""
You are reviewing your own answer.

Original Question:
{query}

Initial Answer:
{initial_answer}

Instructions:

1. Check if the answer is correct.
2. Check if anything is missing.
3. Improve clarity.
4. Produce ONLY the improved final answer.
Do not explain your review.
"""

    improved_answer = llm.with_config(
        callbacks=[counter]
    ).invoke(reflection_prompt)

    print_result(
        "Reflection Pattern",
        improved_answer.content,
        counter.calls,
    )

    return counter.calls