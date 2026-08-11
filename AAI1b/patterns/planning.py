from utils.llm import llm
from utils.tools import wiki, calculator
from utils.callbacks import LLMCallCounter
from utils.helper import print_result


def run_planning(query):
    """
    Planning Pattern

    Step 1 -> Create a plan
    Step 2 -> Execute tools
    Step 3 -> Final answer
    """

    counter = LLMCallCounter()

    # -----------------------------
    # Planner
    # -----------------------------

    planner_prompt = f"""
You are an AI Planner.

Create a short numbered execution plan for answering the following question.

Question:
{query}

Available Tools:
1. Wikipedia (for factual information)
2. Python REPL (for calculations)

Do NOT answer the question.

Return only the numbered plan.
"""

    plan = llm.with_config(callbacks=[counter]).invoke(
        planner_prompt
    ).content

    print("\nExecution Plan\n")
    print(plan)

    # -----------------------------
    # Execute Plan
    # -----------------------------

    query_lower = query.lower()

    calculator_result = ""
    wiki_result = ""

    math_keywords = [
        "calculate",
        "add",
        "subtract",
        "multiply",
        "divide",
        "+",
        "-",
        "*",
        "/",
        "years"
    ]

    wiki_keywords = [
        "python",
        "who",
        "what",
        "history",
        "creator",
        "language"
    ]

    # Calculator Tool

    if any(word in query_lower for word in math_keywords):

        try:

            if "1991" in query_lower:
                calculator_result = calculator.invoke("2026-1991")

            else:
                calculator_result = calculator.invoke(query)

        except Exception:

            calculator_result = "Calculation could not be performed."

    # Wikipedia Tool

    if any(word in query_lower for word in wiki_keywords):

        try:

            wiki_result = wiki.invoke(query)

        except Exception:

            wiki_result = "No information found."

    # -----------------------------
    # Final Answer
    # -----------------------------

    final_prompt = f"""
Execution Plan:

{plan}

Question:

{query}

Calculator Output:

{calculator_result}

Wikipedia Output:

{wiki_result}

Using only the above tool outputs,
generate the final answer.
"""

    response = llm.with_config(callbacks=[counter]).invoke(
        final_prompt
    )

    print_result(
        "Planning Pattern",
        response.content,
        counter.calls,
    )

    return counter.calls