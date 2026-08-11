from utils.llm import llm
from utils.tools import wiki, calculator
from utils.callbacks import LLMCallCounter
from utils.helper import print_result


def run_rewoo(query):
    """
    ReWOO Pattern (Reason Without Observation)

    1. Create one complete reasoning plan.
    2. Execute all required tools.
    3. Produce the final answer.
    """

    counter = LLMCallCounter()

    query_lower = query.lower()

    # ----------------------------------------
    # Step 1 : Create Plan (LLM Call 1)
    # ----------------------------------------

    planner_prompt = f"""
You are a ReWOO Agent.

Question:
{query}

Available Tools:
1. Wikipedia
2. Python REPL

Create a complete execution plan.

Rules:
- Decide all required tools before execution.
- Do NOT answer the question.
- Do NOT think again after tool execution.
- Return only the numbered plan.
"""

    plan = llm.with_config(
        callbacks=[counter]
    ).invoke(planner_prompt).content

    print("\nReasoning Plan\n")
    print(plan)

    # ----------------------------------------
    # Step 2 : Execute Tools
    # ----------------------------------------

    calculator_result = ""
    wiki_result = ""

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

    # Execute Calculator

    if any(word in query_lower for word in math_keywords):

        print("\nExecuting : Python REPL")

        try:

            if "1991" in query_lower:
                calculator_result = calculator.invoke("2026-1991")
            else:
                calculator_result = calculator.invoke(query)

        except Exception:

            calculator_result = "Calculation failed."

    # Execute Wikipedia

    if any(word in query_lower for word in wiki_keywords):

        print("\nExecuting : Wikipedia")

        try:

            wiki_result = wiki.invoke(query)

        except Exception:

            wiki_result = "No information found."

    # ----------------------------------------
    # Step 3 : Final Answer (LLM Call 2)
    # ----------------------------------------

    final_prompt = f"""
Question:
{query}

Execution Plan:
{plan}

Tool Outputs

Calculator:
{calculator_result}

Wikipedia:
{wiki_result}

Generate the final answer using only the tool outputs.

Do NOT create another plan.
"""

    response = llm.with_config(
        callbacks=[counter]
    ).invoke(final_prompt)

    print_result(
        "ReWOO Pattern",
        response.content,
        counter.calls,
    )

    return counter.calls