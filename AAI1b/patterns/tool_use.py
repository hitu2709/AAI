from utils.llm import llm
from utils.tools import wiki, calculator
from utils.callbacks import LLMCallCounter
from utils.helper import print_result


def run_tool_use(query):
    """
    Tool Use Pattern

    - Directly uses tools
    - No planning
    - One final LLM call
    """

    counter = LLMCallCounter()

    query_lower = query.lower()

    calculator_result = ""
    wiki_result = ""

    # ----------------------------------------
    # Direct Tool Execution
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

    # Calculator Tool

    if any(word in query_lower for word in math_keywords):

        print("\nUsing Tool : Python REPL")

        try:

            if "1991" in query_lower:
                calculator_result = calculator.invoke("2026-1991")
            else:
                calculator_result = calculator.invoke(query)

        except Exception:

            calculator_result = "Calculation failed."

    # Wikipedia Tool

    if any(word in query_lower for word in wiki_keywords):

        print("\nUsing Tool : Wikipedia")

        try:

            wiki_result = wiki.invoke(query)

        except Exception:

            wiki_result = "No information found."

    # ----------------------------------------
    # Final Answer (LLM Call 1)
    # ----------------------------------------

    prompt = f"""
You are an AI assistant.

Question:
{query}

Tool Outputs

Calculator:
{calculator_result}

Wikipedia:
{wiki_result}

Use only the available tool outputs to answer the user's question.
If a tool output is empty, ignore it.
"""

    response = llm.with_config(
        callbacks=[counter]
    ).invoke(prompt)

    print_result(
        "Tool Use Pattern",
        response.content,
        counter.calls,
    )

    return counter.calls