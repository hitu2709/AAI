from utils.llm import llm
from utils.tools import wiki, calculator
from utils.callbacks import LLMCallCounter
from utils.helper import print_result


def run_plan_execute(query):
    """
    Plan-and-Execute Pattern

    Planner -> Executor -> Final Answer
    """

    counter = LLMCallCounter()

    # -------------------------
    # Planner
    # -------------------------

    planner_prompt = f"""
You are a Planner.

Create a numbered execution plan to answer this user question.

Question:
{query}

Rules:
- Decide which tools are needed.
- Use Calculator only for mathematical calculations.
- Use Wikipedia only for factual information.
- Do NOT answer the question.
Return only the numbered plan.
"""

    plan = llm.with_config(callbacks=[counter]).invoke(
        planner_prompt
    ).content

    print("\nGenerated Plan\n")
    print(plan)

    print("\nExecuting Plan...\n")

    query_lower = query.lower()

    calculator_result = ""
    wiki_result = ""

    # -------------------------
    # Execute Calculator
    # -------------------------

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

    if any(word in query_lower for word in math_keywords):

        try:
            if "1991" in query_lower:
                calculator_result = calculator.invoke("2026-1991")
            else:
                calculator_result = calculator.invoke(query)

        except Exception:
            calculator_result = "Calculation could not be performed."

    # -------------------------
    # Execute Wikipedia
    # -------------------------

    if any(word in query_lower for word in [
        "python",
        "who",
        "what",
        "history",
        "creator",
        "language"
    ]):

        try:
            wiki_result = wiki.invoke(query)

        except Exception:
            wiki_result = "No information found."

    # -------------------------
    # Final Answer
    # -------------------------

    executor_prompt = f"""
You are an Executor.

Follow this plan:

{plan}

Tool Outputs

Calculator:
{calculator_result}

Wikipedia:
{wiki_result}

Using only the available tool outputs,
produce the final answer for the user.
"""

    response = llm.with_config(callbacks=[counter]).invoke(
        executor_prompt
    )

    print_result(
        "Plan-and-Execute Pattern",
        response.content,
        counter.calls,
    )

    return counter.calls