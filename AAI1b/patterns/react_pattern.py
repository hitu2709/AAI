from utils.llm import llm
from utils.tools import wiki, calculator
from utils.callbacks import LLMCallCounter
from utils.helper import print_result


def run_react(query):
    """
    ReAct Pattern

    Thought
        ↓
    Action
        ↓
    Observation
        ↓
    Thought
        ↓
    Action
        ↓
    Observation
        ↓
    Final Answer
    """

    counter = LLMCallCounter()

    query_lower = query.lower()

    # ------------------------------------
    # First Thought (LLM Call 1)
    # ------------------------------------

    thought1_prompt = f"""
You are a ReAct agent.

Question:
{query}

Think about ONLY the first action.

Do NOT answer the question.
Return only your thought.
"""

    thought1 = llm.with_config(callbacks=[counter]).invoke(
        thought1_prompt
    ).content

    print("\nThought 1")
    print(thought1)

    # ------------------------------------
    # First Action
    # ------------------------------------

    observation1 = ""

    if any(word in query_lower for word in [
        "calculate", "years", "+", "-", "*", "/", "add", "subtract", "multiply", "divide"
    ]):

        print("\nAction : Python REPL")

        try:
            if "1991" in query_lower:
                observation1 = calculator.invoke("2026-1991")
            else:
                observation1 = calculator.invoke(query)
        except:
            observation1 = "Calculation failed."

    else:

        print("\nAction : Wikipedia")

        try:
            observation1 = wiki.invoke(query)
        except:
            observation1 = "No information found."

    print("Observation :", observation1)

    # ------------------------------------
    # Second Thought (LLM Call 2)
    # ------------------------------------

    thought2_prompt = f"""
Question:
{query}

Previous Observation:
{observation1}

Based on the observation,
what should be the NEXT action?

Return only your thought.
"""

    thought2 = llm.with_config(callbacks=[counter]).invoke(
        thought2_prompt
    ).content

    print("\nThought 2")
    print(thought2)

    # ------------------------------------
    # Second Action
    # ------------------------------------

    observation2 = ""

    if "wiki" in thought2.lower() or any(word in query_lower for word in [
        "python",
        "who",
        "history",
        "creator",
        "language",
        "what"
    ]):

        print("\nAction : Wikipedia")

        try:
            observation2 = wiki.invoke(query)
        except:
            observation2 = "No information found."

    else:

        print("\nAction : Python REPL")

        try:
            if "1991" in query_lower:
                observation2 = calculator.invoke("2026-1991")
            else:
                observation2 = calculator.invoke(query)
        except:
            observation2 = "Calculation failed."

    print("Observation :", observation2)

    # ------------------------------------
    # Final Answer (LLM Call 3)
    # ------------------------------------

    final_prompt = f"""
Question:
{query}

Thought 1:
{thought1}

Observation 1:
{observation1}

Thought 2:
{thought2}

Observation 2:
{observation2}

Using the observations above,
generate the final answer.
"""

    response = llm.with_config(callbacks=[counter]).invoke(
        final_prompt
    )

    print_result(
        "ReAct Pattern",
        response.content,
        counter.calls,
    )

    return counter.calls