from patterns.tool_use import run_tool_use
from patterns.planning import run_planning
from patterns.plan_execute import run_plan_execute
from patterns.react_pattern import run_react
from patterns.reflection import run_reflection
from patterns.rewoo import run_rewoo


def print_header(title):
    print("\n" + "=" * 70)
    print(title.center(70))
    print("=" * 70)


def get_prompt():

    print("=" * 70)
    print("AGENTIC AI DESIGN PATTERNS".center(70))
    print("=" * 70)

    print("\nExample Questions:")
    print("1. Python was first released in 1991. How many years has it been since then?")
    print("2. Who created Python?")
    print("3. What is 250 + 450?")
    print("4. What is Machine Learning?")
    print("5. Calculate 45 * 67 and explain what Python is.\n")

    prompt = input("Enter your question:\n> ")

    return prompt


def main():

    prompt = get_prompt()

    results = {}

    print_header("TOOL USE PATTERN")
    calls = run_tool_use(prompt)
    results["Tool Use"] = calls

    print_header("PLANNING PATTERN")
    calls = run_planning(prompt)
    results["Planning"] = calls

    print_header("PLAN AND EXECUTE PATTERN")
    calls = run_plan_execute(prompt)
    results["Plan & Execute"] = calls

    print_header("REACT PATTERN")
    calls = run_react(prompt)
    results["ReAct"] = calls

    print_header("REFLECTION PATTERN")
    calls = run_reflection(prompt)
    results["Reflection"] = calls

    print_header("REWOO PATTERN")
    calls = run_rewoo(prompt)
    results["ReWOO"] = calls

    print("\n")
    print("=" * 70)
    print("LLM CALL COMPARISON".center(70))
    print("=" * 70)

    print(f"{'Pattern':25} {'LLM Calls'}")
    print("-" * 40)

    for pattern, calls in results.items():
        print(f"{pattern:25} {calls}")

    print("\n")

    best = min(results, key=results.get)

    print("=" * 70)
    print("ANALYSIS".center(70))
    print("=" * 70)

    print(f"Best Pattern for this prompt : {best}\n")

    print("Reason:")

    if best == "Tool Use":
        print("- Best for simple questions requiring one or two tools.")
        print("- Fastest execution with minimum LLM calls.")

    elif best == "Planning":
        print("- Good for multi-step reasoning.")
        print("- Creates a structured execution plan.")

    elif best == "Plan & Execute":
        print("- Separates planning from execution.")
        print("- Suitable for long workflows.")

    elif best == "ReAct":
        print("- Dynamically decides the next action based on observations.")
        print("- Best for uncertain or changing problems.")

    elif best == "Reflection":
        print("- Reviews and improves its own answer.")
        print("- Better quality at the cost of more LLM calls.")

    elif best == "ReWOO":
        print("- Plans once and executes efficiently.")
        print("- Avoids repeated reasoning.")


if __name__ == "__main__":
    main()