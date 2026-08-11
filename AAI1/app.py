from agents import decide
from tools import search_wikipedia, calculator

memory = []

print("===== Agentic AI =====")
print("Type 'exit' to quit.\n")

while True:

    question = input("You: ")

    if question.lower() == "exit":
        break

    # Store user message
    memory.append(
        {
            "role": "user",
            "content": question
        }
    )

    # Ask Groq
    decision = decide(memory)

    print("\nDecision:")
    print(decision)

    # ---------------- Wikipedia ----------------

    if "wikipedia" in decision.lower():

        query = decision.split("INPUT:")[1].strip()

        answer = search_wikipedia(query)

        print("\nAnswer:\n")
        print(answer)

        memory.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

    # ---------------- Calculator ----------------

    elif "calculator" in decision.lower():

        expression = decision.split("INPUT:")[1].strip()

        answer = calculator(expression)

        print("\nAnswer:\n")
        print(answer)

        memory.append(
            {
                "role": "assistant",
                "content": str(answer)
            }
        )

    # ---------------- Normal Chat ----------------

    else:

        answer = decision.split("INPUT:")[1].strip()

        print("\nAnswer:\n")
        print(answer)

        memory.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

    # ----------- Show Memory -----------

    print("\n================ Memory ================")

    for msg in memory:
        print(f"{msg['role']} : {msg['content']}")

    print("========================================\n")