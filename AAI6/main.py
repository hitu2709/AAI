import time

from llama_index.llms.groq import Groq
from llama_index.core import Settings

from src.config import GROQ_API_KEY, MODEL_NAME
from src.ingestion import create_index
from src.traditional_rag import traditional_rag
from src.agentic_rag import AgenticRAG
from src.evaluator import save_observation


def main():

    print("=" * 70)
    print("AGENTIC RAG ACADEMIC ASSISTANT")
    print("=" * 70)

    # -----------------------------------------
    # Configure LLM
    # -----------------------------------------

    Settings.llm = Groq(
        model=MODEL_NAME,
        api_key=GROQ_API_KEY
    )

    # -----------------------------------------
    # Create vector index
    # -----------------------------------------

    print("\nLoading documents...")

    index = create_index("Datasets")

    print("Vector index created successfully.")

    # -----------------------------------------
    # Create Agentic RAG
    # -----------------------------------------

    agent = AgenticRAG(index)

    observation_results = []

    # -----------------------------------------
    # User interaction
    # -----------------------------------------

    while True:

        print("\n")

        query = input(
            "Enter your question (or type exit): "
        )

        if query.lower() == "exit":

            break

        # Start timer
        start_time = time.time()

        # -------------------------------------
        # Traditional RAG
        # -------------------------------------

        traditional_answer, traditional_nodes = (
            traditional_rag(
                index,
                query
            )
        )

        # -------------------------------------
        # Agentic RAG
        # -------------------------------------

        result = agent.query(query)

        # End timer
        end_time = time.time()

        latency = round(
            end_time - start_time,
            2
        )

        # -------------------------------------
        # Determine evidence
        # -------------------------------------

        relevant_evidence = (
            "Yes"
            if len(result["evidence"]) > 0
            else "No"
        )

        # -------------------------------------
        # Store observation
        # -------------------------------------

        observation_results.append({

            "Query": query,

            "Traditional RAG Result":
                traditional_answer.replace(
                    "\n",
                    " "
                ),

            "Agentic RAG Result":
                result["answer"].replace(
                    "\n",
                    " "
                ),

            "Retrieval Attempts":
                result["attempts"],

            "Relevant Evidence":
                relevant_evidence,

            "Grounded":
                "Yes"
                if result["grounded"]
                else "No",

            "Latency":
                f"{latency}s"
        })

        # -------------------------------------
        # Display results
        # -------------------------------------

        print("\n" + "=" * 70)
        print("TRADITIONAL RAG")
        print("=" * 70)

        print(traditional_answer)

        print("\n" + "=" * 70)
        print("AGENTIC RAG")
        print("=" * 70)

        print(result["answer"])

        print(
            "\nRetrieval Attempts:",
            result["attempts"]
        )

        print(
            "Grounded:",
            result["grounded"]
        )

        print(
            "Abstained:",
            result["abstained"]
        )

        print(
            "Latency:",
            latency,
            "seconds"
        )

        print("=" * 70)

    # -----------------------------------------
    # Save observation table
    # -----------------------------------------

    if observation_results:

        save_observation(
            observation_results
        )


if __name__ == "__main__":
    main()