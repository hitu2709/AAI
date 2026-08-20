from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from llm import llm


class OrchestratorState(TypedDict):
    topic: str
    tasks: str
    worker_results: str
    final_report: str


# -------------------------------
# ORCHESTRATOR
# -------------------------------

def orchestrator_node(state):

    topic = state["topic"]

    response = llm.invoke(
        f"""
You are an orchestrator for a research report.

Topic: {topic}

Create a short plan for these sections:

1. Introduction
2. Key Concepts
3. Applications
4. Advantages
5. Challenges and Limitations
6. Recent Developments

Keep the plan under 100 words.
"""
    )

    return {
        "tasks": response.content
    }


# -------------------------------
# WORKERS
# -------------------------------

def worker_node(state):

    topic = state["topic"]

    sections = [
        "Introduction",
        "Key Concepts",
        "Applications",
        "Advantages",
        "Challenges and Limitations",
        "Recent Developments"
    ]

    results = []

    for section in sections:

        response = llm.invoke(
            f"""
You are a research worker.

Research Topic: {topic}

Write the section: {section}

Important:
- Keep it concise.
- Maximum 80 words.
- Use simple and clear language.
"""
        )

        results.append(
            f"\n\n## {section}\n{response.content}"
        )

    return {
        "worker_results": "\n".join(results)
    }


# -------------------------------
# SYNTHESIZER
# -------------------------------

def synthesizer_node(state):

    topic = state["topic"]

    # Generate ONLY the conclusion
    response = llm.invoke(
        f"""
Write a concise conclusion for the research topic:

{topic}

Summarize the importance, applications,
advantages, challenges, and future scope.

Keep it under 100 words.
"""
    )

    # Combine worker outputs directly
    final_report = f"""
RESEARCH REPORT: {topic}

{state["worker_results"]}

## Conclusion

{response.content}
"""

    return {
        "final_report": final_report
    }


# -------------------------------
# CREATE LANGGRAPH WORKFLOW
# -------------------------------

def create_orchestrator_workflow():

    graph = StateGraph(OrchestratorState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("worker", worker_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.add_edge(START, "orchestrator")

    graph.add_edge(
        "orchestrator",
        "worker"
    )

    graph.add_edge(
        "worker",
        "synthesizer"
    )

    graph.add_edge(
        "synthesizer",
        END
    )

    return graph.compile()