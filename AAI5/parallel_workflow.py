from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from llm import llm


class ParallelState(TypedDict):
    topic: str
    introduction: str
    key_concepts: str
    applications: str
    advantages: str
    challenges: str
    recent_developments: str
    conclusion: str
    final_report: str


def introduction_node(state):
    topic = state["topic"]

    response = llm.invoke(
        f"Write a short introduction about {topic}."
    )

    return {
        "introduction": response.content
    }


def concepts_node(state):
    topic = state["topic"]

    response = llm.invoke(
        f"Explain the key concepts of {topic}."
    )

    return {
        "key_concepts": response.content
    }


def applications_node(state):
    topic = state["topic"]

    response = llm.invoke(
        f"Explain important applications of {topic}."
    )

    return {
        "applications": response.content
    }


def advantages_node(state):
    topic = state["topic"]

    response = llm.invoke(
        f"Explain the advantages of {topic}."
    )

    return {
        "advantages": response.content
    }


def challenges_node(state):
    topic = state["topic"]

    response = llm.invoke(
        f"Explain challenges and limitations of {topic}."
    )

    return {
        "challenges": response.content
    }


def recent_node(state):
    topic = state["topic"]

    response = llm.invoke(
        f"Describe recent developments in {topic}."
    )

    return {
        "recent_developments": response.content
    }


def conclusion_node(state):

    response = llm.invoke(
        f"""
Write a short conclusion for the research topic:

{state["topic"]}

The conclusion should summarize the main ideas, applications,
advantages, challenges, and recent developments.

Keep it within 150 words.
"""
    )

    final_report = f"""
RESEARCH REPORT: {state["topic"]}

1. INTRODUCTION
{state["introduction"]}

2. KEY CONCEPTS
{state["key_concepts"]}

3. APPLICATIONS
{state["applications"]}

4. ADVANTAGES
{state["advantages"]}

5. CHALLENGES AND LIMITATIONS
{state["challenges"]}

6. RECENT DEVELOPMENTS
{state["recent_developments"]}

7. CONCLUSION
{response.content}
"""

    return {
        "conclusion": response.content,
        "final_report": final_report
    }


def create_parallel_workflow():

    graph = StateGraph(ParallelState)

    graph.add_node("introduction", introduction_node)
    graph.add_node("concepts", concepts_node)
    graph.add_node("applications", applications_node)
    graph.add_node("advantages", advantages_node)
    graph.add_node("challenges", challenges_node)
    graph.add_node("recent", recent_node)
    graph.add_node("conclusion", conclusion_node)

    graph.add_edge(START, "introduction")
    graph.add_edge(START, "concepts")
    graph.add_edge(START, "applications")
    graph.add_edge(START, "advantages")
    graph.add_edge(START, "challenges")
    graph.add_edge(START, "recent")

    graph.add_edge("introduction", "conclusion")
    graph.add_edge("concepts", "conclusion")
    graph.add_edge("applications", "conclusion")
    graph.add_edge("advantages", "conclusion")
    graph.add_edge("challenges", "conclusion")
    graph.add_edge("recent", "conclusion")

    graph.add_edge("conclusion", END)

    return graph.compile()