from llama_index.core import Settings
from src.config import TOP_K


def traditional_rag(index, query):

    retriever = index.as_retriever(
        similarity_top_k=TOP_K
    )

    nodes = retriever.retrieve(query)

    context = "\n\n".join(
        node.get_content()
        for node in nodes
    )

    prompt = f"""
You are an academic assistant.

Answer the user's question using ONLY the
provided context.

If the context does not contain enough information,
say:

"Insufficient evidence found in the provided documents."

Context:
{context}

Question:
{query}

Answer:
"""

    response = Settings.llm.complete(prompt)

    return str(response), nodes