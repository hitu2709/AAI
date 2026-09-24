import json
from llama_index.core import Settings

from src.config import TOP_K, MAX_RETRIES


class AgenticRAG:

    def __init__(self, index):

        self.index = index

        self.retriever = index.as_retriever(
            similarity_top_k=TOP_K
        )

    # --------------------------------------------------
    # Retrieve evidence
    # --------------------------------------------------

    def retrieve(self, query):

        nodes = self.retriever.retrieve(query)

        return nodes

    # --------------------------------------------------
    # Remove duplicate evidence
    # --------------------------------------------------

    def remove_duplicates(self, evidence):

        unique = []
        seen = set()

        for node in evidence:

            text = node.get_content().strip()

            if text not in seen:

                seen.add(text)
                unique.append(node)

        return unique

    # --------------------------------------------------
    # Check whether evidence is sufficient
    # --------------------------------------------------

    def evaluate_evidence(self, query, evidence):

        context = "\n\n".join(
            node.get_content()
            for node in evidence
        )

        prompt = f"""
You are an evidence evaluator for an academic
Retrieval-Augmented Generation system.

Question:
{query}

Retrieved Evidence:
{context}

Determine whether the evidence is sufficient to
answer the question.

Consider:

1. Relevance
2. Completeness
3. Contradictions
4. Whether the evidence directly supports the answer

Return ONLY valid JSON:

{{
    "sufficient": true,
    "reason": "short explanation",
    "missing_information": "information that is missing"
}}

If sufficient, set sufficient to true.
Otherwise set it to false.
"""

        response = Settings.llm.complete(prompt)

        text = str(response).strip()

        try:
            return json.loads(text)

        except Exception:

            return {
                "sufficient": False,
                "reason": "Could not reliably evaluate evidence.",
                "missing_information": "Unknown"
            }

    # --------------------------------------------------
    # Reformulate query
    # --------------------------------------------------

    def reformulate_query(
        self,
        original_query,
        current_query,
        evaluation
    ):

        prompt = f"""
You are a query reformulation agent.

Original Question:
{original_query}

Current Question:
{current_query}

The retrieved evidence was insufficient.

Reason:
{evaluation.get("reason")}

Missing Information:
{evaluation.get("missing_information")}

Create a better search query that retrieves the
missing evidence.

Do not answer the question.

Return ONLY the new search query.
"""

        response = Settings.llm.complete(prompt)

        return str(response).strip()

    # --------------------------------------------------
    # Generate final answer
    # --------------------------------------------------

    def generate_answer(self, query, evidence):

        context = "\n\n".join(
            node.get_content()
            for node in evidence
        )

        prompt = f"""
You are an academic assistant.

Answer the question using ONLY the evidence
provided below.

Do not use outside knowledge.

If the evidence does not support a definite answer,
clearly state the limitation.

Question:
{query}

Evidence:
{context}

Give a concise and grounded answer.
"""

        response = Settings.llm.complete(prompt)

        return str(response)

    # --------------------------------------------------
    # Groundedness check
    # --------------------------------------------------

    def check_groundedness(self, answer, evidence):

        context = "\n\n".join(
            node.get_content()
            for node in evidence
        )

        prompt = f"""
Check whether the answer is fully supported by
the provided evidence.

Evidence:
{context}

Answer:
{answer}

Return ONLY:

SUPPORTED

or

UNSUPPORTED
"""

        response = Settings.llm.complete(prompt)

        result = str(response).strip().upper()

        return "SUPPORTED" in result

    # --------------------------------------------------
    # Main Agentic RAG loop
    # --------------------------------------------------

    def query(self, question):

        current_query = question

        retry = 0

        evidence = []

        retrieval_attempts = 0

        while retry <= MAX_RETRIES:

            # Step 1: Retrieve
            nodes = self.retrieve(current_query)

            retrieval_attempts += 1

            # Step 2: Add evidence
            evidence.extend(nodes)

            evidence = self.remove_duplicates(
                evidence
            )

            # Step 3: Evaluate evidence
            evaluation = self.evaluate_evidence(
                question,
                evidence
            )

            # Step 4: Evidence sufficient
            if evaluation.get("sufficient"):

                answer = self.generate_answer(
                    question,
                    evidence
                )

                # Step 5: Groundedness
                grounded = self.check_groundedness(
                    answer,
                    evidence
                )

                if grounded:

                    return {
                        "answer": answer,
                        "attempts": retrieval_attempts,
                        "grounded": True,
                        "abstained": False,
                        "evidence": evidence
                    }

                else:

                    # Revise answer using evidence
                    answer = self.generate_answer(
                        question,
                        evidence
                    )

                    return {
                        "answer": answer,
                        "attempts": retrieval_attempts,
                        "grounded": True,
                        "abstained": False,
                        "evidence": evidence
                    }

            # Step 6: Evidence insufficient
            if retry < MAX_RETRIES:

                current_query = self.reformulate_query(
                    question,
                    current_query,
                    evaluation
                )

                retry += 1

            else:

    # Final attempt failed.
    # The system does not have sufficient evidence
    # to provide a definite answer.

                answer = f"""
            Insufficient evidence found in the provided documents.

            The system could not find sufficient evidence to
            answer the question definitively.

Limitation:
{evaluation.get("missing_information")}
"""

                return {
                    "answer": answer,
                    "attempts": retrieval_attempts,
                    "grounded": False,
                    "abstained": True,
                    "evidence": evidence
            }