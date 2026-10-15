
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks import BaseCallbackHandler

from tools import TOOLS
from prompts import PROMPT_V1, PROMPT_V2
from evaluation import save_rows, summarize, save_comparison

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY is missing from your .env file.")


TEST_CASES = [
    {"id": "T1", "query": "Calculate 12 + 8.",
     "expected_tool": "calculator", "expected_answer": "20"},
    {"id": "T2", "query": "What is 144 divided by 12?",
     "expected_tool": "calculator", "expected_answer": "12"},
    {"id": "T3", "query": "Calculate 15 multiplied by 7.",
     "expected_tool": "calculator", "expected_answer": "105"},
    {"id": "T4", "query": "What is 100 minus 37?",
     "expected_tool": "calculator", "expected_answer": "63"},
    {"id": "T5", "query": "Calculate (8 + 4) * 3.",
     "expected_tool": "calculator", "expected_answer": "36"},
    {"id": "T6", "query": "Give me the details for student S069.",
     "expected_tool": "student_info", "expected_answer": "Hitanshu Shah"},
    {"id": "T7", "query": "Which department does student S055 belong to?",
     "expected_tool": "student_info", "expected_answer": "Computer Engineering"},
    {"id": "T8", "query": "What is the SAP ID of student S069?",
     "expected_tool": "student_info", "expected_answer": "60018240010"},
    {"id": "T9", "query": "Calculate 10 / 0 and explain why it is invalid.",
     "expected_tool": "calculator", "expected_answer": "division by zero"},
    {"id": "T10", "query": "Hello! How are you?",
     "expected_tool": "none", "expected_answer": "hello"},
]


class TokenUsageCallback(BaseCallbackHandler):
    """Collect token usage reported by LangChain model responses."""

    def __init__(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls_with_usage = 0

    def on_llm_end(self, response, **kwargs):
        for group in response.generations:
            for generation in group:
                message = getattr(generation, "message", None)
                if message is None:
                    continue

                # LangChain-standard usage metadata
                metadata = getattr(message, "usage_metadata", None)

                if metadata:
                    input_tokens = int(metadata.get("input_tokens", 0) or 0)
                    output_tokens = int(metadata.get("output_tokens", 0) or 0)
                    total = int(
                        metadata.get(
                            "total_tokens",
                            input_tokens + output_tokens,
                        ) or 0
                    )

                    self.prompt_tokens += input_tokens
                    self.completion_tokens += output_tokens
                    self.total_tokens += total
                    self.calls_with_usage += 1
                    continue

                # Provider-specific usage in additional keyword arguments
                additional = getattr(message, "additional_kwargs", {}) or {}
                usage = additional.get("usage") or {}

                if usage:
                    input_tokens = int(
                        usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
                    )
                    output_tokens = int(
                        usage.get(
                            "completion_tokens",
                            usage.get("output_tokens", 0),
                        ) or 0
                    )
                    total = int(
                        usage.get(
                            "total_tokens",
                            input_tokens + output_tokens,
                        ) or 0
                    )

                    self.prompt_tokens += input_tokens
                    self.completion_tokens += output_tokens
                    self.total_tokens += total
                    self.calls_with_usage += 1

        # Fallback to response-level metadata when available
        if self.calls_with_usage == 0:
            llm_output = response.llm_output or {}
            usage = (
                llm_output.get("token_usage")
                or llm_output.get("usage")
                or llm_output.get("usage_metadata")
                or {}
            )

            if usage:
                input_tokens = int(
                    usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
                )
                output_tokens = int(
                    usage.get(
                        "completion_tokens",
                        usage.get("output_tokens", 0),
                    ) or 0
                )
                total = int(
                    usage.get(
                        "total_tokens",
                        input_tokens + output_tokens,
                    ) or 0
                )

                self.prompt_tokens += input_tokens
                self.completion_tokens += output_tokens
                self.total_tokens += total
                self.calls_with_usage += 1


def build_agent(system_prompt):
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        max_retries=2,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, TOOLS, prompt)

    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=False,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        max_iterations=4,
    )


def check_answer(case, output):
    text = output.casefold()
    expected = case["expected_answer"].casefold()

    if case["id"] == "T9":
        return (
            "division by zero" in text
            or "dividing by zero" in text
            or ("zero" in text and "error" in text)
        )

    if case["id"] == "T10":
        return any(word in text for word in ["hello", "hi", "hey"])

    return expected in text


def run_benchmark(system_prompt, version):
    executor = build_agent(system_prompt)
    rows = []

    print(f"\n========== PROMPT {version} ==========")

    for case in TEST_CASES:
        start = time.perf_counter()
        token_counter = TokenUsageCallback()

        try:
            result = executor.invoke(
                {"input": case["query"]},
                config={"callbacks": [token_counter]},
            )

            latency = time.perf_counter() - start
            output = str(result.get("output", ""))

            steps = result.get("intermediate_steps", [])
            called_tools = [action.tool for action, _ in steps]
            actual_tool = called_tools[0] if called_tools else "none"

            tool_correct = actual_tool == case["expected_tool"]
            answer_correct = check_answer(case, output)
            task_success = tool_correct and answer_correct

            tokens = (
                token_counter.total_tokens
                if token_counter.calls_with_usage > 0
                else ""
            )

            row = {
                "test_id": case["id"],
                "prompt_version": version,
                "query": case["query"],
                "expected_tool": case["expected_tool"],
                "actual_tool": actual_tool,
                "all_tools_called": (
                    ";".join(called_tools) if called_tools else "none"
                ),
                "tool_call_correct": int(tool_correct),
                "answer_check_passed": int(answer_correct),
                "task_success": int(task_success),
                "latency_seconds": round(latency, 3),
                "tokens": tokens,
                "prompt_tokens": (
                    token_counter.prompt_tokens
                    if token_counter.calls_with_usage > 0 else ""
                ),
                "completion_tokens": (
                    token_counter.completion_tokens
                    if token_counter.calls_with_usage > 0 else ""
                ),
                "output": output.replace("\n", " "),
            }

        except Exception as error:
            latency = time.perf_counter() - start
            print(
                f"\nERROR in {case['id']}: "
                f"{type(error).__name__}: {error}"
            )

            row = {
                "test_id": case["id"],
                "prompt_version": version,
                "query": case["query"],
                "expected_tool": case["expected_tool"],
                "actual_tool": "error",
                "all_tools_called": "error",
                "tool_call_correct": 0,
                "answer_check_passed": 0,
                "task_success": 0,
                "latency_seconds": round(latency, 3),
                "tokens": "",
                "prompt_tokens": "",
                "completion_tokens": "",
                "output": f"ERROR: {type(error).__name__}: {error}",
            }

        rows.append(row)

        print(
            f"{case['id']} | Tool: {row['actual_tool']} | "
            f"Success: {row['task_success']} | "
            f"Latency: {row['latency_seconds']}s | "
            f"Tokens: {row['tokens'] if row['tokens'] != '' else 'Unavailable'}"
        )

    return rows


def main():
    rows_v1 = run_benchmark(PROMPT_V1, "V1")
    rows_v2 = run_benchmark(PROMPT_V2, "V2")

    save_rows(rows_v1, "prompt_v1_results.csv")
    save_rows(rows_v2, "prompt_v2_results.csv")

    summary_v1 = summarize(rows_v1, "V1")
    summary_v2 = summarize(rows_v2, "V2")

    save_comparison(summary_v1, summary_v2)

    print("\n========== FINAL SUMMARY ==========")
    print("Prompt V1:", summary_v1)
    print("Prompt V2:", summary_v2)

    print("\nFor token usage, inspect LangSmith traces if metadata is unavailable.")
    print("LangSmith project: Experiment-8-Tool-Agent")


if __name__ == "__main__":
    main()
