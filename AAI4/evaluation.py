import asyncio
import csv
import os
import time

import matplotlib.pyplot as plt

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient


# ==========================================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================================

load_dotenv()


# ==========================================================
# CONFIGURATION
# ==========================================================

RESULTS_DIR = "results"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ==========================================================
# GLOBAL MCP TIMING STORAGE
# ==========================================================

mcp_timings = []


# ==========================================================
# 20 TEST QUERIES
# ==========================================================

TEST_QUERIES = [

    # ======================================================
    # CALCULATOR QUERIES
    # ======================================================

    {
        "id": 1,
        "query": "What is 25 + 75?",
        "category": "Calculator",
        "expected_tools": ["calculator"]
    },

    {
        "id": 2,
        "query": "Calculate 125 * 8",
        "category": "Calculator",
        "expected_tools": ["calculator"]
    },

    {
        "id": 3,
        "query": "What is 1000 / 25?",
        "category": "Calculator",
        "expected_tools": ["calculator"]
    },

    {
        "id": 4,
        "query": "Calculate (45 + 15) * 2",
        "category": "Calculator",
        "expected_tools": ["calculator"]
    },

    {
        "id": 5,
        "query": "What is 2^10?",
        "category": "Calculator",
        "expected_tools": ["calculator"]
    },


    # ======================================================
    # WIKIPEDIA QUERIES
    # ======================================================

    {
        "id": 6,
        "query": "Who was Albert Einstein and what were his major contributions?",
        "category": "Wikipedia",
        "expected_tools": ["wikipedia_search"]
    },

    {
        "id": 7,
        "query": "What is the history of the Internet?",
        "category": "Wikipedia",
        "expected_tools": ["wikipedia_search"]
    },

    {
        "id": 8,
        "query": "Tell me about the history and creator of the Python programming language.",
        "category": "Wikipedia",
        "expected_tools": ["wikipedia_search"]
    },

    {
        "id": 9,
        "query": "Who was Alan Turing and what was his contribution to computer science?",
        "category": "Wikipedia",
        "expected_tools": ["wikipedia_search"]
    },

    {
        "id": 10,
        "query": "What is the history and origin of artificial intelligence?",
        "category": "Wikipedia",
        "expected_tools": ["wikipedia_search"]
    },


    # ======================================================
    # NO TOOL QUERIES
    # ======================================================

    {
        "id": 11,
        "query": "What is the capital of India?",
        "category": "No Tool",
        "expected_tools": []
    },

    {
        "id": 12,
        "query": "Explain what machine learning means.",
        "category": "No Tool",
        "expected_tools": []
    },

    {
        "id": 13,
        "query": "What are the advantages of cloud computing?",
        "category": "No Tool",
        "expected_tools": []
    },

    {
        "id": 14,
        "query": "What is the difference between AI and ML?",
        "category": "No Tool",
        "expected_tools": []
    },

    {
        "id": 15,
        "query": "Give me three uses of databases.",
        "category": "No Tool",
        "expected_tools": []
    },


    # ======================================================
    # MIXED / AMBIGUOUS QUERIES
    # ======================================================

    {
        "id": 16,
        "query": "Calculate 50 * 20 and tell me about Alan Turing.",
        "category": "Mixed",
        "expected_tools": [
            "calculator",
            "wikipedia_search"
        ]
    },

    {
        "id": 17,
        "query": "What is 100 / 4 and who was Albert Einstein?",
        "category": "Mixed",
        "expected_tools": [
            "calculator",
            "wikipedia_search"
        ]
    },

    {
        "id": 18,
        "query": "Is Python a programming language and when was it created?",
        "category": "Ambiguous",
        "expected_tools": [
            "wikipedia_search"
        ]
    },

    {
        "id": 19,
        "query": "Calculate the square of 25 and explain artificial intelligence.",
        "category": "Mixed",
        "expected_tools": [
            "calculator"
        ]
    },

    {
        "id": 20,
        "query": "Tell me about the Internet and calculate 250 + 750.",
        "category": "Mixed",
        "expected_tools": [
            "calculator",
            "wikipedia_search"
        ]
    }
]


# ==========================================================
# EXTRACT TOOL CALLS
# ==========================================================

def extract_tool_calls(messages):

    selected_tools = []

    for message in messages:

        tool_calls = getattr(
            message,
            "tool_calls",
            []
        )

        if tool_calls:

            for call in tool_calls:

                tool_name = call.get(
                    "name"
                )

                if tool_name:

                    selected_tools.append(
                        tool_name
                    )

    # Remove duplicates

    return list(
        dict.fromkeys(
            selected_tools
        )
    )


# ==========================================================
# EXTRACT TOKEN USAGE
# ==========================================================

def extract_token_usage(messages):

    input_tokens = 0
    output_tokens = 0
    total_tokens = 0

    for message in messages:

        usage = getattr(
            message,
            "usage_metadata",
            None
        )

        if usage:

            input_tokens += usage.get(
                "input_tokens",
                0
            )

            output_tokens += usage.get(
                "output_tokens",
                0
            )

            total_tokens += usage.get(
                "total_tokens",
                0
            )

    return (
        input_tokens,
        output_tokens,
        total_tokens
    )


# ==========================================================
# CHECK TOOL SELECTION ACCURACY
# ==========================================================

def check_tool_accuracy(
    expected_tools,
    selected_tools
):

    return (
        set(expected_tools)
        ==
        set(selected_tools)
    )


# ==========================================================
# RATE LIMIT RETRY FUNCTION
# ==========================================================

async def invoke_with_retry(
    agent,
    input_data,
    max_retries=3
):

    for attempt in range(max_retries):

        try:

            response = await agent.ainvoke(
                input_data
            )

            return response

        except Exception as e:

            error_message = str(e)

            # Check for rate limit error

            if (
                "429" in error_message
                or
                "rate_limit" in error_message.lower()
            ):

                wait_time = 5 * (
                    attempt + 1
                )

                print(
                    f"\nRate limit reached."
                )

                print(
                    f"Waiting {wait_time} seconds before retry..."
                )

                await asyncio.sleep(
                    wait_time
                )

            else:

                raise e


    raise Exception(
        "Maximum retry attempts reached due to rate limit."
    )


# ==========================================================
# RUN EVALUATION
# ==========================================================

async def run_evaluation():

    global mcp_timings

    mcp_timings = []


    # ======================================================
    # CONNECT TO MCP SERVER
    # ======================================================

    print(
        "\nConnecting to MCP Server..."
    )


    client = MultiServerMCPClient(

        {
            "calculator_wikipedia": {

                "transport":
                    "streamable_http",

                "url":
                    "http://127.0.0.1:8000/mcp"
            }
        }

    )


    # ======================================================
    # GET MCP TOOLS
    # ======================================================

    original_tools = await client.get_tools()


    print(
        "\nDiscovered MCP Tools:"
    )


    for tool_item in original_tools:

        print(
            "-",
            tool_item.name
        )


    # ======================================================
    # CREATE TOOL MAP
    # ======================================================

    tool_map = {}


    for tool_item in original_tools:

        tool_map[
            tool_item.name
        ] = tool_item


    # ======================================================
    # WRAPPED CALCULATOR TOOL
    # ======================================================

    @tool("calculator")
    async def calculator(
        expression: str
    ) -> str:

        """
        MANDATORY TOOL for any mathematical calculation.

        Use this tool for:
        addition,
        subtraction,
        multiplication,
        division,
        powers,
        percentages,
        square roots,
        and all numerical expressions.

        NEVER solve a mathematical expression directly.
        Always use this tool.
        """

        start = time.perf_counter()


        try:

            result = await tool_map[
                "calculator"
            ].ainvoke(

                {
                    "expression":
                        expression
                }

            )


            return str(
                result
            )


        finally:

            elapsed = (
                time.perf_counter()
                -
                start
            )


            mcp_timings.append(

                {
                    "tool":
                        "calculator",

                    "time":
                        elapsed
                }

            )


    # ======================================================
    # WRAPPED WIKIPEDIA TOOL
    # ======================================================

    @tool("wikipedia_search")
    async def wikipedia_search(
        query: str
    ) -> str:

        """
        MANDATORY TOOL for factual knowledge retrieval.

        Use this tool for:

        - People
        - Biographies
        - Historical events
        - History
        - Inventions
        - Technologies
        - Programming languages
        - Origins
        - Creation dates
        - Discoveries
        - Detailed factual background

        Examples:

        Albert Einstein
        Alan Turing
        History of the Internet
        Python programming language
        When Python was created

        Do not answer these types of questions directly.
        Search Wikipedia first.
        """

        start = time.perf_counter()


        try:

            result = await tool_map[
                "wikipedia_search"
            ].ainvoke(

                {
                    "query":
                        query
                }

            )


            return str(
                result
            )


        finally:

            elapsed = (
                time.perf_counter()
                -
                start
            )


            mcp_timings.append(

                {
                    "tool":
                        "wikipedia_search",

                    "time":
                        elapsed
                }

            )


    # ======================================================
    # AVAILABLE TOOLS
    # ======================================================

    tools = [

        calculator,

        wikipedia_search

    ]


    # ======================================================
    # GROQ MODEL
    # ======================================================

    llm = ChatGroq(

        model=
            "openai/gpt-oss-20b",

        temperature=0,

        max_tokens=250
    )


    # ======================================================
    # STRICT SYSTEM PROMPT
    # ======================================================

    system_prompt = """

You are an MCP-enabled chatbot being evaluated for correct tool selection.

Your tool selection is extremely important.

You MUST follow all rules below strictly.

==================================================

CALCULATOR RULE

If the user asks ANY mathematical or numerical calculation,
you MUST call the calculator tool.

This includes:

- Addition
- Subtraction
- Multiplication
- Division
- Powers
- Square calculations
- Percentages
- Numerical expressions

Examples:

"What is 25 + 75?"
-> MUST call calculator

"Calculate 125 * 8"
-> MUST call calculator

"What is 2^10?"
-> MUST call calculator

You MUST NOT calculate mathematical expressions yourself.

==================================================

WIKIPEDIA RULE

If the user asks for information involving:

- A person
- Biography
- History
- Historical event
- Origin
- Invention
- Creator
- Creation date
- Programming language history
- Technology history
- Detailed factual information

you MUST call wikipedia_search.

Examples:

"Who was Albert Einstein?"
-> MUST call wikipedia_search

"Who is Alan Turing?"
-> MUST call wikipedia_search

"What is the history of the Internet?"
-> MUST call wikipedia_search

"Who created Python?"
-> MUST call wikipedia_search

"When was Python created?"
-> MUST call wikipedia_search

Do not answer these questions directly from your own knowledge.

==================================================

NO TOOL RULE

Do NOT use any tool for general conceptual questions
when the question does not explicitly require:

- Mathematical calculation
- Historical information
- Biography
- Origin
- Creator
- Creation date
- Detailed factual retrieval

Examples:

"What is machine learning?"
-> No tool

"What is the difference between AI and ML?"
-> No tool

"What are the advantages of cloud computing?"
-> No tool

==================================================

MIXED QUERY RULE

If a query contains both mathematical calculation
and factual/historical information,
you MUST call ALL required tools.

Example:

"Calculate 50 * 20 and tell me about Alan Turing"

You MUST:

1. Call calculator
2. Call wikipedia_search

Another example:

"What is 100 / 4 and who was Albert Einstein?"

You MUST:

1. Call calculator
2. Call wikipedia_search

==================================================

IMPORTANT RULES

Correct tool usage is more important than answering quickly.

Never calculate mathematical expressions yourself.

Always use calculator for mathematics.

Always use wikipedia_search for biography,
history, origin, creator, creation date,
or detailed factual information.

For mixed queries, use every required tool.

Do not call unnecessary tools.

Keep the final answer concise.

Normally answer in 2 to 5 sentences.
Do not generate unnecessarily long responses.
"""


    # ======================================================
    # CREATE AGENT
    # ======================================================

    agent = create_agent(

        model=llm,

        tools=tools,

        system_prompt=system_prompt

    )


    results = []


    print("\n")
    print("=" * 70)

    print(
        "STARTING 20-QUERY EVALUATION"
    )

    print("=" * 70)


    # ======================================================
    # RUN EACH QUERY
    # ======================================================

    for test in TEST_QUERIES:


        query_id = test[
            "id"
        ]


        query = test[
            "query"
        ]


        category = test[
            "category"
        ]


        expected_tools = test[
            "expected_tools"
        ]


        print(
            f"\n[{query_id}/20] {query}"
        )


        # --------------------------------------------------
        # Remember MCP timing position
        # --------------------------------------------------

        mcp_start_index = len(
            mcp_timings
        )


        # --------------------------------------------------
        # Start response timer
        # --------------------------------------------------

        start_time = time.perf_counter()


        error = ""


        actual_tools = []


        input_tokens = 0


        output_tokens = 0


        total_tokens = 0


        try:


            # ----------------------------------------------
            # CALL AGENT WITH RETRY
            # ----------------------------------------------

            response = await invoke_with_retry(

                agent,

                {
                    "messages": [

                        {
                            "role":
                                "user",

                            "content":
                                query
                        }

                    ]
                }

            )


            # ----------------------------------------------
            # END TIMER
            # ----------------------------------------------

            end_time = (
                time.perf_counter()
            )


            response_time = (

                end_time
                -
                start_time

            )


            messages = response[
                "messages"
            ]


            # ----------------------------------------------
            # EXTRACT TOOL CALLS
            # ----------------------------------------------

            actual_tools = (

                extract_tool_calls(
                    messages
                )

            )


            # ----------------------------------------------
            # EXTRACT TOKEN USAGE
            # ----------------------------------------------

            (

                input_tokens,

                output_tokens,

                total_tokens

            ) = extract_token_usage(
                messages
            )


        except Exception as e:


            end_time = (
                time.perf_counter()
            )


            response_time = (

                end_time
                -
                start_time

            )


            error = str(
                e
            )


        # ==================================================
        # CALCULATE MCP OVERHEAD
        # ==================================================

        query_mcp_calls = (

            mcp_timings[
                mcp_start_index:
            ]

        )


        mcp_overhead = sum(

            item["time"]

            for item
            in query_mcp_calls

        )


        # ==================================================
        # CHECK TOOL ACCURACY
        # ==================================================

        tool_correct = (

            check_tool_accuracy(

                expected_tools,

                actual_tools

            )

        )


        # ==================================================
        # CHECK ERROR
        # ==================================================

        has_error = bool(
            error
        )


        # ==================================================
        # CHECK UNNECESSARY TOOL
        # ==================================================

        unnecessary_tool = False


        if not expected_tools:

            if actual_tools:

                unnecessary_tool = True


        # ==================================================
        # STORE RESULT
        # ==================================================

        result = {

            "Query No":
                query_id,

            "Category":
                category,

            "Query":
                query,

            "Expected Tools":

                ", ".join(
                    expected_tools
                ),

            "Selected Tools":

                ", ".join(
                    actual_tools
                ),

            "Tool Selection Correct":

                "Yes"

                if tool_correct

                else "No",

            "Error":

                "Yes"

                if has_error

                else "No",

            "Error Details":
                error,

            "Input Tokens":
                input_tokens,

            "Output Tokens":
                output_tokens,

            "Total Tokens":
                total_tokens,

            "MCP Communication Overhead (s)":

                round(
                    mcp_overhead,
                    4
                ),

            "Unnecessary Tool":

                "Yes"

                if unnecessary_tool

                else "No",

            "Response Time (s)":

                round(
                    response_time,
                    4
                )
        }


        results.append(
            result
        )


        # ==================================================
        # PRINT RESULTS
        # ==================================================

        print(
            "Selected Tools:",
            actual_tools
        )


        print(
            "Correct:",
            tool_correct
        )


        print(
            f"Response Time: "
            f"{response_time:.4f}s"
        )


        print(
            f"Input Tokens: "
            f"{input_tokens}"
        )


        print(
            f"Output Tokens: "
            f"{output_tokens}"
        )


        print(
            f"Total Tokens: "
            f"{total_tokens}"
        )


        print(
            f"MCP Overhead: "
            f"{mcp_overhead:.4f}s"
        )


        if error:

            print(
                "ERROR:",
                error
            )


        # ==================================================
        # DELAY TO AVOID GROQ RATE LIMIT
        # ==================================================

        print(
            "Waiting 3 seconds before next query..."
        )


        await asyncio.sleep(
            3
        )


    # ======================================================
    # SAVE CSV
    # ======================================================

    csv_file = os.path.join(

        RESULTS_DIR,

        "evaluation_results.csv"

    )


    with open(

        csv_file,

        "w",

        newline="",

        encoding="utf-8"

    ) as file:


        writer = csv.DictWriter(

            file,

            fieldnames=
                results[0].keys()

        )


        writer.writeheader()


        writer.writerows(
            results
        )


    # ======================================================
    # CALCULATE FINAL METRICS
    # ======================================================

    total_queries = len(
        results
    )


    correct_tools = sum(

        r[
            "Tool Selection Correct"
        ] == "Yes"

        for r in results

    )


    errors = sum(

        r[
            "Error"
        ] == "Yes"

        for r in results

    )


    unnecessary = sum(

        r[
            "Unnecessary Tool"
        ] == "Yes"

        for r in results

    )


    total_tokens_used = sum(

        r[
            "Total Tokens"
        ]

        for r in results

    )


    total_response_time = sum(

        r[
            "Response Time (s)"
        ]

        for r in results

    )


    total_mcp_time = sum(

        r[
            "MCP Communication Overhead (s)"
        ]

        for r in results

    )


    # ======================================================
    # FINAL METRICS
    # ======================================================

    tool_accuracy = (

        correct_tools
        /
        total_queries

    ) * 100


    error_rate = (

        errors
        /
        total_queries

    ) * 100


    unnecessary_rate = (

        unnecessary
        /
        total_queries

    ) * 100


    average_tokens = (

        total_tokens_used
        /
        total_queries

    )


    average_response_time = (

        total_response_time
        /
        total_queries

    )


    average_mcp_time = (

        total_mcp_time
        /
        total_queries

    )


    # ======================================================
    # CATEGORY ACCURACY
    # ======================================================

    categories = [

        "Calculator",

        "Wikipedia",

        "No Tool",

        "Mixed",

        "Ambiguous"

    ]


    category_accuracy = {}


    for category in categories:


        category_rows = [

            r

            for r in results

            if r["Category"]
            ==
            category

        ]


        if category_rows:


            correct = sum(

                r[
                    "Tool Selection Correct"
                ]
                ==
                "Yes"

                for r
                in category_rows

            )


            category_accuracy[
                category
            ] = (

                correct
                /
                len(category_rows)

            ) * 100


    # ======================================================
    # GRAPH 1
    # QUERY VS RESPONSE TIME
    # ======================================================

    query_numbers = [

        r["Query No"]

        for r in results

    ]


    response_times = [

        r[
            "Response Time (s)"
        ]

        for r in results

    ]


    plt.figure(
        figsize=(12, 6)
    )


    plt.plot(

        query_numbers,

        response_times,

        marker="o"

    )


    plt.xlabel(
        "Query Number"
    )


    plt.ylabel(
        "Response Time (seconds)"
    )


    plt.title(
        "Query vs Response Time"
    )


    plt.xticks(
        query_numbers
    )


    plt.grid(
        True
    )


    plt.tight_layout()


    graph1 = os.path.join(

        RESULTS_DIR,

        "query_response_time.png"

    )


    plt.savefig(

        graph1,

        dpi=300

    )


    plt.close()


    # ======================================================
    # GRAPH 2
    # TOOL SELECTION ACCURACY BY CATEGORY
    # ======================================================

    category_names = list(
        category_accuracy.keys()
    )


    category_values = list(
        category_accuracy.values()
    )


    plt.figure(
        figsize=(10, 6)
    )


    plt.bar(

        category_names,

        category_values

    )


    plt.xlabel(
        "Query Category"
    )


    plt.ylabel(
        "Tool Selection Accuracy (%)"
    )


    plt.title(
        "Tool Selection Accuracy by Query Category"
    )


    plt.ylim(
        0,
        100
    )


    plt.grid(
        axis="y"
    )


    plt.tight_layout()


    graph2 = os.path.join(

        RESULTS_DIR,

        "tool_selection_accuracy.png"

    )


    plt.savefig(

        graph2,

        dpi=300

    )


    plt.close()


    # ======================================================
    # PRINT FINAL RESULTS
    # ======================================================

    print("\n")

    print("=" * 70)

    print(
        "FINAL EVALUATION RESULTS"
    )

    print("=" * 70)


    print(
        f"\nTotal Queries: "
        f"{total_queries}"
    )


    print(
        f"Tool Selection Accuracy: "
        f"{tool_accuracy:.2f}%"
    )


    print(
        f"Error Rate: "
        f"{error_rate:.2f}%"
    )


    print(
        f"Average Token Usage: "
        f"{average_tokens:.2f}"
    )


    print(
        f"Average MCP Communication Overhead: "
        f"{average_mcp_time:.4f}s"
    )


    print(
        f"Unnecessary Tool Rate: "
        f"{unnecessary_rate:.2f}%"
    )


    print(
        f"Average Response Time: "
        f"{average_response_time:.4f}s"
    )


    print(
        "\nCategory Accuracy:"
    )


    for category, accuracy in (
        category_accuracy.items()
    ):


        print(

            f"{category}: "
            f"{accuracy:.2f}%"

        )


    print(
        "\nFiles Generated:"
    )


    print(
        f"CSV: {csv_file}"
    )


    print(
        f"Graph 1: {graph1}"
    )


    print(
        f"Graph 2: {graph2}"
    )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    asyncio.run(
        run_evaluation()
    )