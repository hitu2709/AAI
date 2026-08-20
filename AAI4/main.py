import asyncio

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from client import create_mcp_client


load_dotenv()


async def main():

    # -----------------------------
    # Create MCP Client
    # -----------------------------

    client = create_mcp_client()

    # -----------------------------
    # Discover MCP Tools
    # -----------------------------

    tools = await client.get_tools()

    print("\nAvailable MCP Tools:")

    for tool in tools:
        print("-", tool.name)

    # -----------------------------
    # Initialize Groq
    # -----------------------------

    llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=250
)

    # -----------------------------
    # Bind MCP Tools to LLM
    # -----------------------------

    llm_with_tools = llm.bind_tools(tools)

    # -----------------------------
    # Create LangGraph Agent
    # -----------------------------

    agent = create_react_agent(
        model=llm_with_tools,
        tools=tools
    )

    print("\n===================================")
    print("     MCP Agentic Chatbot")
    print("===================================")
    print("Type 'exit' to stop.\n")

    # -----------------------------
    # Chat Loop
    # -----------------------------

    while True:

        query = input("You: ")

        if query.lower() in [
            "exit",
            "quit",
            "stop"
        ]:
            break

        try:

            response = await agent.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": query
                        }
                    ]
                }
            )

            answer = response["messages"][-1].content

            print("\nAssistant:")
            print(answer)
            print()

        except Exception as e:

            print("\nError:", e)


if __name__ == "__main__":
    asyncio.run(main())