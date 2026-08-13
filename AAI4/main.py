import asyncio
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent


load_dotenv()


async def main():

    # Connect to MCP Server
    client = MultiServerMCPClient(
        {
            "calculator_wikipedia": {
                "transport": "streamable_http",
                "url": "http://127.0.0.1:8000/mcp"
            }
        }
    )


    # Discover MCP Tools
    tools = await client.get_tools()

    print("\nAvailable MCP Tools:")

    for tool in tools:
        print("-", tool.name)


    # Initialize Groq LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )


    # Create LangGraph Agent
    agent = create_react_agent(
        llm,
        tools
    )


    print("\n===================================")
    print(" Agentic MCP Chatbot Started")
    print(" Type 'exit' to stop")
    print("===================================\n")


    while True:

        query = input("You: ")

        if query.lower() in ["exit", "quit", "stop"]:
            print("Chatbot stopped.")
            break


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


        print("\nAssistant:")
        print(response["messages"][-1].content)
        print()


if __name__ == "__main__":
    asyncio.run(main())