from langchain_mcp_adapters.client import MultiServerMCPClient


def create_mcp_client():

    client = MultiServerMCPClient(
        {
            "calculator_wikipedia": {
                "transport": "streamable_http",
                "url": "http://127.0.0.1:8000/mcp"
            }
        }
    )

    return client