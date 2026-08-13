from fastmcp import Client


async def get_mcp_client():

    client = Client(
        "http://127.0.0.1:8000/mcp"
    )

    return client