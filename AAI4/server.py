from fastmcp import FastMCP
import wikipedia

# Create MCP Server
mcp = FastMCP("Calculator and Wikipedia MCP Server")


# -----------------------------
# Calculator Tool
# -----------------------------
@mcp.tool()
def calculator(expression: str) -> str:
    """
    Perform basic mathematical calculations.

    Example:
    10 + 20
    50 * 2
    100 / 5
    """

    try:
        allowed_chars = "0123456789+-*/(). %"

        if not all(char in allowed_chars for char in expression):
            return "Invalid mathematical expression."

        result = eval(expression, {"__builtins__": {}}, {})

        return f"Result: {result}"

    except Exception as e:
        return f"Calculation error: {str(e)}"


# -----------------------------
# Wikipedia Search Tool
# -----------------------------
@mcp.tool()
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia and return a summary.
    """

    try:
        summary = wikipedia.summary(
            query,
            sentences=3,
            auto_suggest=True
        )

        return summary

    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple results found. Try one of: {e.options[:5]}"

    except wikipedia.exceptions.PageError:
        return "No Wikipedia page found for this topic."

    except Exception as e:
        return f"Wikipedia search error: {str(e)}"


# -----------------------------
# Start MCP Server
# -----------------------------
if __name__ == "__main__":

    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=8000,
        path="/mcp"
    )