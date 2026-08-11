from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools.python.tool import PythonREPLTool


class SearchTool:
    """
    Wrapper around DuckDuckGo Search
    so it behaves like a Wikipedia tool.
    """

    def __init__(self):
        self.search = DuckDuckGoSearchRun()

    def invoke(self, query):
        try:
            return self.search.invoke(query)
        except Exception as e:
            return f"Search failed: {e}"


# Search Tool
wiki = SearchTool()

# Calculator Tool
calculator = PythonREPLTool()

# Available Tools
tools = [wiki, calculator]