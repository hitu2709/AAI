import os

from openai import OpenAI
from crewai import Agent
from crewai.llms.base_llm import BaseLLM


class GroqGPTOSS(BaseLLM):
    """
    Custom CrewAI LLM using Groq's OpenAI-compatible API.

    This does NOT use LiteLLM.
    """

    def __init__(self):
        super().__init__(
            model="openai/gpt-oss-20b",
            temperature=0.3
        )

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing from the .env file."
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
        **kwargs
    ):

        # Convert CrewAI messages into OpenAI format
        formatted_messages = []

        for message in messages:

            if isinstance(message, dict):
                role = message.get("role", "user")
                content = message.get("content", "")

            else:
                role = getattr(
                    message,
                    "role",
                    "user"
                )

                content = getattr(
                    message,
                    "content",
                    ""
                )

            formatted_messages.append(
                {
                    "role": role,
                    "content": content
                }
            )

        # Basic request
        request = {
            "model": "openai/gpt-oss-20b",
            "messages": formatted_messages,
            "temperature": 0.3
        }

        # Only add tools if CrewAI provides them
        if tools:
            request["tools"] = tools

        # Call Groq directly
        response = self.client.chat.completions.create(
            **request
        )

        message = response.choices[0].message

        # Handle tool calls
        if getattr(message, "tool_calls", None):

            tool_calls = []

            for tool_call in message.tool_calls:

                tool_calls.append(
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }
                )

            return {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": tool_calls
            }

        return message.content or ""


def create_agents():

    # ---------------------------------------------
    # Create custom Groq LLM
    # ---------------------------------------------

    llm = GroqGPTOSS()

    # ---------------------------------------------
    # Research Agent
    # ---------------------------------------------

    research_agent = Agent(

        role="Research Agent",

        goal=(
            "Collect accurate and relevant information "
            "about the given research topic and organize "
            "the findings clearly."
        ),

        backstory=(
            "You are an expert research analyst. "
            "You investigate a topic and collect important "
            "concepts, facts, applications, advantages, "
            "limitations and real-world examples."
        ),

        llm=llm,

        verbose=True,

        allow_delegation=True
    )

    # ---------------------------------------------
    # Content Writer Agent
    # ---------------------------------------------

    writer_agent = Agent(

        role="Content Writer Agent",

        goal=(
            "Create a structured and informative research "
            "report using the research findings provided "
            "by the Research Agent."
        ),

        backstory=(
            "You are a professional technical writer. "
            "You convert research findings into a clear, "
            "well-organized and readable technical report."
        ),

        llm=llm,

        verbose=True,

        allow_delegation=False
    )

    # ---------------------------------------------
    # Reviewer Agent
    # ---------------------------------------------

    reviewer_agent = Agent(

        role="Reviewer Agent",

        goal=(
            "Review the generated report for accuracy, "
            "completeness, clarity and consistency."
        ),

        backstory=(
            "You are an experienced technical reviewer. "
            "You identify missing information, incorrect "
            "statements, repetition and unclear explanations "
            "and improve the final report."
        ),

        llm=llm,

        verbose=True,

        allow_delegation=False
    )

    return (
        research_agent,
        writer_agent,
        reviewer_agent
    )