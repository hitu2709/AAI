from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

SYSTEM_PROMPT = """
You are an AI Agent.

You have two tools.

Tool 1:
Wikipedia
Use when the user asks factual questions.

Tool 2:
Calculator
Use when mathematical calculations are required.

Respond ONLY with one of the following formats:

TOOL: wikipedia
INPUT: search term

OR

TOOL: calculator
INPUT: mathematical expression

OR

TOOL: none
INPUT: answer yourself
"""


def decide(memory):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    messages.extend(memory)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )

    return response.choices[0].message.content