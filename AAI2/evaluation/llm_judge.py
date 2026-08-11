import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from groq import Groq
from config.config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)


def judge(response):

    prompt = f"""
Rate the following customer support response on a scale of 1 to 10.

Criteria:

- Correctness
- Completeness
- Politeness
- Helpfulness

Response:

{response}

Return only a single number.
"""

    result = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return result.choices[0].message.content