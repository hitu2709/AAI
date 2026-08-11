import time
import pandas as pd
from groq import Groq
from config.config import GROQ_API_KEY


from prompting.role_prompt import role_prompt
from prompting.instruction_prompt import instruction_prompt
from prompting.contextual_prompt import contextual_prompt
from prompting.structured_prompt import structured_prompt

client = Groq(api_key=GROQ_API_KEY)

prompts = {
    "Role": role_prompt,
    "Instruction": instruction_prompt,
    "Contextual": contextual_prompt,
    "Structured": structured_prompt
}

results=[]

for name,prompt in prompts.items():

    start=time.time()

    response = client.chat.completions.create(

    model="llama-3.3-70b-versatile",

    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

    end=time.time()

    answer=response.choices[0].message.content

    latency=end-start

    tokens=response.usage.total_tokens

    results.append([

        name,

        answer,

        latency,

        tokens

    ])

df=pd.DataFrame(results,

columns=[

"Prompt",

"Response",

"Latency",

"Tokens"

])

df.to_csv("outputs/responses.csv",index=False)

print(df)