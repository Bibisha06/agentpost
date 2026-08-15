import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()

client = Groq(
    api_key=os.environ["GROQ_API_KEY"]
)

MODEL = "llama-3.3-70b-versatile"


def generate(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.8,
    )

    return response.choices[0].message.content