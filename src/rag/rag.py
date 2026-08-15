from pathlib import Path

from retriever import Retriever
from generator import generate


ROOT = Path(__file__).resolve().parents[2]

retriever = Retriever(
    index_path=ROOT / "data" / "embeddings" / "faiss.index",
    metadata_path=ROOT / "data" / "embeddings" / "metadata.json",
)


def build_context(results):
    context_parts = []

    for i, result in enumerate(results, start=1):
        context_parts.append(
            f"""
Example {i}
Source: {result["source"]}
Subreddit: {result.get("subreddit")}
Score: {result.get("score")}

{result["text"]}
"""
        )

    return "\n".join(context_parts)


def rag(query: str) -> str:

    # 1. Retrieve relevant historical examples
    results = retriever.search(query, k=5)

    # 2. Turn retrieved documents into context
    context = build_context(results)

    # 3. Give context + task to the LLM
    prompt = f"""
You are a creative internet shitposter.

Use the historical Reddit examples below as inspiration.
Do NOT copy them directly.

Historical examples:
--------------------
{context}
--------------------

Current topic:
{query}

Write ONE original shitpost about the current topic.

Requirements:
- Be genuinely funny.
- Keep it concise.
- Use the style of internet humor.
- Do not explain the joke.
- Do not mention that you were given examples.
"""

    # 4. Generate
    return generate(prompt)


if __name__ == "__main__":
    query = input("Topic: ")

    answer = rag(query)

    print("\n===== GENERATED SHITPOST =====\n")
    print(answer)