from pathlib import Path
from typing import TypedDict

import json

from langgraph.graph import StateGraph, START, END

from src.rag.retriever import Retriever
from src.rag.generator import generate


ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# 1. STATE
# ============================================================

class AgentState(TypedDict):
    topic: str
    context: str
    draft: str
    critique: dict
    attempts: int


# ============================================================
# 2. RETRIEVER
# ============================================================

retriever = Retriever(
    index_path=ROOT / "data" / "embeddings" / "faiss.index",
    metadata_path=ROOT / "data" / "embeddings" / "metadata.json",
)


# ============================================================
# 3. NODES
# ============================================================

def retrieve(state: AgentState):
    """
    Retrieve relevant historical Reddit examples
    from our FAISS vector database.
    """

    results = retriever.search(
        state["topic"],
        k=5,
    )

    context = "\n\n".join(
        result["text"]
        for result in results
    )

    print("\n===== RETRIEVED EXAMPLES =====")

    for i, result in enumerate(results, start=1):
        print(f"\n#{i}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Source: {result['source']}")
        print(f"Subreddit: {result.get('subreddit')}")
        print(f"Score: {result.get('score')}")
        print(f"Text: {result['text'][:300]}")

    return {
        "context": context
    }


def generate_post(state: AgentState):
    """
    Generate a shitpost using the retrieved Reddit examples.

    If this is a retry, include the critic's previous
    feedback so the model knows what to improve.
    """

    previous_feedback = ""

    if state["critique"]:
        previous_feedback = f"""
The previous attempt was rejected.

Here is the critic's feedback:

{json.dumps(state["critique"], indent=2)}

Improve the new post specifically based on this feedback.
Do NOT simply rewrite the previous joke.
"""

    prompt = f"""
You are an internet shitposter who writes clever,
natural Reddit-style humor.

Current topic:
{state["topic"]}

Historical Reddit examples:
--------------------
{state["context"]}
--------------------

{previous_feedback}

Write ONE original shitpost about the current topic.

Style requirements:
- Make it sound like a real person wrote it.
- Be smart, sarcastic, witty, or absurd depending on the topic.
- Use different styles of humor.
- Do not make every joke follow the same sentence structure.
- Do not repeatedly use phrases like "I just realized".
- Avoid generic AI-sounding humor.
- Avoid unnecessary explanations.
- Keep it concise.
- The joke should feel natural enough to be posted on Reddit.
- Use the historical examples only as inspiration.
- NEVER copy an example.
- Do not explain the joke.

Return ONLY the shitpost.
"""

    draft = generate(prompt)

    attempt = state["attempts"] + 1

    print(f"\n===== DRAFT {attempt} =====")
    print(draft)

    return {
        "draft": draft,
        "attempts": attempt,
    }


def critic(state: AgentState):
    """
    Evaluate the generated post using structured scores.
    """

    prompt = f"""
You are a strict Reddit humor critic.

Evaluate this shitpost:

"{state["draft"]}"

Current topic:

"{state["topic"]}"

Evaluate these four dimensions from 1 to 10:

1. humor
   How genuinely funny is it?

2. relevance
   How strongly does it relate to the topic?

3. originality
   Does it feel fresh rather than like a generic AI joke?

4. cringe
   How forced, predictable, or AI-generated does it feel?
   Higher score = MORE cringe.

A post can be considered GOOD only when:

- humor >= 7
- relevance >= 7
- originality >= 6
- cringe <= 4

Return ONLY valid JSON.

Use exactly this structure:

{{
    "humor": 0,
    "relevance": 0,
    "originality": 0,
    "cringe": 0,
    "verdict": "GOOD",
    "feedback": "short explanation of what should be improved"
}}

The verdict must be either:

GOOD
BAD
"""

    response = generate(prompt).strip()

    # --------------------------------------------------------
    # Parse JSON returned by the LLM
    # --------------------------------------------------------

    try:
        critique = json.loads(response)

    except json.JSONDecodeError:
        print("\nWARNING: Critic returned invalid JSON.")
        print("Raw response:")
        print(response)

        # Treat malformed critic output as BAD so that
        # the agent gets another chance.
        critique = {
            "humor": 0,
            "relevance": 0,
            "originality": 0,
            "cringe": 10,
            "verdict": "BAD",
            "feedback": "Critic failed to return valid structured output.",
        }

    print("\n===== CRITIC =====")
    print(json.dumps(critique, indent=2))

    return {
        "critique": critique
    }


# ============================================================
# 4. ROUTING
# ============================================================

def route_after_critic(state: AgentState):

    critique = state["critique"]

    good = (
        critique.get("humor", 0) >= 7
        and critique.get("relevance", 0) >= 7
        and critique.get("originality", 0) >= 6
        and critique.get("cringe", 10) <= 4
    )

    if good or state["attempts"] >= 3:
        return "end"

    return "retry"


# ============================================================
# 5. BUILD LANGGRAPH
# ============================================================

builder = StateGraph(AgentState)

# Nodes
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate_post)
builder.add_node("critic", critic)

# Initial flow
builder.add_edge(
    START,
    "retrieve",
)

builder.add_edge(
    "retrieve",
    "generate",
)

builder.add_edge(
    "generate",
    "critic",
)

# Critic decides what happens next
builder.add_conditional_edges(
    "critic",
    route_after_critic,
    {
        "retry": "generate",
        "end": END,
    },
)

# Compile graph
graph = builder.compile()


# ============================================================
# 6. RUN
# ============================================================

if __name__ == "__main__":

    topic = input("Topic: ")

    initial_state: AgentState = {
        "topic": topic,
        "context": "",
        "draft": "",
        "critique": {},
        "attempts": 0,
    }

    result = graph.invoke(initial_state)

    print("\n")
    print("=" * 50)
    print("FINAL POST")
    print("=" * 50)

    print(result["draft"])

    print("\n")
    print("=" * 50)
    print("FINAL CRITIQUE")
    print("=" * 50)

    print(json.dumps(
        result["critique"],
        indent=2
    ))

    print("\n")
    print("=" * 50)
    print("ATTEMPTS")
    print("=" * 50)

    print(result["attempts"])