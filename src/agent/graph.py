from pathlib import Path
from typing import TypedDict
import json

from langgraph.graph import StateGraph, START, END

from src.rag.retriever import Retriever
from src.rag.generator import generate
from src.tools.news import get_latest_news


ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# 1. STATE
# ============================================================

class AgentState(TypedDict):
    topic: str

    # RAG context
    context: str

    # Current news
    news: str

    # News routing decision
    categories: list[str]
    needs_news: bool

    # Generated post
    draft: str

    # Critic output
    critique: dict

    # Number of generation attempts
    attempts: int


# ============================================================
# 2. RETRIEVER
# ============================================================

retriever = Retriever(
    index_path=ROOT / "data" / "embeddings" / "faiss.index",
    metadata_path=ROOT / "data" / "embeddings" / "metadata.json",
)


# ============================================================
# 3. TOPIC CLASSIFIER
# ============================================================

def classify_topic(state: AgentState):
    """
    Decide whether current news is useful for the topic
    and determine which news categories are relevant.
    """

    prompt = f"""
You are a topic classification agent.

The user wants to create a shitpost about:

"{state["topic"]}"

Your job is to determine:

1. Whether CURRENT news would improve the shitpost.
2. Which news categories are relevant.

Available categories:

- tech
- gaming
- entertainment
- sports
- politics
- science
- business
- social_media
- general

Rules:

- Use current news when the topic refers to something
  that could have changed recently.
- Use current news for recent events, people, products,
  releases, announcements, controversies, trends, etc.
- If the topic is timeless, current news is unnecessary.
- You may select multiple categories.
- Do not select unrelated categories.
- If current news is not useful, return an empty category list.

Examples:

Topic:
"Google released a new AI model"

Output:
{{
    "needs_news": true,
    "categories": ["tech"]
}}

Topic:
"The new GTA trailer is insane"

Output:
{{
    "needs_news": true,
    "categories": ["gaming", "entertainment"]
}}

Topic:
"Programmers forgetting semicolons"

Output:
{{
    "needs_news": false,
    "categories": []
}}

Return ONLY valid JSON.

Use exactly this structure:

{{
    "needs_news": true,
    "categories": ["tech"]
}}
"""

    response = generate(prompt).strip()

    try:
        decision = json.loads(response)

    except json.JSONDecodeError:

        print("\nWARNING: Classifier returned invalid JSON.")
        print("Raw response:")
        print(response)

        decision = {
            "needs_news": False,
            "categories": [],
        }

    # --------------------------------------------------------
    # Validate categories
    # --------------------------------------------------------

    allowed_categories = {
        "tech",
        "gaming",
        "entertainment",
        "sports",
        "politics",
        "science",
        "business",
        "social_media",
        "general",
    }

    categories = [
        category
        for category in decision.get("categories", [])
        if category in allowed_categories
    ]

    needs_news = bool(
        decision.get("needs_news", False)
    )

    # If no valid categories exist, news isn't useful.
    if not categories:
        needs_news = False

    print("\n===== TOPIC CLASSIFICATION =====")

    print(
        json.dumps(
            {
                "needs_news": needs_news,
                "categories": categories,
            },
            indent=2,
        )
    )

    return {
        "needs_news": needs_news,
        "categories": categories,
    }


# ============================================================
# 4. NEWS TOOL NODE
# ============================================================

def fetch_news(state: AgentState):
    """
    Fetch current news based on the categories selected
    by the classifier.
    """

    if not state["needs_news"]:

        print("\n===== CURRENT NEWS =====")
        print("News not required for this topic.")

        return {
            "news": ""
        }

    categories = state["categories"]

    if not categories:

        print("\n===== CURRENT NEWS =====")
        print("No news categories selected.")

        return {
            "news": ""
        }

    print("\n===== FETCHING NEWS =====")
    print(f"Categories: {categories}")

    articles = get_latest_news(
        categories=categories,
        limit_per_category=5,
    )

    if not articles:

        print("No current news was found.")

        return {
            "news": ""
        }

    # --------------------------------------------------------
    # Convert articles into context for the LLM
    # --------------------------------------------------------

    news = "\n\n".join(
        f"Category: {article['category']}\n"
        f"Title: {article['title']}\n"
        f"Summary: {article['summary']}\n"
        f"URL: {article['url']}"
        for article in articles
    )

    # --------------------------------------------------------
    # Display fetched news
    # --------------------------------------------------------

    print("\n===== CURRENT NEWS =====")

    for i, article in enumerate(articles, start=1):

        print(f"\n#{i}")
        print(f"Category: {article['category']}")
        print(f"Title: {article['title']}")
        print(f"URL: {article['url']}")

    return {
        "news": news
    }


# ============================================================
# 5. RAG RETRIEVAL NODE
# ============================================================

def retrieve(state: AgentState):
    """
    Retrieve historically similar Reddit content
    from the FAISS vector database.
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

        print(
            f"Similarity: "
            f"{result.get('similarity', 0):.4f}"
        )

        print(
            f"Source: "
            f"{result.get('source', 'unknown')}"
        )

        print(
            f"Subreddit: "
            f"{result.get('subreddit')}"
        )

        print(
            f"Score: "
            f"{result.get('score')}"
        )

        print(
            f"Text: "
            f"{result['text'][:300]}"
        )

    return {
        "context": context
    }


# ============================================================
# 6. GENERATION NODE
# ============================================================

def generate_post(state: AgentState):
    """
    Generate a new original shitpost using:

    - the user's topic
    - historical Reddit examples
    - current news
    - previous critic feedback
    """

    previous_feedback = ""

    # --------------------------------------------------------
    # Add critic feedback when retrying
    # --------------------------------------------------------

    if state["critique"]:

        previous_feedback = f"""
The previous attempt was rejected.

Here is the critic's feedback:

{json.dumps(state["critique"], indent=2)}

Improve the new post specifically based on this feedback.

Do NOT simply rewrite the previous joke.
"""

    # --------------------------------------------------------
    # Build generation prompt
    # --------------------------------------------------------

    prompt = f"""
You are an internet shitposter who writes clever,
natural Reddit-style humor.

Current topic:

{state["topic"]}

Historical Reddit examples:
--------------------
{state["context"]}
--------------------

Current news:
--------------------
{state["news"]}
--------------------

{previous_feedback}

Write ONE original shitpost about the current topic.

If current news is provided, use it as inspiration
when relevant.

Do NOT simply repeat a news headline.

Style requirements:

- Make it sound like a real person wrote it.
- Be smart, sarcastic, witty, absurd, observational,
  self-deprecating, or use wordplay depending on the topic.
- Use different styles of humor.
- Do not make every joke follow the same structure.
- Do not repeatedly use phrases like "I just realized".
- Avoid generic AI-sounding humor.
- Avoid predictable setups whenever possible.
- Avoid unnecessary explanations.
- Keep it concise.
- Make it feel natural enough to post on Reddit.
- Use historical examples only as inspiration.
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


# ============================================================
# 7. CRITIC NODE
# ============================================================

def critic(state: AgentState):
    """
    Evaluate the generated shitpost using structured scores.
    """

    prompt = f"""
You are a strict Reddit humor critic.

Evaluate this shitpost:

"{state["draft"]}"

Current topic:

"{state["topic"]}"

Score each category from 1 to 10.

1. humor
   How genuinely funny is it?

2. relevance
   How strongly does it relate to the topic?

3. originality
   Does it feel fresh rather than like a generic AI joke?

4. cringe
   How forced, predictable, or AI-generated does it feel?

   Higher score = MORE cringe.

A post is considered GOOD only when:

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

or:

BAD
"""

    response = generate(prompt).strip()

    try:

        critique = json.loads(response)

    except json.JSONDecodeError:

        print("\nWARNING: Critic returned invalid JSON.")
        print("Raw response:")
        print(response)

        critique = {
            "humor": 0,
            "relevance": 0,
            "originality": 0,
            "cringe": 10,
            "verdict": "BAD",
            "feedback": (
                "Critic failed to return valid "
                "structured output."
            ),
        }

    print("\n===== CRITIC =====")

    print(
        json.dumps(
            critique,
            indent=2,
        )
    )

    return {
        "critique": critique
    }


# ============================================================
# 8. CRITIC ROUTING
# ============================================================

def route_after_critic(state: AgentState):
    """
    Decide whether to accept the post or retry.

    The application code determines whether the post
    passes the required quality thresholds rather than
    blindly trusting the LLM's verdict.
    """

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
# 9. BUILD LANGGRAPH
# ============================================================

builder = StateGraph(AgentState)

# ------------------------------------------------------------
# Nodes
# ------------------------------------------------------------

builder.add_node(
    "classify",
    classify_topic,
)

builder.add_node(
    "news",
    fetch_news,
)

builder.add_node(
    "retrieve",
    retrieve,
)

builder.add_node(
    "generate",
    generate_post,
)

builder.add_node(
    "critic",
    critic,
)


# ------------------------------------------------------------
# Main flow
# ------------------------------------------------------------

builder.add_edge(
    START,
    "classify",
)

builder.add_edge(
    "classify",
    "news",
)

builder.add_edge(
    "news",
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


# ------------------------------------------------------------
# Critic routing
# ------------------------------------------------------------

builder.add_conditional_edges(
    "critic",
    route_after_critic,
    {
        "retry": "generate",
        "end": END,
    },
)


# ------------------------------------------------------------
# Compile
# ------------------------------------------------------------

graph = builder.compile()


# ============================================================
# 10. RUN
# ============================================================

if __name__ == "__main__":

    topic = input("Topic: ")

    initial_state: AgentState = {
        "topic": topic,
        "context": "",
        "news": "",
        "categories": [],
        "needs_news": False,
        "draft": "",
        "critique": {},
        "attempts": 0,
    }

    result = graph.invoke(
        initial_state
    )

    print("\n")
    print("=" * 60)
    print("FINAL POST")
    print("=" * 60)

    print(result["draft"])

    print("\n")
    print("=" * 60)
    print("FINAL CRITIQUE")
    print("=" * 60)

    print(
        json.dumps(
            result["critique"],
            indent=2,
        )
    )

    print("\n")
    print("=" * 60)
    print("ATTEMPTS")
    print("=" * 60)

    print(result["attempts"])