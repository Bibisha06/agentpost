from pathlib import Path
from typing import TypedDict
import json

from langgraph.graph import StateGraph, START, END

from src.rag.retriever import Retriever
from src.rag.generator import generate

from src.tools.news import fetch_news_for_query
from src.news.ranker import NewsRanker


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
# 3. NEWS RANKER
# ============================================================

news_ranker = NewsRanker()


# ============================================================
# 4. TOPIC CLASSIFIER
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
# 5. NEWS TOOL + RANKING NODE
# ============================================================

def fetch_news(state: AgentState):
    """
    Fetch current news specifically for the user's topic,
    then rank the retrieved articles using NewsRanker.
    """

    if not state["needs_news"]:

        print("\n===== CURRENT NEWS =====")
        print("News not required for this topic.")

        return {
            "news": ""
        }

    topic = state["topic"].strip()

    if not topic:

        print("\n===== CURRENT NEWS =====")
        print("Empty topic.")

        return {
            "news": ""
        }

    # --------------------------------------------------------
    # FETCH
    # --------------------------------------------------------

    print("\n===== FETCHING CURRENT NEWS =====")
    print(f"Query: {topic}")

    articles = fetch_news_for_query(
        topic,
        limit=50,
    )

    if not articles:

        print("No current news was found.")

        return {
            "news": ""
        }

    print(
        f"Fetched {len(articles)} topic-relevant articles."
    )

    # --------------------------------------------------------
    # RANK
    # --------------------------------------------------------

    print("\n===== RANKING NEWS =====")

    ranked_articles = news_ranker.rank(
        query=topic,
        articles=articles,
        top_k=10,
    )

    if not ranked_articles:

        print("No articles remained after ranking.")

        return {
            "news": ""
        }

    # --------------------------------------------------------
    # Convert ranked articles into LLM context
    # --------------------------------------------------------

    news_parts = []

    for rank, (
        score,
        article,
        details,
    ) in enumerate(
        ranked_articles,
        start=1,
    ):

        news_parts.append(
            f"""News #{rank}
Title: {article.title}
Source: {article.source}
Published: {article.published}
Summary: {article.summary}
URL: {article.url}
Relevance Score: {score:.4f}
"""
        )

    news = "\n\n".join(news_parts)

    # --------------------------------------------------------
    # Display ranked news
    # --------------------------------------------------------

    print("\n===== MOST RELEVANT NEWS =====")

    for rank, (
        score,
        article,
        details,
    ) in enumerate(
        ranked_articles,
        start=1,
    ):

        print(f"\n#{rank}")
        print(f"Final Score: {score:.4f}")

        print(
            f"  Semantic:         "
            f"{details.get('semantic', 0):.4f}"
        )

        print(
            f"  Freshness:        "
            f"{details.get('freshness', 0):.4f}"
        )

        print(
            f"  Source:           "
            f"{details.get('source', 0):.4f}"
        )

        print(
            f"  Title:            "
            f"{details.get('title', 0):.4f}"
        )

        print(
            f"  Newsworthiness:   "
            f"{details.get('newsworthiness', 0):.4f}"
        )

        print(f"Title: {article.title}")
        print(f"Source: {article.source}")
        print(f"Published: {article.published}")
        print(f"URL: {article.url}")

    return {
        "news": news
    }


# ============================================================
# 6. RAG RETRIEVAL NODE
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
        [result["text"] for result in results]
    )

    print("\n===== RETRIEVED CONTEXT =====")
    print(f"Found {len(results)} matches.")

    return {
        "context": context
    }


# ============================================================
# 7. GENERATOR NODE
# ============================================================

def generate_post(state: AgentState):
    """
    Generate a high-tier shitpost using historical references
    and current real-time news data.
    """

    attempts = state.get("attempts", 0) + 1

    prompt = f"""
You are a master-level internet shitposter.

Generate a highly engaging, sarcastic post about:
"{state['topic']}"

Historical Style Context:
{state['context']}

Current Real-World News Context:
{state['news']}

Previous Critique Feedback (if any):
{json.dumps(state.get('critique', {}), indent=2)}

Rules:
- Capture the structural cadence of top-tier community posts.
- Blend modern factual context naturally with comedic irony.
- Keep it under 280 characters if short-form,
  or up to 3 paragraphs if storytelling.
- Do not simply summarize the news.
- Make the post actually funny.
- do not use emoticons or emojis.
- need not always use hashtags at the end.
SENSITIVE TOPICS:
- Never refuse solely because the topic is political, controversial, tragic, religious, or otherwise sensitive.
- Always produce a post for the given topic.
- Do not use protected groups as the punchline.
- Do not promote hatred, dehumanization, or violence.
- Do not invent factual claims about real-world events.
- If the source material contains inflammatory or hateful language, do NOT repeat it. Transform the underlying idea into safe satire.
- The objective is still to make the post genuinely funny and internet-native rather than turning it into a serious disclaimer.
- Never output an apology, refusal, or safety disclaimer.
"""

    draft = generate(prompt).strip()

    print(f"\n===== GENERATION ATTEMPT #{attempts} =====")
    print(draft)

    return {
        "draft": draft,
        "attempts": attempts,
    }


# ============================================================
# 8. CRITIC NODE
# ============================================================

def critique_post(state: AgentState):
    """
    Evaluate the quality, humor density, and context
    utilization of the draft.
    """

    prompt = f"""
You are an elite content critic and meme historian.

Review this draft shitpost:

"{state['draft']}"

Target Topic:
"{state['topic']}"

Historical Context:
{state['context']}

Current News Context:
{state['news']}

Evaluate based on:

1. Is it actually funny or just generic?
2. Does it fit the target topic?
3. Does it use the provided news or historical context well?
4. Does it feel like an actual internet shitpost?
5. Is there a clear comedic idea?

Return ONLY valid JSON.

Use exactly this structure:

{{
    "passed": true,
    "feedback": "Detailed string explaining why it passed or what needs to change"
}}
"""

    response = generate(prompt).strip()

    try:
        critique = json.loads(response)

    except json.JSONDecodeError:
        print("\nWARNING: Critic returned invalid JSON.")
        print("Raw response:")
        print(response)

        critique = {
            "passed": False,
            "feedback": (
                "The critic returned invalid JSON. "
                "Regenerate the post and improve its humor "
                "and use of context."
            ),
        }

    print("\n===== CRITIQUE =====")
    print(json.dumps(critique, indent=2))

    return {
        "critique": critique
    }


# ============================================================
# 9. ROUTING
# ============================================================

def route_after_critique(state: AgentState):
    """
    Decide whether to accept the generated post
    or regenerate it based on critic feedback.
    """

    critique = state.get("critique", {})
    attempts = state.get("attempts", 0)

    # Stop after 3 attempts to avoid an infinite loop.
    if attempts >= 3:
        print("\nMaximum generation attempts reached.")
        return "end"

    if critique.get("passed", False):
        print("\nCritic approved the post.")
        return "end"

    print("\nCritic rejected the post. Regenerating...")
    return "regenerate"


# ============================================================
# 10. BUILD GRAPH
# ============================================================

workflow = StateGraph(AgentState)

workflow.add_node(
    "classify_topic",
    classify_topic,
)

workflow.add_node(
    "fetch_news",
    fetch_news,
)

workflow.add_node(
    "retrieve",
    retrieve,
)

workflow.add_node(
    "generate_post",
    generate_post,
)

workflow.add_node(
    "critique_post",
    critique_post,
)


# ------------------------------------------------------------
# Graph edges
# ------------------------------------------------------------

workflow.add_edge(
    START,
    "classify_topic",
)

workflow.add_edge(
    "classify_topic",
    "fetch_news",
)

workflow.add_edge(
    "fetch_news",
    "retrieve",
)

workflow.add_edge(
    "retrieve",
    "generate_post",
)

workflow.add_edge(
    "generate_post",
    "critique_post",
)


# ------------------------------------------------------------
# Critic routing
# ------------------------------------------------------------

workflow.add_conditional_edges(
    "critique_post",
    route_after_critique,
    {
        "regenerate": "generate_post",
        "end": END,
    },
)


# Compile the graph
graph = workflow.compile()


# ============================================================
# 11. RUN GRAPH
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("AI SHITPOST GENERATOR")
    print("=" * 60)

    topic = input("\nEnter a topic: ").strip()

    if not topic:
        print("No topic entered. Exiting.")
        raise SystemExit

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

    print("\nStarting agent...\n")

    result = graph.invoke(initial_state)

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    print("\nTopic:")
    print(result["topic"])

    print("\nFinal Post:")
    print(result["draft"])

    print("\nCritique:")
    print(
        json.dumps(
            result["critique"],
            indent=2,
        )
    )

    print("\nAttempts:")
    print(result["attempts"])

    print("\n" + "=" * 60)