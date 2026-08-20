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

"{state['topic']}"

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
    # Convert ranked news into LLM context
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
    Generate a shitpost using:
    - historical Reddit examples
    - current news
    - previous critic feedback
    """

    attempts = state.get("attempts", 0) + 1

    print(
        f"\n===== GENERATION ATTEMPT #{attempts} ====="
    )

    previous_critique = state.get("critique", {})

    prompt = f"""
You are an expert internet shitposter.

Create ONE genuinely funny, internet-native shitpost about:

"{state['topic']}"

============================================================
HISTORICAL REDDIT STYLE CONTEXT
============================================================

{state['context']}

============================================================
CURRENT NEWS CONTEXT
============================================================

{state['news']}

============================================================
PREVIOUS CRITIC FEEDBACK
============================================================

{json.dumps(previous_critique, indent=2)}

============================================================
OBJECTIVE
============================================================

Create a post that feels like something an actual person
would post on Reddit.

The post should have:

- One clear comedic premise.
- A strong punchline or payoff.
- Natural internet/shitpost language.
- Humor rather than generic observations.
- Strong relevance to the requested topic.
- Original wording.
- A concise structure.
- No unnecessary explanations.

If current news is provided, use it only when it
actually improves the joke.

Do NOT simply summarize the news.

If previous critic feedback exists, actively fix the
specific problems identified by the critic.

============================================================
IMPORTANT
============================================================

Sensitive topics:

- Never refuse solely because the topic is political,
  controversial, tragic, religious, or otherwise sensitive.
- Always produce a post for the given topic.
- Do not use protected groups as the punchline.
- Do not promote hatred, dehumanization, or violence.
- Do not invent factual claims about real-world events.
- If source material contains inflammatory or hateful
  language, transform the underlying idea into safe satire.
- Never output an apology.
- Never output a refusal.
- Never output a safety disclaimer.
- Do not use emojis or emoticons.
- Do not add hashtags unless they are genuinely part
  of the joke.

Return ONLY the final post text.

Do not explain your reasoning.
Do not label it "Post:".
"""

    draft = generate(prompt).strip()

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
    Evaluate the generated post.

    The critic is intentionally strict so that weak,
    generic posts are regenerated.
    """

    print("\n===== CRITIQUE =====")

    prompt = f"""
You are an extremely strict Reddit shitpost critic.

Evaluate the following generated post:

------------------------------------------------------------
DRAFT
------------------------------------------------------------

{state['draft']}

------------------------------------------------------------
TARGET TOPIC
------------------------------------------------------------

{state['topic']}

------------------------------------------------------------
HISTORICAL CONTEXT
------------------------------------------------------------

{state['context']}

------------------------------------------------------------
CURRENT NEWS
------------------------------------------------------------

{state['news']}

------------------------------------------------------------
EVALUATION
------------------------------------------------------------

Judge the post on:

1. Humor
   - Is it actually funny?
   - Does it have a real punchline?
   - Is it more than a generic observation?

2. Topic relevance
   - Is the joke clearly about the requested topic?

3. Comedic premise
   - Is there ONE identifiable central joke?
   - Does the post build toward a payoff?

4. Internet authenticity
   - Does it sound like an actual Reddit shitpost?
   - Does it avoid sounding like an AI-generated essay?

5. Context usage
   - If useful news exists, did the post use it naturally?
   - If the news is irrelevant to the joke, do not penalize
     the post for not forcing it into the joke.

6. Originality
   - Does it avoid tired generic jokes?
   - Does it contain a fresh angle?

7. Conciseness
   - Is every line contributing to the joke?
   - Remove unnecessary filler and unrelated jokes.

============================================================
PASS CRITERIA
============================================================

Return "passed": true ONLY if the post is genuinely
good enough to publish.

Do NOT pass a weak post merely because it is grammatically
correct or technically related to the topic.

If the post is generic, predictable, incoherent,
overwritten, or lacks a punchline, reject it.

Return ONLY valid JSON.

Use exactly:

{{
    "passed": true,
    "feedback": "Short but specific explanation."
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
                "Critic returned invalid JSON. "
                "Generate a substantially better post."
            ),
        }

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
# 9. ROUTING AFTER CRITIC
# ============================================================

def route_after_critique(state: AgentState):
    """
    If the critic approves the post, finish.

    If the critic rejects it, regenerate.

    Maximum of 3 generation attempts.
    """

    critique = state.get("critique", {})
    attempts = state.get("attempts", 0)

    # --------------------------------------------------------
    # Approved
    # --------------------------------------------------------

    if critique.get("passed", False):

        print("\nCritic approved the post.")

        return "end"

    # --------------------------------------------------------
    # Maximum attempts reached
    # --------------------------------------------------------

    if attempts >= 3:

        print(
            "\nMaximum generation attempts reached."
        )

        return "end"

    # --------------------------------------------------------
    # Regenerate
    # --------------------------------------------------------

    print(
        "\nCritic rejected the post. "
        "Regenerating..."
    )

    return "regenerate"


# ============================================================
# 10. BUILD GRAPH
# ============================================================

workflow = StateGraph(AgentState)


# ------------------------------------------------------------
# Nodes
# ------------------------------------------------------------

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
# Initial flow
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


# ------------------------------------------------------------
# Compile
# ------------------------------------------------------------

app = workflow.compile()


# ============================================================
# 11. GENERATE SHITPOST FUNCTION
# ============================================================

def generate_shitpost(topic: str) -> dict:
    """
    Run the LangGraph pipeline for a given topic.
    This function is used by both the CLI and FastAPI.
    """

    topic = topic.strip()

    if not topic:
        raise ValueError("Topic cannot be empty")

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

    print("\nStarting agent...")
    print(f"Topic: {topic}")

    final_state = app.invoke(initial_state)

    return {
        "topic": final_state["topic"],
        "draft": final_state["draft"],
        "critique": final_state["critique"],
        "attempts": final_state["attempts"],
    }


# ============================================================
# 12. CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("AI SHITPOST GENERATOR")
    print("=" * 60)

    topic = input("\nEnter a topic: ").strip()

    if not topic:
        print("\nNo topic entered. Exiting.")
        raise SystemExit(1)

    result = generate_shitpost(topic)

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