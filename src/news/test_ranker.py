from src.news.ranker import NewsRanker
from src.tools.news import fetch_news_for_query


def main():
    topic = input("Topic: ").strip()

    if not topic:
        print("No topic provided.")
        return

    print("\nFetching current news...")

    articles = fetch_news_for_query(
        topic,
        limit=50,
    )

    print(
        f"Fetched {len(articles)} "
        f"topic-relevant articles."
    )

    if not articles:
        print("No relevant articles found.")
        return

    print("\nLoading news embedding model...")

    ranker = NewsRanker()

    print("\nRanking articles...")

    ranked = ranker.rank(
        topic,
        articles,
        top_k=10,
    )

    if not ranked:
        print("No articles could be ranked.")
        return

    print("\n" + "=" * 60)
    print("MOST RELEVANT NEWS")
    print("=" * 60)

    for i, item in enumerate(ranked, start=1):

        # Expected format:
        # (score, article, details)
        if len(item) != 3:
            print(f"\n#{i}")
            print(f"Unexpected ranking result: {item}")
            continue

        score, article, details = item

        print(f"\n#{i}")

        print(
            f"Final Score: {float(score):.4f}"
        )

        # Safely display every known ranking component.
        # Missing components default to 0.0 instead of crashing.
        ranking_fields = [
            ("semantic", "Semantic"),
            ("freshness", "Freshness"),
            ("source", "Source"),
            ("title", "Title"),
            ("directness", "Directness"),
            ("newsworthiness", "Newsworthiness"),
        ]

        for key, label in ranking_fields:
            value = details.get(key)

            if value is not None:
                try:
                    print(
                        f"  {label + ':':<18}"
                        f"{float(value):.4f}"
                    )
                except (TypeError, ValueError):
                    print(
                        f"  {label + ':':<18}"
                        f"{value}"
                    )

        print(
            f"Title: {article.title}"
        )

        print(
            f"Source: {article.source}"
        )

        print(
            f"Published: {article.published}"
        )

        print(
            f"URL: {article.url}"
        )

        print()

    print("=" * 60)


if __name__ == "__main__":
    main()