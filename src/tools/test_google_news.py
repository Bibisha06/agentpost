import feedparser


def main():

    queries = [
        "GTA 6",
        "Grand Theft Auto VI",
        "Rockstar Games",
    ]

    for query in queries:

        print("\n" + "=" * 60)
        print(f"QUERY: {query}")
        print("=" * 60)

        url = (
            "https://news.google.com/rss/search"
            f"?q={query.replace(' ', '+')}"
        )

        feed = feedparser.parse(url)

        print(f"Entries: {len(feed.entries)}")

        for i, entry in enumerate(feed.entries[:10], 1):

            print(f"\n#{i}")
            print(f"Title: {entry.get('title', '')}")
            print(f"Link: {entry.get('link', '')}")


if __name__ == "__main__":
    main()