import feedparser


FEEDS = {
    "BBC News": "https://feeds.bbci.co.uk/news/rss.xml",
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
}


for name, url in FEEDS.items():

    print("\n" + "=" * 60)
    print(f"TESTING: {name}")
    print(f"URL: {url}")
    print("=" * 60)

    try:
        feed = feedparser.parse(url)

        print(f"Status: {feed.get('status')}")
        print(f"Entries: {len(feed.entries)}")

        if feed.entries:
            article = feed.entries[0]

            print("\nLatest article:")
            print("Title:", article.get("title"))
            print("Link:", article.get("link"))

        else:
            print("No articles found.")

        if feed.bozo:
            print("\nWARNING:")
            print(feed.bozo_exception)

    except Exception as e:
        print(f"ERROR: {e}")