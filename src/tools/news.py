import calendar
import html
import re

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List
from urllib.parse import quote_plus

import feedparser


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class NewsArticle:
    title: str
    url: str
    source: str
    category: str
    summary: str = ""
    published: datetime | None = None


# ============================================================
# RSS SOURCES
# ============================================================

RSS_FEEDS = {
    "tech": {
        "Ars Technica": "https://feeds.arstechnica.com/arstechnica/index",
        "The Verge": "https://www.theverge.com/rss/index.xml",
        "TechCrunch": "https://techcrunch.com/feed/",
        "BBC News": "https://feeds.bbci.co.uk/news/technology/rss.xml",
    },
    "gaming": {
        "Polygon": "https://www.polygon.com/rss/index.xml",
        "GameSpot": "https://www.gamespot.com/feeds/mashup/",
        "Nintendo Life": "https://www.nintendolife.com/feeds/latest",
    },
    "entertainment": {
        "Variety": "https://variety.com/feed/",
        "Deadline": "https://deadline.com/feed",
        "Hollywood Reporter": "https://www.hollywoodreporter.com/feed/",
    },
    "business": {
        "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "TechCrunch": "https://techcrunch.com/feed/",
        "BBC News": "https://feeds.bbci.co.uk/news/business/rss.xml",
    },
    "politics": {
        "BBC News": "https://feeds.bbci.co.uk/news/politics/rss.xml",
    },
    "world": {
        "BBC News": "https://feeds.bbci.co.uk/news/world/rss.xml",
        "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    },
}


# ============================================================
# TEXT HELPERS
# ============================================================

def _clean_text(text: str) -> str:
    """Remove HTML and normalize whitespace."""

    if not text:
        return ""

    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _normalize(text: str) -> str:
    """Normalize text for lexical matching."""

    text = html.unescape(str(text)).lower()

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def _tokenize(text: str) -> list[str]:
    """Return normalized word tokens."""

    normalized = _normalize(text)

    if not normalized:
        return []

    return normalized.split()


# ============================================================
# DATE HELPERS
# ============================================================

def _parse_published(entry) -> datetime | None:
    """
    Extract publication time from a feedparser entry.
    """

    parsed = (
        entry.get("published_parsed")
        or entry.get("updated_parsed")
    )

    if parsed is not None:
        try:
            timestamp = calendar.timegm(parsed)

            return datetime.fromtimestamp(
                timestamp,
                tz=timezone.utc,
            )

        except (
            TypeError,
            ValueError,
            OverflowError,
        ):
            pass

    raw = (
        entry.get("published")
        or entry.get("updated")
        or ""
    )

    if raw:
        try:
            from email.utils import parsedate_to_datetime

            value = parsedate_to_datetime(raw)

            if value.tzinfo is None:
                value = value.replace(
                    tzinfo=timezone.utc
                )

            return value.astimezone(
                timezone.utc
            )

        except (
            TypeError,
            ValueError,
        ):
            pass

    return None


def _is_recent(
    article: NewsArticle,
    max_age_hours: int = 72,
) -> bool:
    """
    Keep recent articles.

    Articles without timestamps are retained because
    Google News does not always expose one.
    """

    if article.published is None:
        return True

    now = datetime.now(timezone.utc)

    age = now - article.published

    # Future timestamps can occasionally happen because of
    # feed clock issues. Do not reject them.
    if age.total_seconds() < 0:
        return True

    return age <= timedelta(
        hours=max_age_hours
    )


# ============================================================
# ARTICLE ID / DUPLICATION
# ============================================================

def _article_key(
    article: NewsArticle,
) -> str:
    """
    Stable key for exact duplicate removal.
    """

    if article.url:
        return article.url.strip().lower()

    return _normalize(article.title)


def _title_key(title: str) -> str:
    """
    Normalized title used for cheap duplicate detection.
    """

    return _normalize(title)


# ============================================================
# QUERY HELPERS
# ============================================================

# Common words which provide very little topical information.
_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "from",
    "by",
    "at",
    "is",
    "are",
    "was",
    "were",
    "be",
    "as",
    "it",
    "its",
    "this",
    "that",
    "latest",
    "news",
    "update",
    "updates",
}


# Explicit aliases for important entities.
QUERY_ALIASES = {
    "gta 6": {
        "gta 6",
        "gta vi",
        "grand theft auto 6",
        "grand theft auto vi",
    },
    "donald trump": {
        "donald trump",
        "trump",
    },
    "artificial intelligence": {
        "artificial intelligence",
        "ai",
    },
}


def _query_terms(
    query: str,
) -> list[str]:
    """
    Return meaningful normalized query terms.
    """

    terms = _tokenize(query)

    return [
        term
        for term in terms
        if term not in _STOPWORDS
    ]


def _query_aliases(
    query: str,
) -> set[str]:
    """
    Return aliases for a known query.
    """

    normalized = _normalize(query)

    return QUERY_ALIASES.get(
        normalized,
        {normalized},
    )


def _contains_alias(
    text: str,
    aliases: set[str],
) -> bool:
    """
    Check whether an alias occurs as a phrase.
    """

    normalized = _normalize(text)

    return any(
        alias in normalized
        for alias in aliases
    )


# ============================================================
# ARTICLE QUALITY FILTERS
# ============================================================

# These strongly suggest a page is not a concrete news report.
_LANDING_PAGE_PATTERNS = (
    r"\bbreaking news\b",
    r"\blatest news\b",
    r"\blive updates\b",
    r"\bnews and videos\b",
    r"\bnews hub\b",
    r"\bnews roundup\b",
    r"\bhomepage\b",
    r"\bhome page\b",
    r"\bcategory\b",
    r"\btopic page\b",
    r"\ball news\b",
    r"\blatest stories\b",
)

# These usually indicate evergreen/editorial content rather
# than a concrete current event.
_EVERGREEN_PATTERNS = (
    r"^what is\b",
    r"^what are\b",
    r"^how to\b",
    r"^how does\b",
    r"^why is\b",
    r"\bexplained\b",
    r"\bguide\b",
    r"\bbest .* to\b",
    r"\bgames to play\b",
    r"\bthings to know\b",
    r"\beverything you need to know\b",
    r"\bwhat you need to know\b",
)

# These are legitimate article types, but generally less useful
# for a "current news" feed.
_ANALYSIS_PATTERNS = (
    r"\bopinion\b",
    r"\banalysis\b",
    r"\bcommentary\b",
    r"\beditorial\b",
    r"\bcolumn\b",
    r"\btheory\b",
    r"\bprediction\b",
    r"\bpredictions\b",
)

# Listicles are often useful content but should not outrank
# actual breaking/current events.
_LISTICLE_PATTERN = re.compile(
    r"^\s*\d+\s+"
)


def _looks_like_landing_page(
    article: NewsArticle,
) -> bool:
    """
    Detect category/topic/landing pages.
    """

    title = _normalize(article.title)

    if not title:
        return True

    for pattern in _LANDING_PAGE_PATTERNS:
        if re.search(
            pattern,
            title,
        ):
            return True

    return False


def _looks_evergreen(
    article: NewsArticle,
) -> bool:
    """
    Detect generic evergreen/search-oriented articles.
    """

    title = _normalize(article.title)

    for pattern in _EVERGREEN_PATTERNS:
        if re.search(
            pattern,
            title,
        ):
            return True

    return False


def _looks_like_listicle(
    article: NewsArticle,
) -> bool:
    return bool(
        _LISTICLE_PATTERN.search(
            article.title
        )
    )


def _looks_like_analysis(
    article: NewsArticle,
) -> bool:
    title = _normalize(article.title)

    for pattern in _ANALYSIS_PATTERNS:
        if re.search(
            pattern,
            title,
        ):
            return True

    return False


def _has_meaningful_content(
    article: NewsArticle,
) -> bool:
    """
    Reject entries that contain essentially no usable content.
    """

    title = article.title.strip()
    summary = article.summary.strip()

    if len(title) < 10:
        return False

    # A title alone is acceptable for Google News.
    # If a summary exists, require it to contain real content.
    if summary and len(summary) < 20:
        return False

    return True


# ============================================================
# QUERY RELEVANCE
# ============================================================

def _matches_query(
    article: NewsArticle,
    query: str,
) -> bool:
    """
    Require meaningful query relevance.

    Rules:
      1. Known entities can use aliases.
      2. Otherwise all meaningful query terms must occur
         in title + summary.
      3. Prefer title matches when available.
    """

    title = _normalize(
        article.title
    )

    summary = _normalize(
        article.summary
    )

    text = f"{title} {summary}".strip()

    normalized_query = _normalize(
        query
    )

    if not normalized_query:
        return False

    aliases = _query_aliases(
        query
    )

    # --------------------------------------------------------
    # Known entity
    # --------------------------------------------------------

    if normalized_query in QUERY_ALIASES:
        return _contains_alias(
            text,
            aliases,
        )

    # --------------------------------------------------------
    # General query
    # --------------------------------------------------------

    terms = _query_terms(
        query
    )

    if not terms:
        return _contains_alias(
            text,
            aliases,
        )

    # If the exact query phrase appears, accept immediately.
    if normalized_query in title:
        return True

    # Otherwise require every meaningful term.
    return all(
        re.search(
            rf"\b{re.escape(term)}\b",
            text,
        )
        for term in terms
    )


# ============================================================
# FETCH CATEGORY NEWS
# ============================================================

def fetch_category(
    category: str,
    limit_per_source: int = 10,
) -> List[NewsArticle]:
    """
    Fetch recent articles from configured RSS feeds.
    """

    if category not in RSS_FEEDS:
        return []

    articles = []
    seen = set()

    for source, url in RSS_FEEDS[
        category
    ].items():

        try:
            feed = feedparser.parse(
                url
            )

            if getattr(
                feed,
                "bozo",
                False,
            ):
                print(
                    f"Warning: RSS feed may be malformed: {source}"
                )

            for entry in feed.entries[
                :limit_per_source
            ]:

                title = _clean_text(
                    entry.get(
                        "title",
                        "",
                    )
                )

                link = (
                    entry.get(
                        "link",
                        "",
                    )
                    .strip()
                )

                if not title or not link:
                    continue

                summary = _clean_text(
                    entry.get(
                        "summary",
                        "",
                    )
                )

                article = NewsArticle(
                    title=title,
                    url=link,
                    source=source,
                    category=category,
                    summary=summary,
                    published=_parse_published(
                        entry
                    ),
                )

                if not _has_meaningful_content(
                    article
                ):
                    continue

                if not _is_recent(
                    article
                ):
                    continue

                key = _article_key(
                    article
                )

                if key in seen:
                    continue

                seen.add(key)
                articles.append(
                    article
                )

        except Exception as e:
            print(
                f"Failed to fetch {source}: {e}"
            )

    return articles


def fetch_categories(
    categories: List[str],
    limit_per_source: int = 10,
) -> List[NewsArticle]:
    """
    Fetch news from multiple categories and
    deduplicate across feeds.
    """

    articles = []
    seen = set()

    for category in categories:

        category_articles = fetch_category(
            category,
            limit_per_source=limit_per_source,
        )

        for article in category_articles:

            key = _article_key(
                article
            )

            if key in seen:
                continue

            seen.add(key)
            articles.append(
                article
            )

    return articles


# ============================================================
# GOOGLE NEWS SEARCH
# ============================================================

def _build_google_news_url(
    query: str,
) -> str:
    """
    Build Google News RSS search URL.
    """

    encoded = quote_plus(
        query
    )

    return (
        "https://news.google.com/rss/search"
        f"?q={encoded}"
        "&hl=en-IN"
        "&gl=IN"
        "&ceid=IN:en"
    )


def fetch_news_for_query(
    query: str,
    limit: int = 50,
    max_age_hours: int = 72,
) -> List[NewsArticle]:
    """
    Fetch current news for a query.

    Pipeline:

        Google News
          ↓
        clean article
          ↓
        exact duplicate removal
          ↓
        query relevance
          ↓
        landing-page removal
          ↓
        low-quality content removal
          ↓
        recency filter
          ↓
        return candidates
    """

    query = query.strip()

    if not query:
        return []

    print(
        f"Fetching Google News results for: {query}"
    )

    url = _build_google_news_url(
        query
    )

    try:
        feed = feedparser.parse(
            url
        )

    except Exception as e:
        print(
            f"Failed to fetch Google News: {e}"
        )
        return []

    if getattr(
        feed,
        "bozo",
        False,
    ):
        print(
            "Warning: Google News RSS feed may be malformed."
        )

    articles = []

    seen_urls = set()
    seen_titles = set()

    # Inspect substantially more results than requested
    # because quality filters may remove many entries.
    entries = feed.entries[
        :max(
            limit * 5,
            100,
        )
    ]

    for entry in entries:

        title = _clean_text(
            entry.get(
                "title",
                "",
            )
        )

        link = (
            entry.get(
                "link",
                "",
            )
            .strip()
        )

        summary = _clean_text(
            entry.get(
                "summary",
                "",
            )
        )

        if not title or not link:
            continue

        source_data = entry.get(
            "source"
        )

        if isinstance(
            source_data,
            dict,
        ):
            source = (
                source_data.get(
                    "title"
                )
                or "Google News"
            )
        else:
            source = "Google News"

        article = NewsArticle(
            title=title,
            url=link,
            source=str(source).strip(),
            category="news",
            summary=summary,
            published=_parse_published(
                entry
            ),
        )

        # ----------------------------------------------------
        # Basic quality
        # ----------------------------------------------------

        if not _has_meaningful_content(
            article
        ):
            continue

        # ----------------------------------------------------
        # Exact URL duplicate
        # ----------------------------------------------------

        url_key = link.lower()

        if url_key in seen_urls:
            continue

        # ----------------------------------------------------
        # Exact normalized title duplicate
        # ----------------------------------------------------

        title_key = _title_key(
            title
        )

        if title_key in seen_titles:
            continue

        # ----------------------------------------------------
        # Query relevance
        # ----------------------------------------------------

        if not _matches_query(
            article,
            query,
        ):
            continue

        # ----------------------------------------------------
        # Landing page filter
        # ----------------------------------------------------

        if _looks_like_landing_page(
            article
        ):
            continue

        # ----------------------------------------------------
        # Recency
        # ----------------------------------------------------

        if not _is_recent(
            article,
            max_age_hours=max_age_hours,
        ):
            continue

        seen_urls.add(
            url_key
        )

        seen_titles.add(
            title_key
        )

        articles.append(
            article
        )

        if len(articles) >= limit:
            break

    # Newest first at the retrieval stage.
    # Final relevance ordering happens in NewsRanker.
    articles.sort(
        key=lambda article: (
            article.published
            or datetime.min.replace(
                tzinfo=timezone.utc
            )
        ),
        reverse=True,
    )

    print(
        f"Found {len(articles)} "
        f"query-relevant articles."
    )

    return articles


# ============================================================
# BACKWARDS-COMPATIBLE API
# ============================================================

def get_latest_news(
    categories: List[str] | None = None,
    limit_per_category: int = 5,
) -> list[dict]:
    """
    Existing category-based API used elsewhere
    in the project.

    Kept backwards compatible.
    """

    if not categories:
        return []

    articles = fetch_categories(
        categories,
        limit_per_source=limit_per_category,
    )

    return [
        {
            "title": article.title,
            "url": article.url,
            "source": article.source,
            "category": article.category,
            "summary": article.summary,
            "published": article.published,
        }
        for article in articles
    ]


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    query = input(
        "Search news for: "
    ).strip()

    articles = fetch_news_for_query(
        query,
        limit=20,
    )

    print(
        f"\nFetched {len(articles)} "
        f"relevant articles\n"
    )

    for i, article in enumerate(
        articles,
        start=1,
    ):

        print(
            f"#{i}"
        )

        print(
            f"Source: {article.source}"
        )

        print(
            f"Title: {article.title}"
        )

        print(
            f"Published: {article.published}"
        )

        print(
            f"URL: {article.url}"
        )

        print()