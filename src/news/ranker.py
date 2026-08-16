from datetime import datetime, timezone
from math import exp
import re

from sentence_transformers import SentenceTransformer

from src.tools.news import NewsArticle


MODEL_NAME = "BAAI/bge-small-en-v1.5"


class NewsRanker:
    """
    Rank news articles using multiple signals:

        1. Semantic relevance
        2. Freshness
        3. Source quality
        4. Query/title relevance
        5. Newsworthiness / current-event signal

    Then apply:

        - duplicate-story filtering
        - source diversity

    The goal is not simply to find articles that mention the topic.
    The goal is to find articles that are:

        - actually about the topic
        - currently relevant
        - reporting something that happened
        - from reasonably trustworthy sources
        - not generic guides/opinions/old stories
    """

    # ============================================================
    # SOURCE QUALITY
    # ============================================================

    SOURCE_QUALITY = {
        # Major general/news organizations
        "reuters": 1.00,
        "associated press": 1.00,
        "ap news": 1.00,

        "bbc": 0.98,
        "bbc news": 0.98,

        "the guardian": 0.95,
        "the new york times": 0.95,
        "new york times": 0.95,
        "financial times": 0.95,
        "the washington post": 0.95,

        "cnn": 0.93,
        "abc news": 0.93,
        "cbs news": 0.93,
        "nbc news": 0.93,

        "al jazeera": 0.90,

        # Technology / gaming
        "ign": 0.90,
        "ign india": 0.90,
        "gamespot": 0.90,
        "the verge": 0.90,
        "ars technica": 0.90,
        "wired": 0.90,

        # Other established publications
        "yahoo news": 0.85,
        "yahoo sports": 0.80,
        "gulf news": 0.82,
        "the jerusalem post": 0.82,
        "nature": 0.95,

        # Indian publications
        "the indian express": 0.88,
        "indian express": 0.88,
        "ndtv": 0.88,
        "the hindu": 0.92,
        "hindustan times": 0.85,
        "times of india": 0.82,
    }

    # ============================================================
    # NEWSWORTHINESS
    # ============================================================

    CURRENT_EVENT_TERMS = {
        "announces",
        "announced",
        "announcement",

        "launches",
        "launched",
        "launch",

        "releases",
        "released",
        "release",

        "reveals",
        "revealed",
        "reveal",

        "unveils",
        "unveiled",
        "unveil",

        "confirms",
        "confirmed",
        "confirmation",

        "reports",
        "reported",
        "report",

        "says",
        "said",

        "update",
        "updates",
        "updated",

        "breaking",
        "breaks",

        "new",
        "newly",

        "develops",
        "developed",
        "development",
        "developing",

        "signs",
        "signed",

        "deal",
        "agreement",

        "acquires",
        "acquired",
        "acquisition",

        "approves",
        "approved",
        "approval",

        "rejects",
        "rejected",

        "resigns",
        "resigned",
        "resignation",

        "wins",
        "won",

        "loses",
        "lost",

        "elected",
        "election",

        "votes",
        "voted",

        "investigation",
        "investigates",
        "investigated",

        "lawsuit",
        "sues",
        "sued",

        "ban",
        "banned",

        "recall",
        "recalled",

        "opens",
        "opened",

        "closes",
        "closed",

        "crashes",
        "crashed",

        "outage",
        "outages",

        "earnings",
        "profit",
        "profits",

        "funding",
        "investment",

        "researchers",
        "study",
        "finds",
        "found",

        "discovery",
        "discovers",
        "discovered",

        "breakthrough",

        "leak",
        "leaked",
        "leaks",

        "trailer",
        "footage",
        "gameplay",

        "beta",
        "preview",

        "dies",
        "died",
        "death",

        "arrested",
        "arrest",
        "charged",

        "appointed",
        "appointment",

        "withdraws",
        "withdrawn",

        "extends",
        "extended",

        "delays",
        "delayed",

        "cancels",
        "cancelled",
        "canceled",

        "expands",
        "expanded",

        "cuts",
        "cut",

        "raises",
        "raised",

        "falls",
        "fell",

        "surges",
        "surged",

        "warns",
        "warned",
    }

    # ============================================================
    # EVERGREEN / INFORMATIONAL
    # ============================================================

    EVERGREEN_TERMS = {
        "what is",
        "what are",
        "what does",
        "how does",
        "how to",

        "guide",
        "guides",

        "beginners guide",
        "beginner's guide",

        "explained",
        "explainer",

        "everything you need to know",
        "all you need to know",

        "history of",

        "introduction to",
        "introduction",

        "overview",

        "definition",
        "meaning",

        "basics",
        "fundamentals",

        "tips",
        "tutorial",
        "tutorials",

        "best ways",

        "why it matters",
        "why does",

        "what we know",
        "what we know so far",

        "things to know",

        "complete guide",

        "ultimate guide",

        "beginner",

        "how to use",

        "how it works",

        "explained simply",
    }

    # ============================================================
    # OPINION / ANALYSIS
    # ============================================================

    ANALYSIS_TERMS = {
        "opinion",
        "analysis",
        "commentary",
        "editorial",
        "column",
        "perspective",

        "could",
        "might",
        "may",

        "theory",
        "speculation",

        "predicts",
        "prediction",

        "what to expect",

        "why",

        "should",

        "review",
        "reviews",

        "hands on",
        "hands-on",

        "first impressions",

        "thoughts",

        "i hope",

        "we think",

        "our take",

        "explained",
    }

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(
        self,
        similarity_threshold: float = 0.80,
        max_articles_per_source: int = 3,

        semantic_weight: float = 0.45,
        freshness_weight: float = 0.15,
        source_weight: float = 0.10,
        title_weight: float = 0.15,
        newsworthiness_weight: float = 0.15,
    ):
        self.model = SentenceTransformer(MODEL_NAME)

        self.similarity_threshold = similarity_threshold
        self.max_articles_per_source = max_articles_per_source

        self.semantic_weight = semantic_weight
        self.freshness_weight = freshness_weight
        self.source_weight = source_weight
        self.title_weight = title_weight
        self.newsworthiness_weight = newsworthiness_weight

    # ============================================================
    # TEXT
    # ============================================================

    def _article_text(
        self,
        article: NewsArticle,
    ) -> str:
        """
        Build the text representation used for semantic ranking
        and duplicate detection.
        """

        parts = [
            article.title,
            article.summary,
        ]

        return " ".join(
            part.strip()
            for part in parts
            if part and part.strip()
        )

    # ============================================================
    # NORMALIZATION
    # ============================================================

    def _normalize_text(
        self,
        text: str,
    ) -> str:
        """
        Normalize text for lexical matching.
        """

        text = text.lower()

        text = re.sub(
            r"[^\w\s'-]",
            " ",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    # ============================================================
    # TOKENIZATION
    # ============================================================

    def _tokens(
        self,
        text: str,
    ) -> list[str]:
        """
        Return normalized word tokens.
        """

        normalized = self._normalize_text(text)

        if not normalized:
            return []

        return normalized.split()

    # ============================================================
    # EMBEDDINGS
    # ============================================================

    def _embed(
        self,
        texts: list[str],
    ):
        """
        Generate normalized embeddings.
        """

        return self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=True,
        )

    # ============================================================
    # FRESHNESS
    # ============================================================

    def _freshness_score(
        self,
        published,
    ) -> float:
        """
        Convert publication time into a freshness score.

        Approximate values with a 48-hour half-life:

            0 hours  -> 1.00
            12 hours -> 0.84
            24 hours -> 0.71
            48 hours -> 0.50
            72 hours -> 0.35
            7 days   -> 0.13
        """

        if published is None:
            return 0.50

        try:
            if isinstance(published, str):
                published = datetime.fromisoformat(
                    published.replace(
                        "Z",
                        "+00:00",
                    )
                )

            if published.tzinfo is None:
                published = published.replace(
                    tzinfo=timezone.utc
                )

            now = datetime.now(timezone.utc)

            age_hours = max(
                0.0,
                (
                    now - published
                ).total_seconds() / 3600,
            )

            decay = 0.693 / 48.0

            return exp(
                -decay * age_hours
            )

        except (
            TypeError,
            ValueError,
            AttributeError,
        ):
            return 0.50

    # ============================================================
    # SOURCE QUALITY
    # ============================================================

    def _source_quality(
        self,
        source: str,
    ) -> float:
        """
        Return source quality.

        Unknown sources receive 0.50 rather than 0.70 so that an
        unknown website does not receive a strong prior advantage.
        """

        if not source:
            return 0.50

        normalized = self._normalize_text(
            source
        )

        # Exact match first.
        if normalized in self.SOURCE_QUALITY:
            return self.SOURCE_QUALITY[
                normalized
            ]

        # Handle source names such as:
        #
        # "ABC News - Breaking News, Latest News and Videos"
        #
        # where the actual publisher is at the beginning.
        for known_source, score in self.SOURCE_QUALITY.items():

            if normalized.startswith(
                known_source
            ):
                return score

        return 0.50

    # ============================================================
    # TITLE / QUERY RELEVANCE
    # ============================================================

    def _title_relevance(
        self,
        query: str,
        title: str,
    ) -> float:
        """
        Calculate lexical relevance between query and title.

        Unlike the original implementation, this considers:

            - exact phrase match
            - token coverage
            - multi-word entity matching
            - whether the query dominates the headline

        This prevents:

            "Donald Trump"

        from treating:

            "What Donald Trump said about Arsenal in 2018"

        as equally direct as a headline actually reporting a
        current Trump event.
        """

        query_normalized = self._normalize_text(
            query
        )

        title_normalized = self._normalize_text(
            title
        )

        if not query_normalized or not title_normalized:
            return 0.0

        # --------------------------------------------------------
        # EXACT PHRASE
        # --------------------------------------------------------

        exact_phrase = (
            query_normalized
            in title_normalized
        )

        query_tokens = self._tokens(
            query_normalized
        )

        title_tokens = self._tokens(
            title_normalized
        )

        if not query_tokens:
            return 0.0

        title_token_set = set(
            title_tokens
        )

        # --------------------------------------------------------
        # TOKEN COVERAGE
        # --------------------------------------------------------

        matched_tokens = sum(
            1
            for token in query_tokens
            if token in title_token_set
        )

        token_coverage = (
            matched_tokens
            / len(query_tokens)
        )

        # --------------------------------------------------------
        # ORDERED MATCH
        # --------------------------------------------------------

        ordered_match = False

        if len(query_tokens) > 1:

            for start in range(
                len(title_tokens)
                - len(query_tokens)
                + 1
            ):
                window = title_tokens[
                    start:
                    start + len(query_tokens)
                ]

                if window == query_tokens:
                    ordered_match = True
                    break

        # --------------------------------------------------------
        # QUERY SIZE / TITLE DOMINANCE
        # --------------------------------------------------------

        query_length = len(query_tokens)

        if query_length == 1:
            dominance = (
                1.0
                if query_tokens[0]
                in title_token_set
                else 0.0
            )
        else:
            dominance = min(
                1.0,
                query_length
                / max(
                    len(title_tokens),
                    1,
                ),
            )

        # --------------------------------------------------------
        # SCORE
        # --------------------------------------------------------

        if exact_phrase:
            score = 1.0

        else:
            score = (
                0.60 * token_coverage
                + 0.25 * (
                    1.0
                    if ordered_match
                    else 0.0
                )
                + 0.15 * dominance
            )

        return max(
            0.0,
            min(
                1.0,
                score,
            ),
        )

    # ============================================================
    # NEWSWORTHINESS
    # ============================================================

    def _newsworthiness_score(
        self,
        article: NewsArticle,
    ) -> float:
        """
        Estimate whether an article is reporting a current event
        rather than primarily being evergreen, opinion, theory,
        review, or educational content.

        This is a heuristic signal, not a factual classifier.

        Approximate interpretation:

            0.85 - 1.00 -> strong current-event signal
            0.65 - 0.84 -> normal reporting
            0.45 - 0.64 -> neutral
            0.25 - 0.44 -> mostly evergreen/analysis
            0.00 - 0.24 -> strongly non-news / evergreen
        """

        title = self._normalize_text(
            article.title or ""
        )

        summary = self._normalize_text(
            article.summary or ""
        )

        text = f"{title} {summary}"

        title_words = set(
            title.split()
        )

        text_words = set(
            text.split()
        )

        score = 0.50

        # --------------------------------------------------------
        # CURRENT EVENT SIGNAL
        # --------------------------------------------------------

        current_event_hits = 0

        for term in self.CURRENT_EVENT_TERMS:

            if " " in term:

                if term in title:
                    current_event_hits += 2

                elif term in text:
                    current_event_hits += 1

            else:

                if term in title_words:
                    current_event_hits += 2

                elif term in text_words:
                    current_event_hits += 1

        current_event_hits = min(
            current_event_hits,
            6,
        )

        score += (
            current_event_hits
            * 0.075
        )

        # --------------------------------------------------------
        # EVERGREEN PENALTY
        # --------------------------------------------------------

        evergreen_hits = 0

        for term in self.EVERGREEN_TERMS:

            if term in title:
                evergreen_hits += 2

            elif term in text:
                evergreen_hits += 1

        evergreen_hits = min(
            evergreen_hits,
            5,
        )

        score -= (
            evergreen_hits
            * 0.10
        )

        # --------------------------------------------------------
        # ANALYSIS / OPINION PENALTY
        # --------------------------------------------------------

        analysis_hits = 0

        for term in self.ANALYSIS_TERMS:

            if " " in term:

                if term in title:
                    analysis_hits += 2

                elif term in text:
                    analysis_hits += 1

            else:

                if term in title_words:
                    analysis_hits += 2

                elif term in text_words:
                    analysis_hits += 1

        analysis_hits = min(
            analysis_hits,
            5,
        )

        score -= (
            analysis_hits
            * 0.065
        )

        # --------------------------------------------------------
        # QUESTION HEADLINE
        # --------------------------------------------------------

        if "?" in (
            article.title or ""
        ):
            score -= 0.06

        # --------------------------------------------------------
        # EXPLAINER-STYLE HEADLINES
        # --------------------------------------------------------

        if re.match(
            r"^(what|how|why)\b",
            title,
        ):
            score -= 0.10

        # --------------------------------------------------------
        # HISTORICAL SIGNAL
        # --------------------------------------------------------

        # Headlines explicitly discussing old events are less
        # useful when the user asks for current news.
        if re.search(
            r"\b(19|20)\d{2}\b",
            title,
        ):
            score -= 0.08

        # --------------------------------------------------------
        # REVIEW / WAITING / LISTICLE SIGNAL
        # --------------------------------------------------------

        listicle_patterns = [
            "games to play",
            "things to know",
            "best games",
            "best ways",
            "top games",
            "top reasons",
            "reasons why",
            "alternatives",
            "movies to watch",
            "shows to watch",
        ]

        for pattern in listicle_patterns:
            if pattern in title:
                score -= 0.10
                break

        # --------------------------------------------------------
        # CLAMP
        # --------------------------------------------------------

        return max(
            0.0,
            min(
                1.0,
                score,
            ),
        )

    # ============================================================
    # FINAL SCORE
    # ============================================================

    def _final_score(
        self,
        semantic_score: float,
        freshness_score: float,
        source_score: float,
        title_score: float,
        newsworthiness_score: float,
    ) -> float:
        """
        Combine all ranking signals.
        """

        return (
            self.semantic_weight
            * semantic_score

            + self.freshness_weight
            * freshness_score

            + self.source_weight
            * source_score

            + self.title_weight
            * title_score

            + self.newsworthiness_weight
            * newsworthiness_score
        )

    # ============================================================
    # DIVERSITY
    # ============================================================

    def _select_diverse_articles(
        self,
        ranked_articles,
        article_embeddings,
        top_k: int,
    ):
        """
        Select highly relevant articles while avoiding:

            - near-duplicate stories
            - excessive repetition from one source
        """

        selected = []

        selected_embeddings = []

        source_counts = {}

        for (
            score,
            article,
            index,
            details,
        ) in ranked_articles:

            source = (
                self._normalize_text(
                    article.source
                    or "unknown"
                )
            )

            current_source_count = (
                source_counts.get(
                    source,
                    0,
                )
            )

            # ----------------------------------------------------
            # SOURCE DIVERSITY
            # ----------------------------------------------------

            if (
                current_source_count
                >= self.max_articles_per_source
            ):
                continue

            candidate_embedding = (
                article_embeddings[index]
            )

            # ----------------------------------------------------
            # STORY DUPLICATE DETECTION
            # ----------------------------------------------------

            is_duplicate = False

            for selected_embedding in (
                selected_embeddings
            ):

                similarity = float(
                    candidate_embedding
                    @ selected_embedding
                )

                if (
                    similarity
                    >= self.similarity_threshold
                ):
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            # ----------------------------------------------------
            # SELECT
            # ----------------------------------------------------

            selected.append(
                (
                    score,
                    article,
                    details,
                )
            )

            selected_embeddings.append(
                candidate_embedding
            )

            source_counts[source] = (
                current_source_count + 1
            )

            if len(selected) >= top_k:
                break

        return selected

    # ============================================================
    # MAIN RANKING
    # ============================================================

    def rank(
        self,
        query: str,
        articles: list[NewsArticle],
        top_k: int = 10,
    ):
        """
        Rank articles using multiple signals and then apply
        diversity filtering.

        Returns:

            [
                (
                    final_score,
                    article,
                    score_details,
                ),
                ...
            ]
        """

        if not articles:
            return []

        query = query.strip()

        if not query:
            return []

        # --------------------------------------------------------
        # BUILD ARTICLE TEXT
        # --------------------------------------------------------

        texts = [
            self._article_text(article)
            for article in articles
        ]

        print(
            "Embedding query and articles..."
        )

        # --------------------------------------------------------
        # QUERY EMBEDDING
        # --------------------------------------------------------

        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )

        # --------------------------------------------------------
        # ARTICLE EMBEDDINGS
        # --------------------------------------------------------

        article_embeddings = self._embed(
            texts
        )

        # --------------------------------------------------------
        # SEMANTIC SIMILARITY
        # --------------------------------------------------------

        semantic_scores = (
            article_embeddings
            @ query_embedding
        )

        ranked = []

        for index, (
            article,
            semantic_score,
        ) in enumerate(
            zip(
                articles,
                semantic_scores,
            )
        ):

            semantic_score = float(
                semantic_score
            )

            # ----------------------------------------------------
            # FRESHNESS
            # ----------------------------------------------------

            freshness_score = (
                self._freshness_score(
                    article.published
                )
            )

            # ----------------------------------------------------
            # SOURCE QUALITY
            # ----------------------------------------------------

            source_score = (
                self._source_quality(
                    article.source
                )
            )

            # ----------------------------------------------------
            # TITLE RELEVANCE
            # ----------------------------------------------------

            title_score = (
                self._title_relevance(
                    query,
                    article.title or "",
                )
            )

            # ----------------------------------------------------
            # NEWSWORTHINESS
            # ----------------------------------------------------

            newsworthiness_score = (
                self._newsworthiness_score(
                    article
                )
            )

            # ----------------------------------------------------
            # FINAL SCORE
            # ----------------------------------------------------

            final_score = self._final_score(
                semantic_score=semantic_score,
                freshness_score=freshness_score,
                source_score=source_score,
                title_score=title_score,
                newsworthiness_score=(
                    newsworthiness_score
                ),
            )

            details = {
                "semantic": semantic_score,
                "freshness": freshness_score,
                "source": source_score,
                "title": title_score,
                "newsworthiness": (
                    newsworthiness_score
                ),
            }

            ranked.append(
                (
                    final_score,
                    article,
                    index,
                    details,
                )
            )

        # --------------------------------------------------------
        # SORT
        # --------------------------------------------------------

        ranked.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        # --------------------------------------------------------
        # DIVERSITY FILTER
        # --------------------------------------------------------

        return self._select_diverse_articles(
            ranked_articles=ranked,
            article_embeddings=article_embeddings,
            top_k=top_k,
        )