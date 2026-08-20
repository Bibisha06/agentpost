from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

COMMENTS_PATH = ROOT / "data" / "processed" / "comments_clean.csv"
JOKES_PATH = ROOT / "data" / "processed" / "jokes_clean.csv"

OUTPUT_PATH = ROOT / "data" / "embeddings" / "metadata.json"


def main():
    print("Loading datasets...")

    comments = pd.read_csv(COMMENTS_PATH)
    jokes = pd.read_csv(JOKES_PATH)

    metadata = []


    for _, row in comments.iterrows():
        metadata.append({
            "text": row["body"],
            "source": "reddit_comment",
            "subreddit": row["subreddit"],
            "score": int(row["score"]),
            "controversiality": int(row["controversiality"]),
        })

    for _, row in jokes.iterrows():
        metadata.append({
            "text": row["text"],
            "source": "reddit_joke",
            "subreddit": row["subreddit.name"],
            "score": int(row["score"]),
            "nsfw": bool(row["subreddit.nsfw"]),
            "created_utc": int(row["created_utc"]),
        })

    print(f"Metadata entries: {len(metadata)}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False
        )

    print(f"Saved metadata to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()