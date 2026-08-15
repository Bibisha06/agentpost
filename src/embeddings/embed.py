from pathlib import Path

import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 32

ROOT = Path(__file__).resolve().parents[2]

COMMENTS_PATH = ROOT / "data" / "processed" / "comments_clean.csv"
JOKES_PATH = ROOT / "data" / "processed" / "jokes_clean.csv"

OUTPUT_DIR = ROOT / "data" / "embeddings"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading datasets...")

    comments = pd.read_csv(COMMENTS_PATH)
    jokes = pd.read_csv(JOKES_PATH)


    comment_texts = comments["body"].fillna("").tolist()
    joke_texts = jokes["text"].fillna("").tolist()

    texts = comment_texts + joke_texts

    print(f"Comments: {len(comment_texts)}")
    print(f"Jokes: {len(joke_texts)}")
    print(f"Total texts: {len(texts)}")


    metadata = []

    # Metadata for comments
    for _, row in comments.iterrows():
        metadata.append({
            "text": row["body"],
            "source": "reddit_comment",
            "subreddit": row["subreddit"],
            "score": int(row["score"]),
            "controversiality": int(row["controversiality"]),
        })

    # Metadata for jokes
    for _, row in jokes.iterrows():
        metadata.append({
            "text": row["text"],
            "source": "reddit_joke",
            "score": int(row["score"]),
        })

    print(f"Metadata entries: {len(metadata)}")

    # Make sure everything lines up
    assert len(texts) == len(metadata)


    print("\nLoading embedding model...")

    model = SentenceTransformer(MODEL_NAME, device="cuda")
    print("Model device:", model.device)

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    print("\nEmbedding shape:", embeddings.shape)


    embeddings_path = OUTPUT_DIR / "embeddings.npy"

    np.save(embeddings_path, embeddings)

    print(f"Saved embeddings to: {embeddings_path}")


    metadata_path = OUTPUT_DIR / "metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()