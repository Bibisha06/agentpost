from pathlib import Path
import json

import faiss
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[2]

INDEX_PATH = ROOT / "data" / "embeddings" / "faiss.index"
METADATA_PATH = ROOT / "data" / "embeddings" / "metadata.json"

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def main():
    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Loading FAISS index...")
    index = faiss.read_index(str(INDEX_PATH))

    print("Loading metadata...")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    query = input("\nEnter your search query: ")

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    k = 10

    scores, indices = index.search(query_embedding, k)

    print("\n===== RESULTS =====")

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):
        result = metadata[idx]

        print(f"\n#{rank}")
        print(f"Similarity: {score:.4f}")
        print(f"Source: {result['source']}")
        print(f"Subreddit: {result.get('subreddit')}")
        print(f"Score: {result.get('score')}")
        print(f"Text: {result['text'][:500]}")


if __name__ == "__main__":
    main()