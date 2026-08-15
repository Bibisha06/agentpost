from pathlib import Path

from retriever import Retriever


ROOT = Path(__file__).resolve().parents[2]

retriever = Retriever(
    index_path=ROOT / "data" / "embeddings" / "faiss.index",
    metadata_path=ROOT / "data" / "embeddings" / "metadata.json",
)

query = "funny programming jokes"

results = retriever.search(query, k=5)

for i, result in enumerate(results, start=1):
    print(f"\n#{i}")
    print(f"Similarity: {result['similarity']:.4f}")
    print(f"Source: {result['source']}")
    print(f"Subreddit: {result.get('subreddit')}")
    print(f"Score: {result.get('score')}")
    print(f"Text: {result['text']}")