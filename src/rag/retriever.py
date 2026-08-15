from pathlib import Path
import json

import faiss
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(
        self,
        index_path: str | Path,
        metadata_path: str | Path,
        model_name: str = "BAAI/bge-small-en-v1.5",
    ):
        self.model = SentenceTransformer(model_name,device="cuda")

        self.index = faiss.read_index(str(index_path))

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def search(self, query: str, k: int = 5):
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )

        scores, indices = self.index.search(
            query_embedding,
            k,
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):
            result = self.metadata[idx].copy()
            result["similarity"] = float(score)
            result["index"] = int(idx)

            results.append(result)

        return results