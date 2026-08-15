from pathlib import Path

import faiss
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

EMBEDDINGS_PATH = ROOT / "data" / "embeddings" / "embeddings.npy"
INDEX_PATH = ROOT / "data" / "embeddings" / "faiss.index"


def main():
    print("Loading embeddings...")

    embeddings = np.load(EMBEDDINGS_PATH)

    print("Shape:", embeddings.shape)
    print("Dtype:", embeddings.dtype)

    # Our embeddings were normalized, so inner product
    # is equivalent to cosine similarity.
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    print("Adding vectors to FAISS...")

    index.add(embeddings)

    print("Total vectors:", index.ntotal)

    faiss.write_index(index, str(INDEX_PATH))

    print(f"Saved FAISS index to: {INDEX_PATH}")


if __name__ == "__main__":
    main()