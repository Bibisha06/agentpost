from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"

model = SentenceTransformer(MODEL_NAME)

texts = [
    "Apple released a new iPhone",
    "The new iPhone launch is crazy",
    "I cooked pasta for dinner"
]

embeddings = model.encode(
    texts,
    normalize_embeddings=True
)

print("Shape:", embeddings.shape)
print("First vector:")
print(embeddings[0])