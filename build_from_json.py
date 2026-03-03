import json
import numpy as np
import faiss

JSON_PATH = "BGD&DT.Vectordatabase.json"  # đổi đúng tên file của bạn
OUTPUT_INDEX = "text_vectors.faiss"

print("Loading JSON vector database...")

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

if "chunks" in data:
    chunks = data["chunks"]
else:
    chunks = []
    for doc in data:
        chunks.extend(doc["chunks"])

print("Total chunks:", len(chunks))

# ==============================
# Extract embeddings
# ==============================
embeddings = np.array([chunk["embedding"] for chunk in chunks], dtype="float32")

# Nếu embedding chưa normalize
faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]

# ==============================
# Build FAISS index
# ==============================
print("Building FAISS index...")

index = faiss.IndexFlatIP(dimension)  # cosine similarity
index.add(embeddings)

faiss.write_index(index, OUTPUT_INDEX)

print("FAISS index saved:", OUTPUT_INDEX)
print("DONE 🚀")
