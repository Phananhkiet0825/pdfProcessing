import os
import re
import glob
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ================= PATH =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Database", "BGD&DT.TXT")
INDEX_PATH = os.path.join(BASE_DIR, "text_vectors.faiss")
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

print("DATA_PATH:", DATA_PATH)

# ================= LOAD MODEL =================
print("🔹 Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)


# ================= TEXT PROCESSING =================
def clean_text(text: str) -> str:
    text = text.replace("\x0c", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_metadata(text: str):
    meta = {}

    so_hieu = re.search(
        r"(Thông tư|Nghị định|Quyết định)[^\n]{0,100}", text, re.IGNORECASE
    )
    trich_yeu = re.search(r"Về việc[^\n]{0,300}", text, re.IGNORECASE)

    meta["so_hieu"] = so_hieu.group(0).strip() if so_hieu else "Không rõ"
    meta["trich_yeu"] = trich_yeu.group(0).strip() if trich_yeu else "Không rõ"

    return meta


def split_legal_chunks(text: str):
    pattern = r"(Điều\s+\d+[\s\S]*?)(?=Điều\s+\d+|$)"
    matches = re.findall(pattern, text)

    cleaned = []
    for m in matches:
        m = m.strip()
        if len(m) > 80:  # loại chunk rác
            cleaned.append(m)

    return cleaned


# ================= LOAD DATASET =================
def load_dataset(folder_path):
    documents = []
    metadatas = []

    files = glob.glob(os.path.join(folder_path, "*.txt"))
    print("Files found:", len(files))

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()

        text = clean_text(text)
        base_meta = extract_metadata(text)
        chunks = split_legal_chunks(text)

        for idx, chunk in enumerate(chunks):
            documents.append(chunk)

            metadatas.append(
                {
                    "file": os.path.basename(file),
                    "chunk_id": idx,
                    "so_hieu": base_meta["so_hieu"],
                    "trich_yeu": base_meta["trich_yeu"],
                }
            )

    return documents, metadatas


# ================= BUILD FAISS =================
def build_faiss_index(documents):
    print("🔹 Encoding documents...")

    embeddings = model.encode(
        documents,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity

    index.add(embeddings)

    print("✅ FAISS index built:", index.ntotal)
    return index


# ================= LOAD OR CREATE INDEX =================
if os.path.exists(INDEX_PATH):
    print("🔹 Loading FAISS index from file...")
    index = faiss.read_index(INDEX_PATH)
    documents, metadatas = load_dataset(DATA_PATH)
else:
    print("🔹 Building new FAISS index...")
    documents, metadatas = load_dataset(DATA_PATH)
    index = build_faiss_index(documents)
    faiss.write_index(index, INDEX_PATH)
    print("✅ Index saved to", INDEX_PATH)


# ================= SEMANTIC SEARCH =================
def semantic_search(query, top_k=5):

    print("Query:", query)

    q_vec = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(q_vec, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < len(documents):
            results.append(
                {
                    "score": float(score),
                    "metadata": metadatas[idx],
                    "text": documents[idx],
                    "pdf_file": metadatas[idx]["file"].replace(".txt", ".pdf"),
                }
            )

    return results


# ================= TEST =================
if __name__ == "__main__":
    query = "bằng tốt nghiệp trung học phổ thông"
    results = semantic_search(query)

    for r in results:
        print("\n---")
        print(r["metadata"])
        print(r["text"])
