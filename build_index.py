import os
import glob
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def clean_text(text: str) -> str:
    text = text.replace("\x0c", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


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

    # Lọc chunk quá ngắn
    cleaned = []
    for m in matches:
        m = m.strip()
        if len(m) > 80:  # tránh chunk rác
            cleaned.append(m)

    return cleaned if cleaned else []
