import os
import time
import pandas as pd
from jiwer import wer, cer

# === Folders ===
ocr_result_folder = "./result_texts"       # OCR output .txt files
ground_truth_folder = "./ground_truths"    # Ground truth .txt files
output_csv = "evaluation_results.csv"

# === Process ===
results = []

for file in os.listdir(ocr_result_folder):
    if not file.endswith(".txt"):
        continue

    name = os.path.splitext(file)[0]
    result_path = os.path.join(ocr_result_folder, file)
    ground_truth_path = os.path.join(ground_truth_folder, file)

    # Check if matching ground truth exists
    if not os.path.exists(ground_truth_path):
        print(f"⚠️ Missing ground truth for: {file}")
        continue

    # Read files
    with open(result_path, "r", encoding="utf-8") as f:
        result_text = f.read().strip()

    with open(ground_truth_path, "r", encoding="utf-8") as f:
        ground_truth = f.read().strip()

    # Time measurement
    start = time.time()
    cer_score = cer(ground_truth, result_text)
    wer_score = wer(ground_truth, result_text)
    end = time.time()

    # Count pages (optional: based on manual record or filename)
    # Here we skip page count unless provided separately

    results.append({
        "Document": name,
        "CER": round(cer_score, 4),
        "WER": round(wer_score, 4),
        "Time (ms)": int((end - start) * 1000)
    })

# === Export ===
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False, encoding="utf-8-sig")
print("✅ Results saved to", output_csv)
