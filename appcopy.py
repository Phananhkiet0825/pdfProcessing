import os
import re
import time
import unicodedata
import numpy as np
from flask import Flask, render_template, request, redirect, send_from_directory
from werkzeug.utils import secure_filename
from fallback_search import *  # Assuming your fallback_search.py is available

from pytesseract import image_to_string
from pdf2image import convert_from_path
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
TEXT_FOLDER = os.path.join('dataset', 'BGD&DT.text')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'pdf'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def normalize_filename(name):
    nfkd = unicodedata.normalize('NFKD', name)
    return ''.join(c for c in nfkd if not unicodedata.combining(c)) \
        .replace('-', '').replace('_', '').replace(' ', '').lower()


def find_matching_txt(pdf_filename):
    base_name = os.path.splitext(pdf_filename)[0]
    norm_pdf = normalize_filename(base_name)
    candidates = [f for f in os.listdir(TEXT_FOLDER) if f.endswith(".txt")]

    # 1. Exact normalized name match
    for fname in candidates:
        norm_txt = normalize_filename(fname)
        if norm_pdf == norm_txt:
            return os.path.join(TEXT_FOLDER, fname)

    # 2. Match if all key digits (e.g., 287, 2019) exist in txt name
    digit_match = re.findall(r"\d{2,4}", base_name)
    if digit_match:
        for fname in candidates:
            if all(d in fname for d in digit_match):
                return os.path.join(TEXT_FOLDER, fname)

    # 3. Match if the TXT filename starts with the same number
    for fname in candidates:
        txt_stem = normalize_filename(os.path.splitext(fname)[0])
        if txt_stem.startswith(norm_pdf[:8]):
            return os.path.join(TEXT_FOLDER, fname)

    return None

import re
import re

def extract_metadata_and_detail(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip().replace("\x0b", "").replace("\x07", "") for line in f if line.strip()]

    # Initialize fields
    cqbh = shvb = lvb = td = detail = noi_nhan = nguoi_ky = ngay_ban_hanh = ""
    qh = "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM"
    tn = "Độc lập - Tự do - Hạnh phúc"
    CQBH_KEYWORDS = ["BỘ", "CỤC", "VĂN PHÒNG", "SỞ", "TỔNG CỤC", "BAN"]
    DOCUMENT_TYPES = ["THÔNG TƯ", "QUYẾT ĐỊNH", "NGHỊ ĐỊNH", "CÔNG VĂN", "CHỈ THỊ"]

    td_line = ""
    content_lines = []
    td_index = None
    start_tail = False
    tail_lines = []

    for i, line in enumerate(lines):
        uline = line.upper()

        # Cơ quan ban hành
        if not cqbh:
            for keyword in CQBH_KEYWORDS:
                if uline.startswith(keyword):
                    cqbh = line
                    break

        # Số hiệu văn bản
        if not shvb and ("Số:" in line or line.startswith("Số ")):
            shvb = line.strip()

        # Ngày ban hành (with location)
        if not ngay_ban_hanh:
            date_match = re.search(
                r'([^\n,]*?,\s*)?ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}',
                line.lower()
            )
            if date_match:
                ngay_ban_hanh = line.strip()

        # Loại văn bản
        if not lvb:
            for doc_type in DOCUMENT_TYPES:
                if doc_type in uline:
                    lvb = doc_type
                    if i + 1 < len(lines):
                        td_line = lines[i + 1].strip()
                        td_index = i + 1
                    break

        # Trích yếu nội dung
        if "TRÍCH YẾU" in uline or "TRÍCH YẾU NỘI DUNG" in uline:
            for j in range(i + 1, len(lines)):
                if lines[j].strip():
                    td_line = lines[j].strip()
                    td_index = j
                    break

    # Extract nội dung chính and tail
    if td_index is not None:
        for line in lines[td_index + 1:]:
            if any(kw in line for kw in ["Nơi nhận", "KT. BỘ TRƯỞNG", "BỘ TRƯỞNG", "THỨ TRƯỞNG", "NGƯỜI KÝ"]):
                start_tail = True
            if start_tail:
                tail_lines.append(line)
            else:
                content_lines.append(line)

    td = td_line
    detail = "\n".join(content_lines).strip()

    # Process tail for "Nơi nhận"
    noi_nhan_started = False
    nguoi_ky_started = False
    noi_nhan_lines = []
    nguoi_ky_lines = []

    for line in tail_lines:
        if "Nơi nhận" in line:
            noi_nhan_started = True
        elif "Người ký" in line:
            nguoi_ky_started = True
            noi_nhan_started = False
            continue

        if noi_nhan_started:
            noi_nhan_lines.append(line)

        elif nguoi_ky_started:
            nguoi_ky_lines.append(line)

    # Fallback for người ký
    if not nguoi_ky_lines:
        for i, line in enumerate(tail_lines):
            if re.match(r"^(KT\.\s*)?BỘ TRƯỞNG|THỨ TRƯỞNG", line.upper()):
                nguoi_ky_lines = tail_lines[i:]
                break

    # Assign final fields
    noi_nhan = "\n".join(noi_nhan_lines).strip()
    nguoi_ky = "\n".join(nguoi_ky_lines).strip()

    return {
        'qh': qh,
        'tn': tn,
        'cqbh': cqbh,
        'shvb': shvb,
        'ngay_ban_hanh': ngay_ban_hanh,
        'lvb': lvb,
        'td': td,
        'detail': detail,
        'noi_nhan': noi_nhan,
        'nguoi_ky': nguoi_ky
    }



def fallback_ocr_if_no_txt(pdf_path, text_folder):
    images = convert_from_path(pdf_path)
    image = images[0]  # only first page
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    text = image_to_string(image, lang='vie')

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    txt_path = os.path.join(text_folder, f"{base_name}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)
    return txt_path


@app.route("/", methods=["GET", "POST"])
def index():
    data = None
    pdf_filename = None

    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)

            matched_txt = find_matching_txt(filename)
            if matched_txt:
                time.sleep(2)
                data = extract_metadata_and_detail(matched_txt)
            else:
                ocr_txt_path = fallback_ocr_if_no_txt(save_path, TEXT_FOLDER)
                data = extract_metadata_and_detail(ocr_txt_path) if ocr_txt_path else {
                    "td": "Không tìm thấy văn bản phù hợp", "detail": ""
                }

            pdf_filename = filename
    time.sleep(2)
    return render_template("index.html", data=data, pdf_filename=pdf_filename)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(TEXT_FOLDER, exist_ok=True)
    app.run(debug=True)
