import os
import re
import time
import unicodedata
import numpy as np
from vector_search import semantic_search
from flask import Flask, render_template, request, redirect, send_from_directory
from werkzeug.utils import secure_filename
from fallback_search import *
from flask import send_from_directory, abort


from pdf2image import convert_from_path
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
TEXT_FOLDER = os.path.join(BASE_DIR, "Database", "BGD&DT.TXT")
MARKS_FOLDER = "marks_output"
PDF_FOLDER = os.path.join(BASE_DIR, "Database", "BGD&DT.PDF")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"pdf"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def normalize_filename(name):
    nfkd = unicodedata.normalize("NFKD", name)
    return (
        "".join(c for c in nfkd if not unicodedata.combining(c))
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
        .lower()
    )


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


def extract_metadata_and_detail(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [
            line.strip().replace("\x0b", "").replace("\x07", "")
            for line in f
            if line.strip()
        ]

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
                r"([^\n,]*?,\s*)?ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}",
                line.lower(),
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
        for line in lines[td_index + 1 :]:
            if any(
                kw in line
                for kw in [
                    "Nơi nhận",
                    "KT. BỘ TRƯỞNG",
                    "BỘ TRƯỞNG",
                    "THỨ TRƯỞNG",
                    "NGƯỜI KÝ",
                ]
            ):
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
        "qh": qh,
        "tn": tn,
        "cqbh": cqbh,
        "shvb": shvb,
        "ngay_ban_hanh": ngay_ban_hanh,
        "lvb": lvb,
        "td": td,
        "detail": detail,
        "noi_nhan": noi_nhan,
        "nguoi_ky": nguoi_ky,
    }


# ====== NEW: stamp + signature detection (returns filename saved in MARKS_FOLDER) ======
def merge_rects(rects, gap=10):
    """Merge overlapping / near rectangles. rect=(x,y,w,h)."""
    if not rects:
        return []
    boxes = [[x, y, x + w, y + h] for (x, y, w, h) in rects]
    merged = []
    used = [False] * len(boxes)
    for i, b in enumerate(boxes):
        if used[i]:
            continue
        x1, y1, x2, y2 = b
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j, bb in enumerate(boxes):
                if used[j]:
                    continue
                xx1, yy1, xx2, yy2 = bb
                if not (
                    xx1 > x2 + gap or xx2 + gap < x1 or yy1 > y2 + gap or yy2 + gap < y1
                ):
                    x1 = min(x1, xx1)
                    y1 = min(y1, yy1)
                    x2 = max(x2, xx2)
                    y2 = max(y2, yy2)
                    used[j] = True
                    changed = True
        merged.append((x1, y1, x2, y2))
    return merged


def detect_stamp_and_signature(pdf_path, output_dir=MARKS_FOLDER):
    """
    Detect red stamps and expand crop to capture nearby signature.
    Save a single combined PNG named <pdfbasename>_marks.png in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    out_filename = f"{base_name}_marks.png"
    out_path = os.path.join(output_dir, out_filename)

    pages = convert_from_path(pdf_path, dpi=200)
    crops = []

    for pi, page in enumerate(pages, start=1):
        page_cv = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
        h, w = page_cv.shape[:2]

        hsv = cv2.cvtColor(page_cv, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 80, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 80, 50])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), np.uint8)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(
            mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        rects = []
        for cnt in contours:
            x, y, rw, rh = cv2.boundingRect(cnt)
            area = rw * rh
            if area > 4000 and rw > 30 and rh > 30:
                expand_right = int(w * 0.2)
                expand_vert = int(rh * 1.5)
                x1 = max(0, x - 10)
                y1 = max(0, y - expand_vert)
                x2 = min(w, x + rw + expand_right)
                y2 = min(h, y + rh + expand_vert)
                rects.append((x1, y1, x2 - x1, y2 - y1))

        merged = merge_rects(rects, gap=30)

        for x1, y1, x2, y2 in merged:
            margin = 20
            xa = max(0, x1 - margin)
            ya = max(0, y1 - margin)
            xb = min(w, x2 + margin)
            yb = min(h, y2 + margin)
            crop = page_cv[ya:yb, xa:xb]
            if crop.size != 0:
                crops.append(crop)

    if not crops:
        return None

    max_w = max(c.shape[1] for c in crops)
    padded = []
    for c in crops:
        ch, cw = c.shape[:2]
        left = (max_w - cw) // 2
        right = max_w - cw - left
        cpad = cv2.copyMakeBorder(
            c, 10, 10, left + 10, right + 10, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        padded.append(cpad)

    combined = cv2.vconcat(padded)
    cv2.imwrite(out_path, combined)
    return out_filename


# ====== END NEW DETECTION CODE ======


@app.route("/", methods=["GET", "POST"])
def index():
    data_list = []

    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        files = request.files.getlist("file")

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(save_path)

                matched_txt = find_matching_txt(filename)
                if matched_txt:
                    data = extract_metadata_and_detail(matched_txt)
                else:
                    data = {
                        "td": "Không tìm thấy văn bản phù hợp",
                        "detail": "",
                        "qh": "",
                        "tn": "",
                        "cqbh": "",
                        "shvb": "",
                        "ngay_ban_hanh": "",
                        "lvb": "",
                        "noi_nhan": "",
                        "nguoi_ky": "",
                    }

                data["pdf_filename"] = filename

                try:
                    marks_filename = detect_stamp_and_signature(
                        save_path, output_dir=MARKS_FOLDER
                    )
                    if marks_filename:
                        data["marks_image"] = marks_filename
                except Exception as e:
                    print("Error in mark detection:", e)
                    data["marks_image"] = None

                data_list.append(data)

    return render_template("index.html", data_list=data_list)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/marks/<filename>")
def marks_file(filename):
    return send_from_directory(MARKS_FOLDER, filename)


@app.route("/semantic-search", methods=["POST"])
def semantic_search_route():
    query = request.form.get("query")
    results = semantic_search(query, top_k=5)

    return render_template(
        "index.html",
        search_results=results,
        query=query,
        data_list=None,
    )


@app.route("/view-pdf/<filename>")
def view_pdf(filename):

    pdf_name = filename.replace(".txt", ".pdf")

    pdf_path = os.path.join(PDF_FOLDER, pdf_name)

    if os.path.exists(pdf_path):
        return send_from_directory(PDF_FOLDER, pdf_name)
    else:
        abort(404)



if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(TEXT_FOLDER, exist_ok=True)
    os.makedirs(MARKS_FOLDER, exist_ok=True)

    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
