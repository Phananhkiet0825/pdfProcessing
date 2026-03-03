import cv2
import numpy as np
from pdf2image import convert_from_path
import os

def detect_stamp_and_signature(pdf_path, output_dir="marks_output"):
    """
    Detect stamp + signature blocks from a PDF and output one combined PNG.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    out_path = os.path.join(output_dir, f"{base_name}_marks.png")

    pages = convert_from_path(pdf_path, dpi=300)
    mark_crops = []

    for page_index, page in enumerate(pages, start=1):
        # Convert to cv2 image
        page_cv = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

        # Convert to HSV to find red stamps
        hsv = cv2.cvtColor(page_cv, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask1, mask2)

        # Clean mask
        kernel = np.ones((5,5), np.uint8)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # This filters typical stamp sizes, adjust as needed
            if 80 < w < 1000 and 80 < h < 1000:
                # Expand bounding box to include signature area around stamp
                expand_x = 300  # go right to catch signature
                expand_y = 150  # go up/down a bit
                x1 = max(0, x - 20)
                y1 = max(0, y - expand_y)
                x2 = min(page_cv.shape[1], x + w + expand_x)
                y2 = min(page_cv.shape[0], y + h + expand_y)

                crop = page_cv[y1:y2, x1:x2]
                mark_crops.append(crop)

    if not mark_crops:
        print("No stamps/signatures found.")
        return None

    # Stack all marks vertically with padding
    max_width = max(img.shape[1] for img in mark_crops)
    padded = []
    for img in mark_crops:
        h, w = img.shape[:2]
        if w < max_width:
            pad_left = (max_width - w) // 2
            pad_right = max_width - w - pad_left
            img = cv2.copyMakeBorder(img, 20, 20, pad_left+20, pad_right+20,
                                     cv2.BORDER_CONSTANT, value=(255,255,255))
        else:
            img = cv2.copyMakeBorder(img, 20, 20, 20, 20,
                                     cv2.BORDER_CONSTANT, value=(255,255,255))
        padded.append(img)

    combined = cv2.vconcat(padded)
    cv2.imwrite(out_path, combined)
    print(f"Combined stamp+signature image saved at: {out_path}")
    return out_path

# Example:
pdf_path = "Thong_tu_06_2016_TT-BGDT.pdf"
output_image = detect_stamp_and_signature(pdf_path)
print("Result:", output_image)
