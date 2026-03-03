import cv2
import numpy as np
from pdf2image import convert_from_path
import os

def detect_stamp_in_pdf(pdf_path, output_dir="stamps_output"):
    os.makedirs(output_dir, exist_ok=True)

    # Convert PDF pages to images
    pages = convert_from_path(pdf_path, dpi=300)
    stamp_locations = []

    for i, page in enumerate(pages):
        img_path = os.path.join(output_dir, f"page_{i+1}.png")
        page.save(img_path, 'PNG')

        # Read as OpenCV image
        img = cv2.imread(img_path)

        # Convert to HSV to isolate red regions (common for seals)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Red has two ranges in HSV
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2

        # Morphological ops to clean noise
        kernel = np.ones((5, 5), np.uint8)
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_DILATE, kernel)

        # Find contours in the red area
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if area > 5000:  # adjust threshold for stamp size
                x, y, w, h = cv2.boundingRect(c)
                # Save region or draw rectangle
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                stamp_crop = img[y:y+h, x:x+w]
                cv2.imwrite(os.path.join(output_dir, f"page_{i+1}_stamp.png"), stamp_crop)
                stamp_locations.append((i+1, x, y, w, h))

        # Save annotated page
        cv2.imwrite(os.path.join(output_dir, f"page_{i+1}_annotated.png"), img)

    return stamp_locations

# Usage example:
pdf_path = "Thong_tu_06_2016_TT-BGDT.pdf"
stamps = detect_stamp_in_pdf(pdf_path)
print("Detected stamp locations:", stamps)
