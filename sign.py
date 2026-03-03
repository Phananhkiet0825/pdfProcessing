import cv2
import numpy as np
from pdf2image import convert_from_path
import os

def detect_signatures(pdf_path, output_dir="signatures_output"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pages = convert_from_path(pdf_path, dpi=300)
    signature_boxes = []

    for i, page in enumerate(pages, start=1):
        # Convert page to OpenCV image
        page_path = os.path.join(output_dir, f"page_{i}.png")
        page.save(page_path, "PNG")
        image = cv2.imread(page_path)

        # Focus only on bottom 1/3 of the page
        h = image.shape[0]
        bottom_crop = image[int(h*0.65):, :]  # bottom 35%
        
        gray = cv2.cvtColor(bottom_crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5,5), 0)
        
        # Adaptive threshold (good for handwriting)
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15, 8
        )

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h_box = cv2.boundingRect(cnt)
            # Filter by size (tune these thresholds)
            if 80 < w < 800 and 20 < h_box < 300:
                # Adjust y to original page coordinate
                y += int(h*0.65)

                signature_boxes.append((i, x, y, w, h_box))

                # Draw rectangle on page image
                cv2.rectangle(image, (x, y), (x+w, y+h_box), (255,0,0), 2)

                # Save cropped signature
                sig_crop = image[y:y+h_box, x:x+w]
                cv2.imwrite(os.path.join(output_dir, f"page_{i}_signature.png"), sig_crop)

        # Save annotated page
        cv2.imwrite(os.path.join(output_dir, f"page_{i}_annotated.png"), image)

    return signature_boxes

# Example usage
pdf_path = "Thong_tu_06_2016_TT-BGDT.pdf"
signatures = detect_signatures(pdf_path)
print("Detected signature boxes:", signatures)
