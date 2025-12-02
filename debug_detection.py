import cv2
import numpy as np
from PIL import Image
from square_detector import SquareDetector
import os

def debug_detection(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return

    print(f"Analyzing image: {image_path}")
    img = Image.open(image_path)
    detector = SquareDetector()
    
    # Manually run steps to debug
    cv_image = detector._pil_to_cv(img)
    
    # 1. Background Color
    bg_color = detector._detect_background_color(cv_image)
    print(f"Detected Background Color (BGR): {bg_color}")
    
    # 2. Color Difference
    diff = np.linalg.norm(cv_image - bg_color, axis=2)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    threshold = mean_diff + 1.5 * std_diff
    print(f"Mean Diff: {mean_diff:.2f}, Std Diff: {std_diff:.2f}, Threshold: {threshold:.2f}")
    
    # 3. Mask
    mask = (diff > threshold).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 4. Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} contours")
    
    if len(contours) == 0:
        return

    areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 0]
    if not areas:
        print("No valid areas found.")
        return
        
    median_area = np.median(areas)
    min_area = median_area * detector.min_area_ratio
    max_area = median_area * detector.max_area_ratio
    print(f"Median Area: {median_area}, Filter Range: {min_area:.1f} - {max_area:.1f}")

    print("\n--- Contour Analysis ---")
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Basic checks
        if area < min_area or area > max_area:
            # print(f"Contour {i}: Rejected by Area ({area:.1f})")
            continue
            
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            print(f"Contour {i}: Rejected by Aspect Ratio ({aspect_ratio:.2f})")
            continue

        # Advanced checks (Noise Filtering)
        roi = cv_image[y:y+h, x:x+w]
        bbox_area = w * h
        filled_ratio = area / bbox_area if bbox_area > 0 else 0
        
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(roi_gray)
        
        contour_mask = np.zeros(roi_gray.shape, dtype=np.uint8)
        adjusted_contour = contour - np.array([x, y])
        cv2.drawContours(contour_mask, [adjusted_contour], -1, 255, -1)
        filled_pixels = np.count_nonzero(contour_mask)
        edge_ratio = filled_pixels / bbox_area if bbox_area > 0 else 0
        
        print(f"Contour {i} (Accepted Candidate?):")
        print(f"  Area: {area:.1f}, BBox: {w}x{h}")
        print(f"  Filled Ratio: {filled_ratio:.2f} (Threshold >= 0.5)")
        print(f"  Brightness: {mean_brightness:.2f} (Threshold <= 40)")
        print(f"  Edge Ratio: {edge_ratio:.2f} (Threshold >= 0.4)")
        
        failed_reasons = []
        if filled_ratio < 0.5: failed_reasons.append("Low Filled Ratio")
        if mean_brightness > 40: failed_reasons.append("High Brightness")
        if edge_ratio < 0.4: failed_reasons.append("Low Edge Ratio")
        
        if failed_reasons:
            print(f"  -> REJECTED: {', '.join(failed_reasons)}")
        else:
            print(f"  -> ACCEPTED")

if __name__ == "__main__":
    # Use the path provided by the user
    # img_path = r"c:\Users\debiancc\.gemini\antigravity\scratch\yolo_vision_project\screenshots\screenshot_20251130_171417.png"
    # debug_detection(img_path)
