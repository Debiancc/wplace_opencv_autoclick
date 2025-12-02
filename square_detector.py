"""
Square point detector using OpenCV.
Detects small square points in screenshots with adaptive sizing.
"""
import cv2
import numpy as np
from PIL import Image


class SquareDetector:
    """Detect square points in images with color difference analysis."""
    
    def __init__(self, min_area_ratio=0.3, max_area_ratio=3.0):
        """
        Initialize the square detector.
        
        Args:
            min_area_ratio (float): Minimum area relative to median (default: 0.3)
            max_area_ratio (float): Maximum area relative to median (default: 3.0)
        """
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
    
    def _pil_to_cv(self, pil_image):
        """
        Convert PIL Image to OpenCV format.
        
        Args:
            pil_image (PIL.Image): Input PIL image
            
        Returns:
            numpy.ndarray: OpenCV BGR image
        """
        if isinstance(pil_image, Image.Image):
            # Convert PIL to numpy array (RGB)
            rgb = np.array(pil_image)
            # Convert RGB to BGR for OpenCV
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return pil_image
    
    def _cv_to_pil(self, cv_image):
        """
        Convert OpenCV image to PIL format.
        
        Args:
            cv_image (numpy.ndarray): OpenCV BGR image
            
        Returns:
            PIL.Image: PIL RGB image
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    
    def _detect_background_color(self, image):
        """
        Detect background color by sampling corners.
        
        Args:
            image (numpy.ndarray): OpenCV BGR image
            
        Returns:
            numpy.ndarray: Background color as [B, G, R]
        """
        h, w = image.shape[:2]
        sample_size = min(20, h // 10, w // 10)
        
        # Sample four corners
        corners = [
            image[0:sample_size, 0:sample_size],              # Top-left
            image[0:sample_size, -sample_size:],              # Top-right
            image[-sample_size:, 0:sample_size],              # Bottom-left
            image[-sample_size:, -sample_size:]               # Bottom-right
        ]
        
        # Calculate average color from all corners
        corner_means = [corner.mean(axis=(0, 1)) for corner in corners]
        background_color = np.mean(corner_means, axis=0)
        
        return background_color
    
    def detect_squares(self, image, background_color=None, visualize_steps=False):
        """
        Detect square points in the image.
        
        Args:
            image (PIL.Image or numpy.ndarray): Input image
            background_color (tuple or None): Background color as (B, G, R). 
                                             If None, auto-detect from corners.
            visualize_steps (bool): If True, return intermediate processing steps
            
        Returns:
            list: List of detected square information dicts with keys:
                  - 'center': (x, y) center coordinates
                  - 'bbox': (x1, y1, x2, y2) bounding box
                  - 'area': area in pixels
                  - 'contour': contour points
        """
        # Convert to OpenCV format
        cv_image = self._pil_to_cv(image)
        
        # Detect background color if not provided
        if background_color is None:
            background_color = self._detect_background_color(cv_image)
            print(f"Detected background color: BGR{background_color.astype(int)}")
        
        # Calculate color difference from background
        diff = np.linalg.norm(cv_image - background_color, axis=2)
        
        # Adaptive thresholding based on statistics
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        threshold = mean_diff + 1.5 * std_diff
        
        print(f"Color difference threshold: {threshold:.2f}")
        
        # Create binary mask
        mask = (diff > threshold).astype(np.uint8) * 255
        
        # Save debug mask
        if visualize_steps:
            cv2.imwrite("debug_mask_raw.png", mask)
        
        # Morphological operations to reduce noise
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        if visualize_steps:
            cv2.imwrite("debug_mask_processed.png", mask)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"Found {len(contours)} contours")
        
        # Calculate area statistics for adaptive filtering
        if len(contours) == 0:
            print("No contours detected")
            return []
        
        # Filter out very small noise first (hard limit)
        valid_contours = [c for c in contours if cv2.contourArea(c) > 10]
        
        if len(valid_contours) == 0:
            print("No valid contours (>10px) detected")
            return []
            
        areas = [cv2.contourArea(c) for c in valid_contours]
        median_area = np.median(areas)
        
        # Relaxed area constraints
        min_area = max(10, median_area * 0.2) # At least 10px
        max_area = median_area * 5.0
        
        print(f"Area filter range: {min_area:.1f} - {max_area:.1f} px (median: {median_area:.1f})")
        
        # Filter and extract square information
        detections = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Area filter
            if area < min_area or area > max_area:
                # print(f"轮廓 #{i} 拒绝: 面积 {area:.1f} 不在范围内")
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Aspect ratio filter (should be close to square)
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                print(f"Contour #{i} rejected: aspect ratio {aspect_ratio:.2f}")
                continue
            
            # Extract region of interest (ROI) for additional validation
            roi = cv_image[y:y+h, x:x+w]
            
            # --- Noise Filtering: Check if square is solid and dark ---
            
            # 1. Check filled ratio (solid vs hollow)
            # Compare contour area with bounding box area
            bbox_area = w * h
            filled_ratio = area / bbox_area if bbox_area > 0 else 0
            
            # If filled ratio is too low, it's likely a hollow shape/icon
            if filled_ratio < 0.4: # Relaxed from 0.5
                print(f"Contour #{i} rejected: fill ratio {filled_ratio:.2f} < 0.4")
                continue
            
            # 2. Check if the region is dark/black (relative to background)
            # Calculate mean brightness of the ROI
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(roi_gray)
            
            # Calculate background brightness (approximate)
            bg_brightness = np.mean(background_color)
            
            # Solid black squares should be significantly darker than background
            # Or just use a relaxed absolute threshold
            if mean_brightness > 100 and mean_brightness > (bg_brightness - 30):
                print(f"Contour #{i} rejected: brightness {mean_brightness:.1f} (background: {bg_brightness:.1f})")
                continue
            
            # 3. Check edge density (hollow shapes have mostly edges, solid shapes are filled)
            # Create a mask for this specific contour
            contour_mask = np.zeros(roi_gray.shape, dtype=np.uint8)
            # Adjust contour coordinates to ROI space
            adjusted_contour = contour - np.array([x, y])
            cv2.drawContours(contour_mask, [adjusted_contour], -1, 255, -1)
            
            # Count non-zero pixels in the ROI mask
            filled_pixels = np.count_nonzero(contour_mask)
            edge_ratio = filled_pixels / bbox_area if bbox_area > 0 else 0
            
            # Solid squares should have high edge ratio
            if edge_ratio < 0.3: # Relaxed from 0.4
                print(f"Contour #{i} rejected: edge ratio {edge_ratio:.2f} < 0.3")
                continue
            
            # Calculate center
            center_x = x + w // 2
            center_y = y + h // 2
            
            detection = {
                'center': (center_x, center_y),
                'bbox': (x, y, x + w, y + h),
                'area': int(area),
                'contour': contour,
                'filled_ratio': filled_ratio,
                'brightness': mean_brightness,
                'edge_ratio': edge_ratio
            }
            
            detections.append(detection)
        
        print(f"Detected {len(detections)} square points")
        
        return detections
    
    def calculate_absolute_coordinates(self, detections, region_offset):
        """
        Calculate absolute screen coordinates for detected squares.
        
        Args:
            detections (list): List of detection dicts
            region_offset (tuple): (offset_x, offset_y) of the screenshot region
            
        Returns:
            list: Detections with added 'absolute_coords' field
        """
        offset_x, offset_y = region_offset
        
        for detection in detections:
            rel_x, rel_y = detection['center']
            detection['absolute_coords'] = (rel_x + offset_x, rel_y + offset_y)
        
        return detections
    
    def visualize_detections(self, image, detections, show_labels=True):
        """
        Draw detection results on the image.
        
        Args:
            image (PIL.Image or numpy.ndarray): Input image
            detections (list): List of detection dicts
            show_labels (bool): Whether to show labels with index numbers
            
        Returns:
            PIL.Image: Image with drawn detections
        """
        # Convert to OpenCV format
        cv_image = self._pil_to_cv(image)
        result = cv_image.copy()
        
        # Draw each detection
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            center_x, center_y = det['center']
            
            # Draw bounding box (green)
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw center point (red)
            cv2.circle(result, (center_x, center_y), 3, (0, 0, 255), -1)
            
            # Draw label with index
            if show_labels:
                label = f"#{i+1}"
                # Add background for text
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(result, 
                            (x1, y1 - text_h - 4), 
                            (x1 + text_w + 4, y1), 
                            (0, 255, 0), -1)
                cv2.putText(result, label, (x1 + 2, y1 - 2), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Add summary text at top
        summary = f"Total: {len(detections)} squares"
        cv2.putText(result, summary, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Convert back to PIL
        return self._cv_to_pil(result)


if __name__ == "__main__":
    # Test with the sample image
    import os
    from pathlib import Path
    
    # Try to find a test image
    test_image_path = "screenshots/screenshot_20251130_174034.png"
    
    if os.path.exists(test_image_path):
        print(f"Test image: {test_image_path}")
        
        # Load image
        img = Image.open(test_image_path)
        print(f"Image size: {img.size}")
        
        # Create detector
        detector = SquareDetector()
        
        # Detect squares
        detections = detector.detect_squares(img, visualize_steps=True)
        
        # Print results
        print(f"\nDetection results:")
        for i, det in enumerate(detections[:10]):  # Show first 10
            print(f"  Square {i+1}: center {det['center']}, area {det['area']}px")
        
        if len(detections) > 10:
            print(f"  ... and {len(detections) - 10} more squares")
        
        # Visualize
        visualized = detector.visualize_detections(img, detections)
        
        # Save result
        output_path = test_image_path.replace('.png', '_squares_detected.png')
        visualized.save(output_path)
        print(f"\nVisualization saved: {output_path}")
    else:
        print(f"Test image does not exist: {test_image_path}")
        print("Please run screenshot function first to generate test image")
