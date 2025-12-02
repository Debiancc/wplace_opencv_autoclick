import pytest
import os
import sys
from PIL import Image
import cv2
import numpy as np

# Add project root to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from square_detector import SquareDetector

@pytest.fixture
def detector():
    return SquareDetector()

@pytest.fixture
def test_image_path():
    return os.path.join("screenshots", "screenshot_20251201_142015.png")

@pytest.fixture
def output_dir():
    path = os.path.join("tests", "output")
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def test_detection_with_sample_image(detector, test_image_path, output_dir):
    """Test detection using the specific sample image provided."""
    if not os.path.exists(test_image_path):
        pytest.skip(f"Test image not found: {test_image_path}")
        
    print(f"\nTesting with image: {test_image_path}")
    image = Image.open(test_image_path)
    
    # Run detection
    detections = detector.detect_squares(image, visualize_steps=True)
    
    # Assertions
    assert isinstance(detections, list)
    print(f"Detected {len(detections)} squares")
    
    # We expect some detections in this image
    assert len(detections) > 0, "Should detect at least one square"
    
    # Calculate absolute coordinates (simulating a region offset)
    offset_x, offset_y = 100, 100
    detections = detector.calculate_absolute_coordinates(detections, (offset_x, offset_y))

    # Verify detection structure
    for det in detections:
        assert 'center' in det
        assert 'bbox' in det
        assert 'area' in det
        assert 'absolute_coords' in det
        
        # Verify absolute coords are correct relative to center
        cx, cy = det['center']
        abs_x, abs_y = det['absolute_coords']
        assert abs_x == cx + offset_x
        assert abs_y == cy + offset_y
        
    # Visualize and save result
    visualized = detector.visualize_detections(image, detections)
    output_path = os.path.join(output_dir, "test_result_screenshot_20251201_142015.png")
    visualized.save(output_path)
    print(f"Saved visualization to: {output_path}")
    
    # Verify output file exists
    assert os.path.exists(output_path)
