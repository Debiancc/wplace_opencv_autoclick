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
    return os.path.join("tests", "test_image.png")

def test_detection_with_sample_image(detector, test_image_path):
    """Test detection using the specific sample image provided."""
    if not os.path.exists(test_image_path):
        pytest.skip(f"Test image not found: {test_image_path}")
        
    print(f"\nTesting with image: {test_image_path}")
    image = Image.open(test_image_path)
    
    # Run detection
    detections = detector.detect_squares(image, visualize_steps=False)
    
    # Assertions
    assert isinstance(detections, list)
    print(f"Detected {len(detections)} squares")
    
    # We expect exactly 12 detections in this image
    assert len(detections) == 12, f"Should detect exactly 12 squares, but found {len(detections)}"
    
    # Calculate absolute coordinates (simulating a region offset)
    offset_x, offset_y = 100, 100
    detections = detector.calculate_absolute_coordinates(detections, (offset_x, offset_y))

    # Verify detection structure and coordinates
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
