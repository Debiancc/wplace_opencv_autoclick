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

def verify_square_detection(detector, image_path, expected_count, offset=(0, 0)):
    """
    Helper function to verify square detection for a given image.
    
    Args:
        detector: SquareDetector instance
        image_path: Path to the test image
        expected_count: Expected number of squares to detect
        offset: Tuple of (offset_x, offset_y) for absolute coordinate calculation
    """
    if not os.path.exists(image_path):
        pytest.skip(f"Test image not found: {image_path}")
        
    print(f"\nTesting with image: {image_path}")
    image = Image.open(image_path)
    
    # Run detection
    detections = detector.detect_squares(image, visualize_steps=False)
    
    # Assertions
    assert isinstance(detections, list)
    print(f"Detected {len(detections)} squares")
    
    # Verify expected count
    assert len(detections) == expected_count, \
        f"Should detect exactly {expected_count} squares, but found {len(detections)}"
    
    # Calculate absolute coordinates
    offset_x, offset_y = offset
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
    
    print(f"✓ Successfully verified all {expected_count} squares")

def test_detection_with_sample_image(detector):
    """Test detection using the specific sample image provided."""
    verify_square_detection(
        detector,
        os.path.join("tests", "basic.png"),
        expected_count=12,
        offset=(100, 100)
    )

def test_detection_with_1199_image(detector):
    """Test detection with 1196.png - should detect exactly 1196 squares."""
    verify_square_detection(
        detector,
        os.path.join("tests", "1196.png"),
        expected_count=1196
    )


def test_detection_with_white16_image(detector):
    """Test detection with white16.png - should detect exactly 16 squares."""
    verify_square_detection(
        detector,
        os.path.join("tests", "white16.png"),
        expected_count=16
    )

def test_detection_with_light_yellow_7_image(detector):
    """Test detection with light_yellow_6.png - should detect exactly 6 squares."""
    verify_square_detection(
        detector,
        os.path.join("tests", "light_yellow_7.png"),
        expected_count=7
    )

def test_detection_with_light_cyan_4_image(detector):
    """Test detection with light_cyan_4.png - should detect exactly 4 squares."""
    verify_square_detection(
        detector,
        os.path.join("tests", "light_cyan_4.png"),
        expected_count=4
    )
