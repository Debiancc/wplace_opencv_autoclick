
import os
import sys
from PIL import Image
from square_detector import SquareDetector

def debug_yellow():
    path = os.path.join("tests", "light_yellow_6.png")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    print(f"Debugging {path}")
    image = Image.open(path)
    detector = SquareDetector()
    detections = detector.detect_squares(image, visualize_steps=True)
    
    for i, det in enumerate(detections):
        x, y, w, h = det['bbox']
        roi = np.array(image)[y:y+h, x:x+w] # RGB
        avg_color = roi.mean(axis=(0,1))
        
        w = det['bbox'][2] - det['bbox'][0]
        h = det['bbox'][3] - det['bbox'][1]
        ar = w / h if h > 0 else 0
        print(f"#{i}: Center={det['center']}, Area={det['area']}, Color={avg_color.astype(int)}, AR={ar:.2f}")

    # Save visualization
    vis = detector.visualize_detections(image, detections)
    vis.save("debug_yellow_7.png")
    print("Saved debug_yellow_7.png")

if __name__ == "__main__":
    debug_yellow()
