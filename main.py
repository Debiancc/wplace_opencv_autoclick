"""
Screenshot Application with Square Detection
Captures screenshots using Ctrl+Shift+Q hotkey and performs square detection.
"""
import argparse
from screenshot_capture import ScreenshotCapture
from hotkey_listener import HotkeyListener
from square_detector import SquareDetector
from pynput.mouse import Button, Controller
import time


class ScreenshotApp:
    """Main application integrating screenshot capture and square detection."""
    
    def __init__(self, enable_square_detection=False, click_delay=0.5):
        """
        Initialize the application.
        
        Args:
            enable_square_detection (bool): Whether to detect square points
            click_delay (float): Delay in seconds between clicks
        """
        self.enable_square_detection = enable_square_detection
        self.click_delay = click_delay
        
        # Initialize square detector first if needed
        self.square_detector = None
        if enable_square_detection:
            print("Initializing square detector...")
            self.square_detector = SquareDetector()
            print("Square detector initialized successfully!")
        
        # Initialize screenshot capture with detector
        self.screenshot_capture = ScreenshotCapture(square_detector=self.square_detector)
        
        # Cleanup screenshots directory on startup
        self.screenshot_capture.clear_directory()
        
        self.mouse = Controller()
    
    def on_screenshot_hotkey(self):
        """Callback function triggered when screenshot hotkey is pressed."""
        try:
            print("\n" + "="*50)
            print("Capturing screenshot...")
            
            # Capture and save screenshot (now returns filepath and region)
            filepath, region = self.screenshot_capture.capture_and_save()
            
            if filepath is None:
                print("Screenshot cancelled")
                print("="*50 + "\n")
                return
            
            # Perform square point detection if enabled
            if self.enable_square_detection and self.square_detector:
                print("\nDetecting square points...")
                from PIL import Image
                img = Image.open(filepath)
                print(f"✓ Loaded image: {img.size}")
                
                # Detect squares
                detections = self.square_detector.detect_squares(img)
                
                if detections:
                    # Calculate absolute coordinates
                    x1, y1, x2, y2 = region
                    detections = self.square_detector.calculate_absolute_coordinates(
                        detections, 
                        region_offset=(x1, y1)
                    )
                    print(f"✓ Calculated absolute coordinates")
                    
                    # --- Mouse Interaction: Click all squares ---
                    print(f"\n✓ Preparing to click {len(detections)} squares (delay: {self.click_delay}s)...")
                    print("  Tip: Move mouse to interrupt clicking")
                    
                    for i, det in enumerate(detections):
                        target_x, target_y = det['absolute_coords']
                        print(f"  [{i+1}/{len(detections)}] Clicking square: ({target_x}, {target_y})")
                        
                        # Move mouse
                        self.mouse.position = (target_x, target_y)
                        
                        # Safety check: Wait and check if mouse moved
                        check_interval = 0.05
                        elapsed = 0
                        interrupted = False
                        
                        while elapsed < self.click_delay:
                            time.sleep(check_interval)
                            elapsed += check_interval
                            
                            # Check if mouse moved significantly from target
                            curr_x, curr_y = self.mouse.position
                            if abs(curr_x - target_x) > 5 or abs(curr_y - target_y) > 5:
                                print("\n!!! Mouse movement detected, interrupting operation !!!")
                                interrupted = True
                                break
                        
                        if interrupted:
                            break
                            
                        # Perform click
                        self.mouse.click(Button.left)
                    
                    if not interrupted:
                        print(f"✓ All squares clicked")
                    
                    # Visualize
                    print(f"✓ Generating visualization...")
                    visualized = self.square_detector.visualize_detections(img, detections)
                    vis_path = filepath.replace('.png', '_squares.png')
                    visualized.save(vis_path)
                    print(f"✓ Visualization saved: {vis_path}")
                    
                    # Print summary
                    print(f"\nSquare point details (first 10):")
                    for i, det in enumerate(detections[:10]):
                        abs_x, abs_y = det['absolute_coords']
                        brightness = det.get('brightness', 'N/A')
                        filled = det.get('filled_ratio', 'N/A')
                        print(f"  Point {i+1}: Screen coords ({abs_x}, {abs_y}) - Area: {det['area']}px - Brightness: {brightness:.1f} - Fill ratio: {filled:.2f}")
                    
                    if len(detections) > 10:
                        print(f"  ... and {len(detections) - 10} more square points")
                    
                    print(f"\n✓ Total: {len(detections)} square points")
                else:
                    print("No square points detected")
            
            print("\nScreenshot complete!")
            print("="*50 + "\n")
            
        except Exception as e:
            import traceback
            print(f"Error capturing screenshot: {e}")
            traceback.print_exc()
            print("="*50 + "\n")
    
    def run(self):
        """Start the application and listen for hotkeys."""
        print("\n" + "="*60)
        print("Screenshot Application")
        print("="*60)
        print(f"Square detection: {'Enabled' if self.enable_square_detection else 'Disabled'}")
        print(f"Click delay: {self.click_delay}s")
        print("="*60 + "\n")
        
        # Create and start hotkey listener
        listener = HotkeyListener(hotkey_callback=self.on_screenshot_hotkey)
        listener.start()


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Screenshot tool with square point detection'
    )
    parser.add_argument(
        '--detect-squares',
        action='store_true',
        help='Enable square point detection and counting'
    )
    parser.add_argument(
        '--click-delay',
        type=float,
        default=0.5,
        help='Delay between automatic clicks (seconds), default 0.5'
    )
    
    args = parser.parse_args()
    
    # Create and run application
    app = ScreenshotApp(
        enable_square_detection=True,
        click_delay=0.01
    )
    app.run()


if __name__ == '__main__':
    main()
