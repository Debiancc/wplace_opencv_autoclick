"""
Screenshot capture module with interactive region selection.
Provides functionality to capture screenshots with mouse-based region selection.
"""
import os
import tkinter as tk
from datetime import datetime
from mss import mss
from PIL import Image, ImageTk, ImageDraw


class RegionSelector:
    """Interactive region selector using tkinter overlay with resize handles."""
    
    def __init__(self, screenshot, square_detector=None):
        """
        Initialize the region selector with a screenshot.
        
        Args:
            screenshot (PIL.Image): The full screen screenshot to select from
            square_detector (SquareDetector): Optional detector for real-time detection
        """
        self.screenshot = screenshot
        self.selected_region = None
        self.square_detector = square_detector
        
        # Detection state
        self.detections = []
        self.detection_markers = []
        self.detection_summary_text = None
        self.detection_pending = None  # Timer for debouncing
        self.detection_running = False  # Flag to prevent overlapping detections
        
        # Create fullscreen transparent window
        self.root = tk.Tk()
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-alpha', 0.3)  # Semi-transparent
        self.root.attributes('-topmost', True)
        self.root.configure(cursor='crosshair')
        
        # Ensure window appears on top and gains focus
        self.root.lift()
        self.root.focus_force()
        self.root.update()
        
        # Get screen dimensions
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        
        # Create canvas
        self.canvas = tk.Canvas(
            self.root, 
            width=self.screen_width, 
            height=self.screen_height,
            highlightthickness=0,
            bg='black'
        )
        self.canvas.pack()
        
        # Display screenshot
        self.tk_image = ImageTk.PhotoImage(screenshot)
        self.canvas.create_image(0, 0, image=self.tk_image, anchor=tk.NW)
        
        # Selection state
        self.mode = "drawing"  # "drawing" or "adjusting"
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.rect_coords = None  # (x1, y1, x2, y2)
        
        # Resize handles
        self.handle_size = 8
        self.handles = {}
        self.current_handle = None
        self.drag_start_pos = None
        self.drag_start_rect = None
        
        # Dimension display
        self.dim_text = None
        
        # Detection display elements
        self.detection_summary_text = None
        self.detection_markers = []
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.on_mouse_down)
        self.canvas.bind('<B1-Motion>', self.on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_up)
        self.canvas.bind('<Motion>', self.on_mouse_move)
        self.canvas.bind('<Double-Button-1>', self.on_confirm)
        
        # Bind keyboard events
        self.root.bind('<Escape>', self.on_cancel)
        self.root.bind('<Return>', self.on_confirm)
        
        # Instructions
        self.instruction_text = None
        self.create_instructions()
    
    def create_instructions(self):
        """Display instructions on the overlay."""
        if self.mode == "drawing":
            text = "Drag to select region | ESC to cancel"
        else:
            text = "Drag handles to resize | Drag inside to move | ENTER or Double-click to confirm | ESC to cancel"
        
        if self.instruction_text:
            self.canvas.delete(self.instruction_text)
        
        self.instruction_text = self.canvas.create_text(
            self.screen_width // 2,
            30,
            text=text,
            fill='white',
            font=('Arial', 12, 'bold'),
            tags='instruction'
        )
    
    def draw_selection_box(self):
        """Draw selection rectangle with resize handles."""
        if not self.rect_coords:
            return
        
        x1, y1, x2, y2 = self.rect_coords
        
        # Delete old elements
        if self.rect:
            self.canvas.delete(self.rect)
        for handle in self.handles.values():
            self.canvas.delete(handle)
        if self.dim_text:
            self.canvas.delete(self.dim_text)
        
        # Draw main rectangle
        self.rect = self.canvas.create_rectangle(
            x1, y1, x2, y2,
            outline='#00FF00',
            width=2,
            tags='selection'
        )
        
        # Draw resize handles
        h = self.handle_size
        self.handles = {}
        
        # Corner handles
        self.handles['nw'] = self.canvas.create_rectangle(
            x1 - h//2, y1 - h//2, x1 + h//2, y1 + h//2,
            fill='#00FF00', outline='white', width=1, tags='handle'
        )
        self.handles['ne'] = self.canvas.create_rectangle(
            x2 - h//2, y1 - h//2, x2 + h//2, y1 + h//2,
            fill='#00FF00', outline='white', width=1, tags='handle'
        )
        self.handles['sw'] = self.canvas.create_rectangle(
            x1 - h//2, y2 - h//2, x1 + h//2, y2 + h//2,
            fill='#00FF00', outline='white', width=1, tags='handle'
        )
        self.handles['se'] = self.canvas.create_rectangle(
            x2 - h//2, y2 - h//2, x2 + h//2, y2 + h//2,
            fill='#00FF00', outline='white', width=1, tags='handle'
        )
        
        # Edge handles
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        
        self.handles['n'] = self.canvas.create_rectangle(
            mid_x - h//2, y1 - h//2, mid_x + h//2, y1 + h//2,
            fill='#00FF00', outline='white', width=1, tags='handle'
        )
        self.handles['s'] = self.canvas.create_rectangle(
            mid_x - h//2, y2 - h//2, mid_x + h//2, y2 + h//2,
            fill='#00FF00', outline='white', width=1, tags='handle'
        )
        self.handles['w'] = self.canvas.create_rectangle(
            x1 - h//2, mid_y - h//2, x1 + h//2, mid_y + h//2,
            fill='#00FF00', outline='white', width=1, tags='handle'
        )
        self.handles['e'] = self.canvas.create_rectangle(
            x2 - h//2, mid_y - h//2, x2 + h//2, mid_y + h//2,
            fill='#00FF00', outline='white', width=1, tags='handle'
        )
        
        # Draw dimensions
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        dim_str = f"{width} × {height}"
        
        self.dim_text = self.canvas.create_text(
            x1 + 5, y1 - 20,
            text=dim_str,
            fill='#00FF00',
            font=('Arial', 10, 'bold'),
            anchor='w',
            tags='dimension'
        )
        
        # Don't run detection during drag - only on mouse release for better performance
    
    def get_handle_at_position(self, x, y):
        """Determine which handle or region is at the given position."""
        if not self.rect_coords:
            return None
        
        # Check handles first
        for name, handle_id in self.handles.items():
            coords = self.canvas.coords(handle_id)
            if coords and len(coords) == 4:
                hx1, hy1, hx2, hy2 = coords
                if hx1 <= x <= hx2 and hy1 <= y <= hy2:
                    return name
        
        # Check if inside rectangle (for moving)
        x1, y1, x2, y2 = self.rect_coords
        if x1 <= x <= x2 and y1 <= y <= y2:
            return 'move'
        
        return None
    
    def update_cursor(self, handle):
        """Update cursor based on handle type."""
        cursor_map = {
            'nw': 'top_left_corner',
            'ne': 'top_right_corner',
            'sw': 'bottom_left_corner',
            'se': 'bottom_right_corner',
            'n': 'top_side',
            's': 'bottom_side',
            'w': 'left_side',
            'e': 'right_side',
            'move': 'fleur',
        }
        cursor = cursor_map.get(handle, 'crosshair')
        self.root.configure(cursor=cursor)
    
    def on_mouse_move(self, event):
        """Handle mouse movement to update cursor."""
        if self.mode == "adjusting":
            handle = self.get_handle_at_position(event.x, event.y)
            self.update_cursor(handle)
    
    def on_mouse_down(self, event):
        """Handle mouse button press."""
        if self.mode == "drawing":
            self.start_x = event.x
            self.start_y = event.y
            
            # Create rectangle
            if self.rect:
                self.canvas.delete(self.rect)
            self.rect = self.canvas.create_rectangle(
                self.start_x, self.start_y, self.start_x, self.start_y,
                outline='#00FF00', width=2
            )
        
        elif self.mode == "adjusting":
            self.current_handle = self.get_handle_at_position(event.x, event.y)
            self.drag_start_pos = (event.x, event.y)
            self.drag_start_rect = self.rect_coords
    
    def on_mouse_drag(self, event):
        """Handle mouse drag."""
        if self.mode == "drawing":
            if self.rect:
                self.canvas.coords(
                    self.rect,
                    self.start_x, self.start_y,
                    event.x, event.y
                )
        
        elif self.mode == "adjusting" and self.current_handle:
            dx = event.x - self.drag_start_pos[0]
            dy = event.y - self.drag_start_pos[1]
            
            x1, y1, x2, y2 = self.drag_start_rect
            
            # Handle resize/move based on handle type
            if self.current_handle == 'move':
                # Move entire rectangle
                x1 += dx
                x2 += dx
                y1 += dy
                y2 += dy
            elif self.current_handle == 'nw':
                x1 += dx
                y1 += dy
            elif self.current_handle == 'ne':
                x2 += dx
                y1 += dy
            elif self.current_handle == 'sw':
                x1 += dx
                y2 += dy
            elif self.current_handle == 'se':
                x2 += dx
                y2 += dy
            elif self.current_handle == 'n':
                y1 += dy
            elif self.current_handle == 's':
                y2 += dy
            elif self.current_handle == 'w':
                x1 += dx
            elif self.current_handle == 'e':
                x2 += dx
            
            # Ensure minimum size
            min_size = 20
            if abs(x2 - x1) < min_size:
                if x2 > x1:
                    x2 = x1 + min_size
                else:
                    x1 = x2 + min_size
            
            if abs(y2 - y1) < min_size:
                if y2 > y1:
                    y2 = y1 + min_size
                else:
                    y1 = y2 + min_size
            
            # Keep within screen bounds
            x1 = max(0, min(x1, self.screen_width))
            x2 = max(0, min(x2, self.screen_width))
            y1 = max(0, min(y1, self.screen_height))
            y2 = max(0, min(y2, self.screen_height))
            
            self.rect_coords = (x1, y1, x2, y2)
            self.draw_selection_box()
    
    def on_mouse_up(self, event):
        """Handle mouse button release."""
        if self.mode == "drawing":
            end_x = event.x
            end_y = event.y
            
            # Calculate the region (ensure positive dimensions)
            x1 = min(self.start_x, end_x)
            y1 = min(self.start_y, end_y)
            x2 = max(self.start_x, end_x)
            y2 = max(self.start_y, end_y)
            
            # Check if selection is large enough
            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                self.rect_coords = (x1, y1, x2, y2)
                self.mode = "adjusting"
                self.create_instructions()
                self.draw_selection_box()
                
                # Run initial detection when selection is first created
                if self.square_detector:
                    self.schedule_detection()
            else:
                # Selection too small, stay in drawing mode
                if self.rect:
                    self.canvas.delete(self.rect)
                    self.rect = None
        
        elif self.mode == "adjusting":
            self.current_handle = None
            self.drag_start_pos = None
            self.drag_start_rect = None
            
            # Run detection after drag completes for better performance
            if self.square_detector:
                self.schedule_detection()
    
    def schedule_detection(self):
        """Schedule detection to run after a short delay (debouncing)."""
        # Cancel any pending detection
        if self.detection_pending:
            self.root.after_cancel(self.detection_pending)
        
        # Schedule new detection after 300ms of inactivity
        self.detection_pending = self.root.after(300, self.run_detection)
    
    def run_detection(self):
        """Run square detection on the current selection region."""
        if not self.rect_coords or not self.square_detector:
            return
        
        # Prevent overlapping detection calls
        if self.detection_running:
            return
        
        self.detection_running = True
        
        try:
            x1, y1, x2, y2 = self.rect_coords
            
            # Crop the screenshot to the current selection
            cropped = self.screenshot.crop((int(x1), int(y1), int(x2), int(y2)))
            
            # Run detection (without printing to avoid console spam)
            import sys
            from io import StringIO
            
            # Suppress print output during detection
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            detections = self.square_detector.detect_squares(cropped)
            
            # Restore stdout
            sys.stdout = old_stdout
            
            # Store detections with adjusted coordinates (relative to selection)
            self.detections = detections
            
            # Update visual feedback
            self.draw_detection_feedback()
            
        except Exception as e:
            # Silently handle errors to avoid disrupting UI
            self.detections = []
            pass
        finally:
            self.detection_running = False
    
    def clear_detection_display(self):
        """Clear all detection visual elements."""
        # Clear markers
        for marker in self.detection_markers:
            self.canvas.delete(marker)
        self.detection_markers = []
        
        # Clear summary text
        if self.detection_summary_text:
            self.canvas.delete(self.detection_summary_text)
            self.detection_summary_text = None
    
    def draw_detection_feedback(self):
        """Draw detection summary and point markers."""
        # Clear previous detection display
        self.clear_detection_display()
        
        if not self.rect_coords:
            return
        
        x1, y1, x2, y2 = self.rect_coords
        
        # Draw detection summary in top-right corner of selection box
        summary_text = f"Detected: {len(self.detections)} squares. Enter to draw."
        
        # Position: top-right corner with some padding
        text_x = x2 - 10
        text_y = y1 + 15
        
        # Draw text with background for visibility
        self.detection_summary_text = self.canvas.create_text(
            text_x, text_y,
            text=summary_text,
            fill='#00FFFF',  # Cyan color
            font=('Arial', 10, 'bold'),
            anchor='e',  # Right-aligned
            tags='detection_summary'
        )
        
        # Draw solid dots at detected square centers
        for detection in self.detections:
            rel_x, rel_y = detection['center']
            
            # Convert to absolute screen coordinates
            abs_x = x1 + rel_x
            abs_y = y1 + rel_y
            
            # Draw filled circle (solid dot)
            radius = 4
            marker = self.canvas.create_oval(
                abs_x - radius, abs_y - radius,
                abs_x + radius, abs_y + radius,
                fill='#FF0000',  # Red color
                outline='#FFFFFF',  # White outline for visibility
                width=1,
                tags='detection_marker'
            )
            self.detection_markers.append(marker)
    
    def on_confirm(self, event=None):
        """Confirm selection and close."""
        if self.mode == "adjusting" and self.rect_coords:
            x1, y1, x2, y2 = self.rect_coords
            # Ensure correct order
            self.selected_region = (
                min(x1, x2), min(y1, y2),
                max(x1, x2), max(y1, y2)
            )
            self.root.quit()
            self.root.destroy()
    
    def on_cancel(self, event):
        """Handle escape key to cancel selection."""
        self.selected_region = None
        self.root.quit()
        self.root.destroy()
    
    def get_region(self):
        """
        Show the selector and return the selected region.
        
        Returns:
            tuple: (x1, y1, x2, y2) coordinates or None if cancelled
        """
        self.root.mainloop()
        return self.selected_region


class ScreenshotCapture:
    """Handle screenshot capture and saving operations."""
    
    def __init__(self, save_dir="screenshots", square_detector=None):
        """
        Initialize the screenshot capture handler.
        
        Args:
            save_dir (str): Directory to save screenshots
            square_detector (SquareDetector): Optional detector for real-time detection
        """
        self.save_dir = save_dir
        self.square_detector = square_detector
        self._ensure_directory_exists()
    
    def _ensure_directory_exists(self):
        """Create screenshots directory if it doesn't exist."""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f"Created directory: {self.save_dir}")
            
    def clear_directory(self):
        """Delete all files in the screenshots directory."""
        if os.path.exists(self.save_dir):
            print(f"Cleaning up directory: {self.save_dir}")
            for filename in os.listdir(self.save_dir):
                file_path = os.path.join(self.save_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                        # print(f"Deleted: {filename}")
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
            print("Cleanup complete.")
    
    def capture_fullscreen(self):
        """
        Capture all screens (supports multi-monitor).
        
        Returns:
            PIL.Image: The captured screenshot as a PIL Image covering all monitors
        """
        with mss() as sct:
            # Capture all monitors (monitor 0 = virtual desktop spanning all monitors)
            monitor = sct.monitors[0]
            screenshot = sct.grab(monitor)
            
            # Convert to PIL Image
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            return img
    
    def capture_region_interactive(self):
        """
        Capture a screen region selected by the user interactively.
        
        Returns:
            tuple: (PIL.Image, tuple) - The captured region and its coordinates (x1, y1, x2, y2)
                   or (None, None) if cancelled
        """
        # First capture the full screen
        fullscreen = self.capture_fullscreen()
        
        # Show region selector with detector
        selector = RegionSelector(fullscreen, square_detector=self.square_detector)
        region = selector.get_region()
        
        if region is None:
            print("Screenshot cancelled")
            return None, None
        
        # Crop the selected region
        x1, y1, x2, y2 = region
        cropped = fullscreen.crop((x1, y1, x2, y2))
        
        print(f"Selected region: ({x1}, {y1}) to ({x2}, {y2})")
        return cropped, region
    
    def save_screenshot(self, img, prefix="screenshot"):
        """
        Save screenshot with timestamp-based filename.
        
        Args:
            img (PIL.Image): The screenshot image to save
            prefix (str): Prefix for the filename
            
        Returns:
            str: Path to the saved screenshot
        """
        if img is None:
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.png"
        filepath = os.path.join(self.save_dir, filename)
        
        img.save(filepath)
        print(f"Screenshot saved: {filepath}")
        return filepath
    
    def capture_and_save(self, prefix="screenshot"):
        """
        Capture screen region interactively and save in one operation.
        
        Args:
            prefix (str): Prefix for the filename
            
        Returns:
            tuple: (filepath, region) - Path to saved screenshot and region coordinates
                   or (None, None) if cancelled
        """
        img, region = self.capture_region_interactive()
        if img is None:
            return None, None
        filepath = self.save_screenshot(img, prefix)
        return filepath, region


if __name__ == "__main__":
    # Test the screenshot capture
    capture = ScreenshotCapture()
    print("Preparing screenshot...")
    filepath, region = capture.capture_and_save()
    if filepath:
        print(f"Screenshot saved to: {filepath}")
        if region:
            x1, y1, x2, y2 = region
            print(f"Region coordinates: ({x1}, {y1}) to ({x2}, {y2})")
    else:
        print("Screenshot not saved")
