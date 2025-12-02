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
    """Interactive region selector using tkinter overlay."""
    
    def __init__(self, screenshot):
        """
        Initialize the region selector with a screenshot.
        
        Args:
            screenshot (PIL.Image): The full screen screenshot to select from
        """
        self.screenshot = screenshot
        self.selected_region = None
        
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
        
        # Selection rectangle
        self.start_x = None
        self.start_y = None
        self.rect = None
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.on_mouse_down)
        self.canvas.bind('<B1-Motion>', self.on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_up)
        
        # Bind escape key to cancel
        self.root.bind('<Escape>', self.on_cancel)
        
        # Instructions
        self.create_instructions()
    
    def create_instructions(self):
        """Display instructions on the overlay."""
        instruction_text = "Drag to select screenshot region | Press ESC to cancel"
        self.canvas.create_text(
            self.screen_width // 2,
            30,
            text=instruction_text,
            fill='white',
            font=('Arial', 14, 'bold')
        )
    
    def on_mouse_down(self, event):
        """Handle mouse button press."""
        self.start_x = event.x
        self.start_y = event.y
        
        # Create rectangle
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline='red', width=2
        )
    
    def on_mouse_drag(self, event):
        """Handle mouse drag to update selection rectangle."""
        if self.rect:
            self.canvas.coords(
                self.rect,
                self.start_x, self.start_y,
                event.x, event.y
            )
    
    def on_mouse_up(self, event):
        """Handle mouse button release to confirm selection."""
        end_x = event.x
        end_y = event.y
        
        # Calculate the region (ensure positive dimensions)
        x1 = min(self.start_x, end_x)
        y1 = min(self.start_y, end_y)
        x2 = max(self.start_x, end_x)
        y2 = max(self.start_y, end_y)
        
        # Store selected region
        self.selected_region = (x1, y1, x2, y2)
        
        # Close the window
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
    
    def __init__(self, save_dir="screenshots"):
        """
        Initialize the screenshot capture handler.
        
        Args:
            save_dir (str): Directory to save screenshots
        """
        self.save_dir = save_dir
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
        
        # Show region selector
        selector = RegionSelector(fullscreen)
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
