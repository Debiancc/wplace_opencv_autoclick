"""
Global hotkey listener using pynput library.
Monitors keyboard input and triggers callbacks on hotkey combinations.
Only triggers when Chrome is the active window.
"""
from pynput import keyboard
from pynput.keyboard import Key, KeyCode
import win32gui


class HotkeyListener:
    """Listen for global hotkey combinations and trigger callbacks."""
    
    def __init__(self, hotkey_callback=None, cancel_callback=None):
        """
        Initialize the hotkey listener.
        
        Args:
            hotkey_callback (callable): Function to call when hotkey is pressed
            cancel_callback (callable): Function to call when cancel key (ESC) is pressed
        """
        self.hotkey_callback = hotkey_callback
        self.cancel_callback = cancel_callback
        self.current_keys = set()
        self.listener = None
        
        # Define the hotkey: F2
        self.hotkey = Key.f2
    
    def _is_chrome_active(self):
        """
        Check if Chrome is the active window.
        
        Returns:
            bool: True if Chrome is the active window
        """
        try:
            # Get the handle of the foreground window
            hwnd = win32gui.GetForegroundWindow()
            # Get the window title
            window_title = win32gui.GetWindowText(hwnd)
            # Check if title contains 'chrome' (case-insensitive)
            is_chrome = 'chrome' in window_title.lower()
            return is_chrome
        except Exception as e:
            print(f"Error checking active window: {e}")
            return False
    
    def _on_press(self, key):
        """
        Handle key press events.
        Only triggers callback when F2 is pressed and Chrome is active.
        
        Args:
            key: The key that was pressed
        """
        # Check if F2 is pressed
        if key == self.hotkey:
            # Check if Chrome is the active window
            if self._is_chrome_active():
                print("✓ F2 pressed with Chrome active - triggering screenshot")
                if self.hotkey_callback:
                    self.hotkey_callback()
            else:
                print("✗ F2 pressed but Chrome is not active - ignoring")
    
    def _on_release(self, key):
        """
        Handle key release events.
        
        Args:
            key: The key that was released
        """
        # Check for Esc to cancel operation (but not exit listener)
        if key == Key.esc:
            if self.cancel_callback:
                print("\nEsc pressed. Triggering cancellation...")
                self.cancel_callback()
            else:
                print("\nEsc pressed (no cancel callback registered).")
    
    def start(self):
        """Start listening for hotkeys in a blocking manner."""
        print("Hotkey listener started.")
        print("Press F2 (when Chrome is active) to capture screenshot")
        print("Press Esc to exit")
        
        with keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        ) as self.listener:
            self.listener.join()
    
    def start_non_blocking(self):
        """Start listening for hotkeys in a non-blocking manner."""
        print("Hotkey listener started (non-blocking).")
        print("Press F2 (when Chrome is active) to capture screenshot")
        print("Press Esc to exit")
        
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self.listener.start()
        return self.listener
    
    def stop(self):
        """Stop the hotkey listener."""
        if self.listener:
            self.listener.stop()
            print("Hotkey listener stopped.")


if __name__ == "__main__":
    # Test the hotkey listener
    def test_callback():
        print("Hotkey triggered! (Test mode)")
    
    listener = HotkeyListener(hotkey_callback=test_callback)
    listener.start()
