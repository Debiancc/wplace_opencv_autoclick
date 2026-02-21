# Wplace.live Auto-Click Screenshot Tool

![Demo](assets/demo.png)

A powerful screenshot script with intelligent wplace.live square detection and automatic clicking capabilities.

## Features

- 📸 **Interactive Screenshot Capture** - Capture screenshots using `F2` hotkey
- 🔍 **Square Detection** - Automatically detect square points in captured images using OpenCV
- 🖱️ **Auto-Click** - Automatically click detected squares with configurable delay
- 🎯 **Real-time Detection** - Detection logic runs concurrently with screenshot selection
- 🛡️ **Safety Features** - Mouse movement interruption to prevent unwanted clicks
- 🎨 **Visualization** - Generate annotated images showing detected squares

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/wplace-opencv-autoclick.git
    cd wplace-opencv-autoclick
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Dependencies include: `opencv-python`, `pillow`, `pynput`, `mss`, `numpy`.*

## Usage

### Basic Screenshot Capture

```bash
python main.py
```

### With Square Detection

```bash
python main.py --detect-squares
```

### Custom Click Delay

```bash
python main.py --detect-squares --click-delay 0.5
```

### Hotkeys

- **F2** - Start screenshot capture (Note: This currently works best when Chrome is the active window)
- **ESC** - Cancel screenshot or interrupt clicking sequence
- **Mouse Movement** - Interrupt automatic clicking

## Project Structure

```
wplace_opencv_autoclick/
├── main.py                  # Main application entry point
├── screenshot_capture.py    # Screenshot capture logic
├── square_detector.py       # Square detection using OpenCV
├── hotkey_listener.py       # Keyboard hotkey handling
├── debug_detection.py       # Debug utilities
├── tests/                   # Unit tests
└── assets/                  # Demo images and resources
```

## How It Works

1. **Screenshot Capture**: Press `F2` to start interactive screenshot selection
2. **Square Detection**: The application analyzes the captured region for square patterns
3. **Coordinate Calculation**: Detected squares are mapped to absolute screen coordinates
4. **Auto-Click**: The mouse automatically clicks each detected square with a configurable delay
5. **Visualization**: Results are saved with visual annotations


## Development

### Running Tests

Run the test suite using `pytest`. This includes tests for square detection accuracy, hotkey handling, and screenshot capture logic.

```bash
pytest tests/
```

*Note: The tests use mocking for `pynput` and `win32gui` to allow running in headless or non-Windows environments.*

### Debug Mode

The application includes debug utilities in `debug_detection.py` for troubleshooting detection issues.

## Demo

A sample image is provided in `assets/demo.png` to test the detection logic. You can verify it by running the tests.

![Demo](assets/demo.png)

## Requirements

- Python 3.10+
- OpenCV
- Pillow
- pynput
- mss
- numpy
- pywin32 (Windows only)

## License

MIT
