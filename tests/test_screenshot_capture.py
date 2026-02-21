import sys
from unittest.mock import MagicMock, patch
import pytest

@pytest.fixture(scope="module")
def screenshot_capture_module():
    """
    Patches sys.modules to mock mss, tkinter, and PIL, imports screenshot_capture,
    and yields the module. Cleans up afterwards.
    """
    # Mock mss
    mock_mss_pkg = MagicMock()
    mock_mss_class = MagicMock()
    mock_mss_instance = MagicMock()
    mock_mss_class.return_value.__enter__.return_value = mock_mss_instance
    mock_mss_pkg.mss = mock_mss_class

    # Mock tkinter
    mock_tkinter = MagicMock()

    # Mock PIL
    mock_PIL = MagicMock()
    mock_Image = MagicMock()
    mock_PIL.Image = mock_Image
    mock_ImageTk = MagicMock()
    mock_PIL.ImageTk = mock_ImageTk
    mock_ImageDraw = MagicMock()
    mock_PIL.ImageDraw = mock_ImageDraw

    modules_to_patch = {
        'mss': mock_mss_pkg,
        'tkinter': mock_tkinter,
        'PIL': mock_PIL
    }

    with patch.dict(sys.modules, modules_to_patch):
        if 'screenshot_capture' in sys.modules:
            del sys.modules['screenshot_capture']

        import screenshot_capture
        yield screenshot_capture

        if 'screenshot_capture' in sys.modules:
            del sys.modules['screenshot_capture']

@pytest.fixture
def capture_instance(screenshot_capture_module):
    """
    Returns a configured ScreenshotCapture instance.
    """
    with patch('os.makedirs') as mock_makedirs, \
         patch('os.path.exists', return_value=False) as mock_exists:
        capture = screenshot_capture_module.ScreenshotCapture(save_dir="test_screenshots")
        return capture

def test_init_creates_directory(screenshot_capture_module):
    with patch('os.makedirs') as mock_makedirs, \
         patch('os.path.exists', return_value=False) as mock_exists:
        screenshot_capture_module.ScreenshotCapture(save_dir="new_dir")
        mock_makedirs.assert_called_with("new_dir")

def test_clear_directory(capture_instance):
    with patch('os.path.exists', return_value=True), \
         patch('os.listdir', return_value=["file1.png", "file2.txt"]), \
         patch('os.path.isfile', return_value=True), \
         patch('os.unlink') as mock_unlink:

        capture_instance.clear_directory()
        assert mock_unlink.call_count == 2

def test_capture_fullscreen(capture_instance):
    # Retrieve the mock instances from sys.modules
    mock_mss_instance = sys.modules['mss'].mss.return_value.__enter__.return_value
    mock_Image = sys.modules['PIL'].Image

    # Setup mock behavior
    mock_mss_instance.monitors = [{}]
    mock_shot = MagicMock()
    mock_shot.bgra = b'\x00' * 100
    mock_shot.size = (10, 10)
    mock_mss_instance.grab.return_value = mock_shot

    mock_image = MagicMock()
    mock_Image.frombytes.return_value = mock_image

    result = capture_instance.capture_fullscreen()

    mock_mss_instance.grab.assert_called_once()
    mock_Image.frombytes.assert_called_once()
    assert result == mock_image

def test_save_screenshot(capture_instance):
    mock_image = MagicMock()
    # patch datetime inside the module
    with patch('screenshot_capture.datetime') as mock_datetime:
        mock_datetime.now.return_value.strftime.return_value = "20230101_120000"

        filepath = capture_instance.save_screenshot(mock_image)

        mock_image.save.assert_called_once()
        assert "screenshot_20230101_120000.png" in str(filepath)
        assert capture_instance.save_dir in str(filepath)

def test_capture_region_interactive_success(capture_instance):
    mock_fullscreen = MagicMock()
    mock_region_image = MagicMock()
    mock_coords = (10, 10, 50, 50)

    with patch.object(capture_instance, 'capture_fullscreen', return_value=mock_fullscreen), \
         patch('screenshot_capture.RegionSelector') as mock_selector_cls:

        mock_selector = mock_selector_cls.return_value
        mock_selector.get_region.return_value = mock_coords

        mock_fullscreen.crop.return_value = mock_region_image

        img, region = capture_instance.capture_region_interactive()

        assert img == mock_region_image
        assert region == mock_coords
        mock_fullscreen.crop.assert_called_with(mock_coords)

def test_capture_region_interactive_cancel(capture_instance):
    mock_fullscreen = MagicMock()

    with patch.object(capture_instance, 'capture_fullscreen', return_value=mock_fullscreen), \
         patch('screenshot_capture.RegionSelector') as mock_selector_cls:

        mock_selector = mock_selector_cls.return_value
        mock_selector.get_region.return_value = None

        img, region = capture_instance.capture_region_interactive()

        assert img is None
        assert region is None
