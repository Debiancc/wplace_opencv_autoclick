import sys
from unittest.mock import MagicMock, patch
import pytest

@pytest.fixture(scope="module")
def main_module():
    """
    Patches sys.modules to mock dependencies, imports main, and yields the module.
    """
    mock_pynput = MagicMock()
    mock_keyboard = MagicMock()
    mock_pynput.keyboard = mock_keyboard
    mock_mouse = MagicMock()
    mock_pynput.mouse = mock_mouse

    mock_hotkey_listener = MagicMock()
    mock_screenshot_capture = MagicMock()
    mock_square_detector = MagicMock()

    modules_to_patch = {
        'pynput': mock_pynput,
        'pynput.keyboard': mock_keyboard,
        'pynput.mouse': mock_mouse,
        'hotkey_listener': mock_hotkey_listener,
        'screenshot_capture': mock_screenshot_capture,
        'square_detector': mock_square_detector,
        'PIL': MagicMock(),
        'win32gui': MagicMock()
    }

    with patch.dict(sys.modules, modules_to_patch):
        if 'main' in sys.modules:
            del sys.modules['main']
        import main
        yield main
        if 'main' in sys.modules:
            del sys.modules['main']

def test_main_default_args(main_module):
    with patch('main.ScreenshotApp') as mock_app_cls:
        mock_app_instance = mock_app_cls.return_value

        with patch.object(sys, 'argv', ['main.py']):
            main_module.main()
            mock_app_cls.assert_called_with(enable_square_detection=False, click_delay=0.5)
            mock_app_instance.run.assert_called_once()

def test_main_detect_squares(main_module):
    with patch('main.ScreenshotApp') as mock_app_cls:
        mock_app_instance = mock_app_cls.return_value

        with patch.object(sys, 'argv', ['main.py', '--detect-squares']):
            main_module.main()
            mock_app_cls.assert_called_with(enable_square_detection=True, click_delay=0.5)
            mock_app_instance.run.assert_called_once()

def test_main_custom_delay(main_module):
    with patch('main.ScreenshotApp') as mock_app_cls:
        mock_app_instance = mock_app_cls.return_value

        with patch.object(sys, 'argv', ['main.py', '--detect-squares', '--click-delay', '0.1']):
            main_module.main()
            mock_app_cls.assert_called_with(enable_square_detection=True, click_delay=0.1)
            mock_app_instance.run.assert_called_once()
