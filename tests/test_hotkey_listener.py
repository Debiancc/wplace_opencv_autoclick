import sys
import importlib
from unittest.mock import MagicMock, patch
import pytest

# Define mocks outside to be reusable or define inside fixture
# We need to mock pynput and win32gui because they are not available/usable in this environment

@pytest.fixture(scope="module")
def hotkey_listener_module():
    """
    Patches sys.modules to mock pynput and win32gui, imports hotkey_listener,
    and yields the module. Cleans up afterwards.
    """
    mock_pynput = MagicMock()
    mock_keyboard = MagicMock()
    mock_pynput.keyboard = mock_keyboard

    # Mock Key enum
    class MockKey:
        f2 = 'f2'
        esc = 'esc'
    mock_keyboard.Key = MockKey
    mock_keyboard.Listener = MagicMock()

    mock_win32gui = MagicMock()

    # Create a dict of modules to patch
    modules_to_patch = {
        'pynput': mock_pynput,
        'pynput.keyboard': mock_keyboard,
        'win32gui': mock_win32gui
    }

    with patch.dict(sys.modules, modules_to_patch):
        if 'hotkey_listener' in sys.modules:
            del sys.modules['hotkey_listener']

        import hotkey_listener
        yield hotkey_listener

        # Cleanup
        if 'hotkey_listener' in sys.modules:
            del sys.modules['hotkey_listener']

@pytest.fixture
def listener(hotkey_listener_module):
    """
    Returns an instance of HotkeyListener with mocked callbacks.
    """
    hotkey_callback = MagicMock()
    cancel_callback = MagicMock()
    listener = hotkey_listener_module.HotkeyListener(
        hotkey_callback=hotkey_callback,
        cancel_callback=cancel_callback
    )
    # Attach mocks to instance for verification in tests
    listener.mock_hotkey_callback = hotkey_callback
    listener.mock_cancel_callback = cancel_callback
    return listener

def test_initialization(listener, hotkey_listener_module):
    # Need to access MockKey from the module's mocked pynput
    # But hotkey_listener_module.pynput is not exposed easily unless we inspect sys.modules or the module itself
    # But we can check values
    assert listener.hotkey_callback == listener.mock_hotkey_callback
    assert listener.cancel_callback == listener.mock_cancel_callback
    # The key is 'f2' because of our MockKey
    assert listener.hotkey == 'f2'

def test_is_chrome_active_always_true(listener):
    # Current implementation returns True always inside try block (when win32gui exists)
    assert listener._is_chrome_active() is True

def test_on_press_f2_triggers_callback(listener):
    # Mock _is_chrome_active to ensure it's called
    with patch.object(listener, '_is_chrome_active', return_value=True):
        listener._on_press('f2')
        listener.mock_hotkey_callback.assert_called_once()

def test_on_press_f2_no_callback_if_chrome_not_active(listener):
    # Even though current implementation always returns True, we can patch it to return False
    # to test the logic flow
    with patch.object(listener, '_is_chrome_active', return_value=False):
        listener._on_press('f2')
        listener.mock_hotkey_callback.assert_not_called()

def test_on_press_other_key_does_nothing(listener):
    listener._on_press('other_key')
    listener.mock_hotkey_callback.assert_not_called()

def test_on_release_esc_triggers_cancel(listener):
    listener._on_release('esc')
    listener.mock_cancel_callback.assert_called_once()

def test_on_release_other_key_does_nothing(listener):
    listener._on_release('other_key')
    listener.mock_cancel_callback.assert_not_called()

def test_start_starts_listener(listener):
    # We need to access the mock_keyboard.Listener used by the module
    mock_listener_cls = sys.modules['pynput.keyboard'].Listener

    listener.start()
    mock_listener_cls.assert_called()
    mock_listener_cls.return_value.__enter__.return_value.join.assert_called_once()

def test_start_non_blocking_starts_listener(listener):
    mock_listener_cls = sys.modules['pynput.keyboard'].Listener

    listener_instance = listener.start_non_blocking()
    mock_listener_cls.assert_called()
    mock_listener_cls.return_value.start.assert_called_once()
    assert listener_instance == mock_listener_cls.return_value

def test_stop_stops_listener(listener):
    listener.listener = MagicMock()
    listener.stop()
    listener.listener.stop.assert_called_once()

def test_no_win32gui_behavior():
    """Test behavior when win32gui is not importable."""
    # We need to simulate the environment where win32gui is missing
    # But preserve pynput

    # 1. Prepare mocks for pynput
    mock_pynput = MagicMock()
    mock_keyboard = MagicMock()
    mock_pynput.keyboard = mock_keyboard
    mock_keyboard.Key = MagicMock()
    mock_keyboard.Listener = MagicMock()

    # 2. Patch sys.modules WITHOUT win32gui
    modules_to_patch = {
        'pynput': mock_pynput,
        'pynput.keyboard': mock_keyboard,
    }

    # Also ensure win32gui is NOT in sys.modules
    # We use a new patch context to isolate this test
    with patch.dict(sys.modules, modules_to_patch):
        if 'win32gui' in sys.modules:
            del sys.modules['win32gui']

        # Reload hotkey_listener
        if 'hotkey_listener' in sys.modules:
            del sys.modules['hotkey_listener']

        import hotkey_listener
        importlib.reload(hotkey_listener)

        # Verify win32gui is None in the module
        assert hotkey_listener.win32gui is None

        # Create listener
        listener = hotkey_listener.HotkeyListener()

        # Verify _is_chrome_active returns True
        assert listener._is_chrome_active() is True
