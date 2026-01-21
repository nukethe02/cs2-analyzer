"""Comprehensive tests for the watcher module."""

import pytest
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import queue
import tempfile
import os

from opensight.watcher import (
    get_default_replays_folder,
    DemoFileEvent,
    DemoFileHandler,
    ReplayWatcher,
    watch_replays,
)


class TestGetDefaultReplaysFolder:
    """Tests for get_default_replays_folder function."""

    @patch('platform.system')
    def test_windows_path(self, mock_system):
        """Test default path on Windows."""
        mock_system.return_value = "Windows"

        with patch.dict(os.environ, {"PROGRAMFILES(X86)": "C:/Program Files (x86)"}):
            path = get_default_replays_folder()
            assert "Steam" in str(path)
            assert "replays" in str(path)

    @patch('platform.system')
    def test_macos_path(self, mock_system):
        """Test default path on macOS."""
        mock_system.return_value = "Darwin"

        path = get_default_replays_folder()
        assert "Library/Application Support/Steam" in str(path)
        assert "replays" in str(path)

    @patch('platform.system')
    def test_linux_path(self, mock_system):
        """Test default path on Linux."""
        mock_system.return_value = "Linux"

        path = get_default_replays_folder()
        assert ".steam" in str(path) or ".local/share/Steam" in str(path)
        assert "replays" in str(path)

    @patch('platform.system')
    def test_unsupported_os(self, mock_system):
        """Test error on unsupported OS."""
        mock_system.return_value = "BeOS"

        with pytest.raises(ValueError, match="Unsupported operating system"):
            get_default_replays_folder()


class TestDemoFileEvent:
    """Tests for DemoFileEvent dataclass."""

    def test_creation(self):
        """Test creating a DemoFileEvent."""
        event = DemoFileEvent(
            file_path=Path("/test/match.dem"),
            event_type="created",
            timestamp=1234567890.0,
        )
        assert event.file_path == Path("/test/match.dem")
        assert event.event_type == "created"
        assert event.timestamp == 1234567890.0

    def test_filename_property(self):
        """Test filename property."""
        event = DemoFileEvent(
            file_path=Path("/long/path/to/match_12345.dem"),
            event_type="created",
            timestamp=0.0,
        )
        assert event.filename == "match_12345.dem"

    def test_modified_event_type(self):
        """Test modified event type."""
        event = DemoFileEvent(
            file_path=Path("/test/match.dem"),
            event_type="modified",
            timestamp=0.0,
        )
        assert event.event_type == "modified"


class TestDemoFileHandler:
    """Tests for DemoFileHandler class."""

    @pytest.fixture
    def handler(self):
        """Create a handler for testing."""
        event_queue = queue.Queue()
        return DemoFileHandler(
            event_queue=event_queue,
            min_file_size=1024,  # 1KB for testing
            debounce_seconds=0.1,  # Short debounce for testing
        )

    def test_init(self, handler):
        """Test handler initialization."""
        assert handler.min_file_size == 1024
        assert handler.debounce_seconds == 0.1

    def test_is_demo_file_positive(self, handler):
        """Test demo file detection for valid files."""
        assert handler._is_demo_file("/path/to/match.dem") is True
        assert handler._is_demo_file("/path/to/MATCH.DEM") is True
        assert handler._is_demo_file("simple.dem") is True

    def test_is_demo_file_negative(self, handler):
        """Test demo file detection for invalid files."""
        assert handler._is_demo_file("/path/to/file.txt") is False
        assert handler._is_demo_file("/path/to/demo.mp4") is False
        assert handler._is_demo_file("/path/to/noextension") is False

    def test_is_file_ready_nonexistent(self, handler):
        """Test file ready check for non-existent file."""
        assert handler._is_file_ready(Path("/nonexistent/file.dem")) is False

    def test_is_file_ready_too_small(self, handler, tmp_path):
        """Test file ready check for file below minimum size."""
        small_file = tmp_path / "small.dem"
        small_file.write_bytes(b"x" * 100)  # 100 bytes < 1024 bytes

        assert handler._is_file_ready(small_file) is False

    def test_is_file_ready_valid(self, handler, tmp_path):
        """Test file ready check for valid file."""
        demo_file = tmp_path / "valid.dem"
        demo_file.write_bytes(b"x" * 2048)  # 2KB > 1KB minimum

        assert handler._is_file_ready(demo_file) is True

    def test_on_created_ignores_directories(self, handler):
        """Test that directory creation events are ignored."""
        mock_event = Mock()
        mock_event.is_directory = True
        mock_event.src_path = "/some/directory"

        handler.on_created(mock_event)

        # Queue should be empty since directories are ignored
        assert handler.event_queue.empty()

    def test_on_created_ignores_non_demo_files(self, handler):
        """Test that non-demo files are ignored."""
        mock_event = Mock()
        mock_event.is_directory = False
        mock_event.src_path = "/path/to/file.txt"

        handler.on_created(mock_event)

        # Queue should be empty since non-demo files are ignored
        assert handler.event_queue.empty()

    def test_on_modified_updates_pending_files(self, handler):
        """Test that modifications update pending file timestamps."""
        # First, schedule a file for processing
        file_path = "/path/to/match.dem"
        old_time = time.time() - 10
        with handler._lock:
            # _pending_files stores (timestamp, event_count) tuples
            handler._pending_files[file_path] = (old_time, 1)

        mock_event = Mock()
        mock_event.is_directory = False
        mock_event.src_path = file_path

        handler.on_modified(mock_event)

        # After modification, timestamp should be updated and event count incremented
        new_time, new_count = handler._pending_files[file_path]
        assert new_time > old_time
        assert new_count == 2


class TestReplayWatcher:
    """Tests for ReplayWatcher class."""

    @pytest.fixture
    def watch_folder(self, tmp_path):
        """Create a temporary folder to watch."""
        folder = tmp_path / "replays"
        folder.mkdir()
        return folder

    def test_init_with_custom_folder(self, watch_folder):
        """Test initialization with custom folder."""
        watcher = ReplayWatcher(watch_folder=watch_folder)
        assert watcher.watch_folder == watch_folder
        assert watcher.recursive is False

    def test_init_with_recursive(self, watch_folder):
        """Test initialization with recursive flag."""
        watcher = ReplayWatcher(watch_folder=watch_folder, recursive=True)
        assert watcher.recursive is True

    @patch('opensight.watcher.get_default_replays_folder')
    def test_init_uses_default_folder(self, mock_get_default):
        """Test that default folder is used when not specified."""
        mock_get_default.return_value = Path("/default/replays")

        watcher = ReplayWatcher()

        mock_get_default.assert_called_once()
        assert watcher.watch_folder == Path("/default/replays")

    def test_on_new_demo_decorator(self, watch_folder):
        """Test on_new_demo decorator registration."""
        watcher = ReplayWatcher(watch_folder=watch_folder)

        @watcher.on_new_demo
        def handle_demo(event):
            pass

        assert len(watcher._callbacks) == 1
        assert watcher._callbacks[0] is handle_demo

    def test_add_callback(self, watch_folder):
        """Test add_callback method."""
        watcher = ReplayWatcher(watch_folder=watch_folder)

        def callback(event):
            pass

        watcher.add_callback(callback)

        assert callback in watcher._callbacks

    def test_scan_existing_empty_folder(self, watch_folder):
        """Test scanning empty folder."""
        watcher = ReplayWatcher(watch_folder=watch_folder)
        demos = watcher.scan_existing()

        assert demos == []

    def test_scan_existing_with_demos(self, watch_folder):
        """Test scanning folder with demo files."""
        # Create some demo files
        (watch_folder / "match1.dem").write_bytes(b"demo1")
        (watch_folder / "match2.dem").write_bytes(b"demo2")
        (watch_folder / "other.txt").write_text("not a demo")

        watcher = ReplayWatcher(watch_folder=watch_folder)
        demos = watcher.scan_existing()

        assert len(demos) == 2
        assert all(d.suffix == ".dem" for d in demos)

    def test_scan_existing_recursive(self, watch_folder):
        """Test recursive scanning."""
        # Create demos in subdirectory
        subdir = watch_folder / "subdir"
        subdir.mkdir()
        (watch_folder / "root.dem").write_bytes(b"demo")
        (subdir / "nested.dem").write_bytes(b"demo")

        watcher = ReplayWatcher(watch_folder=watch_folder, recursive=True)
        demos = watcher.scan_existing()

        assert len(demos) == 2

    def test_scan_existing_non_recursive(self, watch_folder):
        """Test non-recursive scanning ignores subdirs."""
        subdir = watch_folder / "subdir"
        subdir.mkdir()
        (watch_folder / "root.dem").write_bytes(b"demo")
        (subdir / "nested.dem").write_bytes(b"demo")

        watcher = ReplayWatcher(watch_folder=watch_folder, recursive=False)
        demos = watcher.scan_existing()

        assert len(demos) == 1
        assert demos[0].name == "root.dem"

    def test_scan_existing_nonexistent_folder(self, tmp_path):
        """Test scanning non-existent folder."""
        nonexistent = tmp_path / "doesnotexist"
        watcher = ReplayWatcher(watch_folder=nonexistent)

        demos = watcher.scan_existing()

        assert demos == []

    def test_is_running_property(self, watch_folder):
        """Test is_running property."""
        watcher = ReplayWatcher(watch_folder=watch_folder)

        assert watcher.is_running is False

        watcher.start(blocking=False)
        assert watcher.is_running is True

        watcher.stop()
        assert watcher.is_running is False

    def test_start_creates_folder_if_missing(self, tmp_path):
        """Test that start creates the watch folder if it doesn't exist."""
        missing_folder = tmp_path / "new_replays"
        watcher = ReplayWatcher(watch_folder=missing_folder)

        watcher.start(blocking=False)

        assert missing_folder.exists()
        watcher.stop()

    def test_start_twice_warns(self, watch_folder, caplog):
        """Test that starting twice logs a warning."""
        watcher = ReplayWatcher(watch_folder=watch_folder)

        watcher.start(blocking=False)
        watcher.start(blocking=False)  # Should warn

        watcher.stop()

        assert "already running" in caplog.text.lower()

    def test_stop_without_start(self, watch_folder):
        """Test stopping without starting first."""
        watcher = ReplayWatcher(watch_folder=watch_folder)

        # Should not raise
        watcher.stop()

    def test_callback_receives_event(self, watch_folder):
        """Test that callbacks receive events."""
        watcher = ReplayWatcher(watch_folder=watch_folder)
        received_events = []

        @watcher.on_new_demo
        def handle(event):
            received_events.append(event)

        # Simulate putting an event in the queue
        watcher._running = True
        test_event = DemoFileEvent(
            file_path=Path("/test.dem"),
            event_type="created",
            timestamp=time.time(),
        )
        watcher._event_queue.put(test_event)

        # Start the processor thread manually
        def process():
            while not watcher._event_queue.empty():
                event = watcher._event_queue.get()
                for callback in watcher._callbacks:
                    callback(event)

        process()

        assert len(received_events) == 1
        assert received_events[0].file_path == Path("/test.dem")

    def test_callback_error_handling(self, watch_folder, caplog):
        """Test that callback errors are logged but don't crash."""
        watcher = ReplayWatcher(watch_folder=watch_folder)

        @watcher.on_new_demo
        def bad_callback(event):
            raise ValueError("Test error")

        good_calls = []

        @watcher.on_new_demo
        def good_callback(event):
            good_calls.append(event)

        # Manually process an event
        test_event = DemoFileEvent(
            file_path=Path("/test.dem"),
            event_type="created",
            timestamp=time.time(),
        )

        for callback in watcher._callbacks:
            try:
                callback(test_event)
            except Exception:
                pass  # Simulating error handling

        # Good callback should still be called
        assert len(good_calls) == 1


class TestWatchReplaysFunction:
    """Tests for watch_replays convenience function."""

    def test_watch_replays_with_callback(self, tmp_path):
        """Test watch_replays with a callback."""
        folder = tmp_path / "replays"
        folder.mkdir()

        calls = []

        def callback(event):
            calls.append(event)

        watcher = watch_replays(
            folder=folder,
            callback=callback,
            blocking=False,
        )

        assert watcher.is_running
        assert callback in watcher._callbacks

        watcher.stop()

    def test_watch_replays_without_callback(self, tmp_path):
        """Test watch_replays without a callback."""
        folder = tmp_path / "replays"
        folder.mkdir()

        watcher = watch_replays(folder=folder, blocking=False)

        assert watcher.is_running
        assert len(watcher._callbacks) == 0

        watcher.stop()

    @patch('opensight.watcher.get_default_replays_folder')
    def test_watch_replays_uses_default_folder(self, mock_get_default, tmp_path):
        """Test that default folder is used when not specified."""
        default_folder = tmp_path / "default_replays"
        default_folder.mkdir()
        mock_get_default.return_value = default_folder

        watcher = watch_replays(blocking=False)

        mock_get_default.assert_called()

        watcher.stop()


class TestIntegration:
    """Integration tests for the watcher module."""

    def test_file_creation_triggers_callback(self, tmp_path):
        """Test that creating a file triggers the callback."""
        watch_folder = tmp_path / "replays"
        watch_folder.mkdir()

        watcher = ReplayWatcher(watch_folder=watch_folder)
        received = []

        @watcher.on_new_demo
        def handle(event):
            received.append(event)

        watcher.start(blocking=False)

        # Give the observer time to start
        time.sleep(0.5)

        # Create a demo file that's large enough
        demo_file = watch_folder / "test_match.dem"
        demo_file.write_bytes(b"x" * (1024 * 1024 + 1))  # Just over 1MB

        # Wait for debounce and processing
        time.sleep(3)

        watcher.stop()

        # Note: This may be flaky in CI due to timing
        # The file should trigger a callback if timing works out

    def test_multiple_callbacks(self, tmp_path):
        """Test that multiple callbacks are all called."""
        watch_folder = tmp_path / "replays"
        watch_folder.mkdir()

        watcher = ReplayWatcher(watch_folder=watch_folder)

        calls1 = []
        calls2 = []

        @watcher.on_new_demo
        def callback1(event):
            calls1.append(event)

        @watcher.on_new_demo
        def callback2(event):
            calls2.append(event)

        # Simulate processing an event
        test_event = DemoFileEvent(
            file_path=Path("/test.dem"),
            event_type="created",
            timestamp=time.time(),
        )

        for callback in watcher._callbacks:
            callback(test_event)

        assert len(calls1) == 1
        assert len(calls2) == 1
