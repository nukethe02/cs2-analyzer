"""
Replay Folder Watchdog for CS2 Demo Files

Monitors the CS2 replays folder for new .dem files and automatically
triggers analysis when new replays are detected.

This eliminates the need for Steam Bot infrastructure by using a
local file system monitoring approach.
"""

import logging
import os
import platform
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


def get_default_replays_folder() -> Path:
    """
    Get the default CS2 replays folder path based on the operating system.

    Returns:
        Path to the replays folder

    Raises:
        ValueError: If the OS is not supported or path cannot be determined
    """
    system = platform.system()

    if system == "Windows":
        # Default Steam installation path on Windows
        steam_path = Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)"))
        replays_path = steam_path / "Steam/steamapps/common/Counter-Strike Global Offensive/game/csgo/replays"

        # Also check common alternative locations
        alternatives = [
            Path("C:/Program Files/Steam/steamapps/common/Counter-Strike Global Offensive/game/csgo/replays"),
            Path("D:/Steam/steamapps/common/Counter-Strike Global Offensive/game/csgo/replays"),
            Path("D:/SteamLibrary/steamapps/common/Counter-Strike Global Offensive/game/csgo/replays"),
        ]

        for alt in [replays_path] + alternatives:
            if alt.exists():
                return alt

        return replays_path  # Return default even if it doesn't exist

    elif system == "Darwin":  # macOS
        home = Path.home()
        return home / "Library/Application Support/Steam/steamapps/common/Counter-Strike Global Offensive/game/csgo/replays"

    elif system == "Linux":
        home = Path.home()
        # Check common Steam locations on Linux
        candidates = [
            home / ".steam/steam/steamapps/common/Counter-Strike Global Offensive/game/csgo/replays",
            home / ".local/share/Steam/steamapps/common/Counter-Strike Global Offensive/game/csgo/replays",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return candidates[0]

    else:
        raise ValueError(f"Unsupported operating system: {system}")


@dataclass
class DemoFileEvent:
    """Event representing a new or modified demo file."""

    file_path: Path
    event_type: str  # "created" or "modified"
    timestamp: float

    @property
    def filename(self) -> str:
        return self.file_path.name


class DemoFileHandler(FileSystemEventHandler):
    """
    Handler for demo file events.

    Filters for .dem files and queues them for processing.
    """

    def __init__(
        self,
        event_queue: queue.Queue,
        min_file_size: int = 1024 * 1024,  # 1MB minimum
        debounce_seconds: float = 2.0
    ):
        """
        Initialize the handler.

        Args:
            event_queue: Queue to put detected events
            min_file_size: Minimum file size in bytes to process
            debounce_seconds: Time to wait for file to finish writing
        """
        super().__init__()
        self.event_queue = event_queue
        self.min_file_size = min_file_size
        self.debounce_seconds = debounce_seconds
        self._pending_files: dict[str, float] = {}
        self._lock = threading.Lock()

    def _is_demo_file(self, path: str) -> bool:
        """Check if the path is a demo file."""
        return path.lower().endswith(".dem")

    def _is_file_ready(self, path: Path) -> bool:
        """
        Check if a file is ready for processing.

        Ensures the file exists, meets minimum size, and isn't being written to.
        """
        if not path.exists():
            return False

        try:
            size = path.stat().st_size
            if size < self.min_file_size:
                return False

            # Check if file size is stable (not being written)
            time.sleep(0.5)
            new_size = path.stat().st_size
            return size == new_size

        except OSError:
            return False

    def on_created(self, event: FileCreatedEvent) -> None:
        """Handle file creation events."""
        if event.is_directory:
            return

        if not self._is_demo_file(event.src_path):
            return

        logger.debug(f"Demo file created: {event.src_path}")
        self._schedule_processing(event.src_path, "created")

    def on_modified(self, event: FileModifiedEvent) -> None:
        """Handle file modification events."""
        if event.is_directory:
            return

        if not self._is_demo_file(event.src_path):
            return

        # Only process modifications if we're already tracking this file
        with self._lock:
            if event.src_path in self._pending_files:
                self._pending_files[event.src_path] = time.time()

    def _schedule_processing(self, file_path: str, event_type: str) -> None:
        """Schedule a file for processing after debounce period."""
        with self._lock:
            self._pending_files[file_path] = time.time()

        # Start a thread to wait for debounce and then process
        def process_after_debounce():
            time.sleep(self.debounce_seconds)

            with self._lock:
                last_modified = self._pending_files.get(file_path)
                if last_modified is None:
                    return

                # Check if file was modified during debounce
                if time.time() - last_modified < self.debounce_seconds:
                    # Re-schedule
                    threading.Thread(target=process_after_debounce, daemon=True).start()
                    return

                del self._pending_files[file_path]

            path = Path(file_path)
            if self._is_file_ready(path):
                event = DemoFileEvent(
                    file_path=path,
                    event_type=event_type,
                    timestamp=time.time()
                )
                self.event_queue.put(event)
                logger.info(f"Demo file ready for processing: {path.name}")

        threading.Thread(target=process_after_debounce, daemon=True).start()


class ReplayWatcher:
    """
    Watches a folder for new CS2 replay files.

    Example usage:
        watcher = ReplayWatcher()

        @watcher.on_new_demo
        def handle_demo(event):
            print(f"New demo: {event.file_path}")
            # Process the demo...

        watcher.start()
        # Keep running...
        watcher.stop()
    """

    def __init__(
        self,
        watch_folder: Optional[Path] = None,
        recursive: bool = False
    ):
        """
        Initialize the replay watcher.

        Args:
            watch_folder: Folder to watch (defaults to CS2 replays folder)
            recursive: Whether to watch subdirectories
        """
        if watch_folder is None:
            watch_folder = get_default_replays_folder()

        self.watch_folder = Path(watch_folder)
        self.recursive = recursive

        self._event_queue: queue.Queue[DemoFileEvent] = queue.Queue()
        self._observer: Optional[Observer] = None
        self._callbacks: list[Callable[[DemoFileEvent], None]] = []
        self._running = False
        self._processor_thread: Optional[threading.Thread] = None

    def on_new_demo(self, callback: Callable[[DemoFileEvent], None]) -> Callable:
        """
        Decorator to register a callback for new demo files.

        Args:
            callback: Function to call when a new demo is detected

        Returns:
            The callback function (for decorator chaining)

        Example:
            @watcher.on_new_demo
            def process(event):
                parser = DemoParser(event.file_path)
                data = parser.parse()
        """
        self._callbacks.append(callback)
        return callback

    def add_callback(self, callback: Callable[[DemoFileEvent], None]) -> None:
        """
        Add a callback for new demo files.

        Args:
            callback: Function to call when a new demo is detected
        """
        self._callbacks.append(callback)

    def start(self, blocking: bool = False) -> None:
        """
        Start watching for new demo files.

        Args:
            blocking: If True, blocks until stop() is called
        """
        if self._running:
            logger.warning("Watcher is already running")
            return

        if not self.watch_folder.exists():
            logger.warning(f"Watch folder does not exist: {self.watch_folder}")
            logger.info("Creating watch folder...")
            self.watch_folder.mkdir(parents=True, exist_ok=True)

        self._running = True

        # Set up the observer
        handler = DemoFileHandler(self._event_queue)
        self._observer = Observer()
        self._observer.schedule(handler, str(self.watch_folder), recursive=self.recursive)

        # Start the event processor thread
        self._processor_thread = threading.Thread(target=self._process_events, daemon=True)
        self._processor_thread.start()

        # Start the observer
        self._observer.start()
        logger.info(f"Watching for demos in: {self.watch_folder}")

        if blocking:
            try:
                while self._running:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.stop()

    def stop(self) -> None:
        """Stop watching for demo files."""
        self._running = False

        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None

        logger.info("Replay watcher stopped")

    def _process_events(self) -> None:
        """Process events from the queue and call callbacks."""
        while self._running:
            try:
                event = self._event_queue.get(timeout=1)
            except queue.Empty:
                continue

            for callback in self._callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in callback: {e}")

    def scan_existing(self) -> list[Path]:
        """
        Scan for existing demo files in the watch folder.

        Returns:
            List of paths to existing .dem files
        """
        if not self.watch_folder.exists():
            return []

        pattern = "**/*.dem" if self.recursive else "*.dem"
        return list(self.watch_folder.glob(pattern))

    @property
    def is_running(self) -> bool:
        """Check if the watcher is currently running."""
        return self._running


def watch_replays(
    folder: Optional[Path] = None,
    callback: Optional[Callable[[DemoFileEvent], None]] = None,
    blocking: bool = True
) -> ReplayWatcher:
    """
    Convenience function to start watching for replays.

    Args:
        folder: Folder to watch (defaults to CS2 replays folder)
        callback: Optional callback for new demos
        blocking: If True, blocks until interrupted

    Returns:
        The ReplayWatcher instance
    """
    watcher = ReplayWatcher(folder)

    if callback:
        watcher.add_callback(callback)

    watcher.start(blocking=blocking)
    return watcher
