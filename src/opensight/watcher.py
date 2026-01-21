"""
Replay Folder Watchdog for CS2 Demo Files

Monitors the CS2 replays folder for new .dem files and automatically
triggers analysis when new replays are detected.

This eliminates the need for Steam Bot infrastructure by using a
local file system monitoring approach.

Optimizations:
- Debounce/coalesce file events (2 second default)
- File size stability checks before processing
- Cache detection to skip already-analyzed demos
"""

import hashlib
import json
import logging
import os
import platform
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from watchdog.events import FileCreatedEvent, FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


# Cache version - increment to invalidate old caches
CACHE_VERSION = 1


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
        replays_path = (
            steam_path / "Steam/steamapps/common/Counter-Strike Global Offensive/game/csgo/replays"
        )

        # Also check common alternative locations
        alternatives = [
            Path(
                "C:/Program Files/Steam/steamapps/common/Counter-Strike Global Offensive/game/csgo/replays"
            ),
            Path("D:/Steam/steamapps/common/Counter-Strike Global Offensive/game/csgo/replays"),
            Path(
                "D:/SteamLibrary/steamapps/common/Counter-Strike Global Offensive/game/csgo/replays"
            ),
        ]

        for alt in [replays_path] + alternatives:
            if alt.exists():
                return alt

        return replays_path  # Return default even if it doesn't exist

    elif system == "Darwin":  # macOS
        home = Path.home()
        return (
            home
            / "Library/Application Support/Steam/steamapps/common/Counter-Strike Global Offensive/game/csgo/replays"
        )

    elif system == "Linux":
        home = Path.home()
        # Check common Steam locations on Linux
        candidates = [
            home
            / ".steam/steam/steamapps/common/Counter-Strike Global Offensive/game/csgo/replays",
            home
            / ".local/share/Steam/steamapps/common/Counter-Strike Global Offensive/game/csgo/replays",
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
    from_cache: bool = False  # True if this is a cached/already-processed demo

    @property
    def filename(self) -> str:
        return self.file_path.name


@dataclass
class DemoCacheEntry:
    """Cache entry for an analyzed demo."""

    file_path: str
    file_size: int
    mtime: float
    analysis_timestamp: float
    file_hash: str = ""  # Optional SHA256 of first 64KB


class DemoCache:
    """
    Cache to track already-analyzed demo files.

    Stores file path, size, and modification time to detect changes.
    """

    def __init__(self, cache_dir: Path | None = None):
        """
        Initialize the demo cache.

        Args:
            cache_dir: Directory to store cache file. Defaults to ~/.opensight/
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".opensight"
        self.cache_dir = cache_dir
        self.cache_file = cache_dir / "demo_cache.json"
        self._cache: dict[str, DemoCacheEntry] = {}
        self._lock = threading.Lock()
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cache from disk."""
        if not self.cache_file.exists():
            return

        try:
            with open(self.cache_file) as f:
                data = json.load(f)

            # Check cache version
            if data.get("version") != CACHE_VERSION:
                logger.info("Cache version mismatch, clearing cache")
                self._cache = {}
                return

            # Load entries
            entries = data.get("entries", {})
            for path, entry_data in entries.items():
                self._cache[path] = DemoCacheEntry(
                    file_path=entry_data["file_path"],
                    file_size=entry_data["file_size"],
                    mtime=entry_data["mtime"],
                    analysis_timestamp=entry_data["analysis_timestamp"],
                    file_hash=entry_data.get("file_hash", ""),
                )

            logger.debug(f"Loaded {len(self._cache)} cached demos")

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to load cache: {e}")
            self._cache = {}

    def _save_cache(self) -> None:
        """Save cache to disk."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "version": CACHE_VERSION,
            "entries": {
                path: {
                    "file_path": entry.file_path,
                    "file_size": entry.file_size,
                    "mtime": entry.mtime,
                    "analysis_timestamp": entry.analysis_timestamp,
                    "file_hash": entry.file_hash,
                }
                for path, entry in self._cache.items()
            },
        }

        try:
            with open(self.cache_file, "w") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            logger.warning(f"Failed to save cache: {e}")

    def _compute_file_hash(self, path: Path) -> str:
        """Compute SHA256 of first 64KB of file for quick identity check."""
        try:
            with open(path, "rb") as f:
                data = f.read(65536)  # First 64KB
                return hashlib.sha256(data).hexdigest()[:16]
        except OSError:
            return ""

    def is_analyzed(self, path: Path) -> bool:
        """
        Check if a demo file has already been analyzed.

        Returns True if the file is in cache and hasn't changed.
        """
        with self._lock:
            str_path = str(path.absolute())
            if str_path not in self._cache:
                return False

            entry = self._cache[str_path]

            try:
                stat = path.stat()
                # Check if file has been modified
                if stat.st_size != entry.file_size:
                    logger.debug(f"Cache miss (size changed): {path.name}")
                    return False
                if stat.st_mtime != entry.mtime:
                    # Size same but mtime different - compute hash
                    current_hash = self._compute_file_hash(path)
                    if current_hash != entry.file_hash:
                        logger.debug(f"Cache miss (content changed): {path.name}")
                        return False
                    # Hash matches, update mtime in cache
                    entry.mtime = stat.st_mtime
                    self._save_cache()

                logger.debug(f"Cache hit: {path.name}")
                return True

            except OSError:
                return False

    def mark_analyzed(self, path: Path) -> None:
        """Mark a demo file as analyzed."""
        with self._lock:
            str_path = str(path.absolute())
            try:
                stat = path.stat()
                self._cache[str_path] = DemoCacheEntry(
                    file_path=str_path,
                    file_size=stat.st_size,
                    mtime=stat.st_mtime,
                    analysis_timestamp=time.time(),
                    file_hash=self._compute_file_hash(path),
                )
                self._save_cache()
                logger.debug(f"Cached: {path.name}")
            except OSError as e:
                logger.warning(f"Failed to cache {path}: {e}")

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache = {}
            if self.cache_file.exists():
                self.cache_file.unlink()
            logger.info("Cache cleared")

    def get_stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            return {
                "total_entries": len(self._cache),
                "cache_file": str(self.cache_file),
            }


class DemoFileHandler(FileSystemEventHandler):
    """
    Handler for demo file events with intelligent debouncing.

    Filters for .dem files and queues them for processing only when:
    - File meets minimum size requirement
    - File size has stabilized (not being written)
    - Debounce period has elapsed without new modifications
    """

    # Number of stability checks to perform
    STABILITY_CHECKS = 3
    # Time between stability checks (seconds)
    STABILITY_CHECK_INTERVAL = 0.3

    def __init__(
        self,
        event_queue: queue.Queue,
        min_file_size: int = 1024 * 1024,  # 1MB minimum
        debounce_seconds: float = 2.0,
        demo_cache: DemoCache | None = None,
    ):
        """
        Initialize the handler.

        Args:
            event_queue: Queue to put detected events
            min_file_size: Minimum file size in bytes to process
            debounce_seconds: Time to wait for file to finish writing
            demo_cache: Optional cache to skip already-analyzed demos
        """
        super().__init__()
        self.event_queue = event_queue
        self.min_file_size = min_file_size
        self.debounce_seconds = debounce_seconds
        self.demo_cache = demo_cache
        self._pending_files: dict[
            str, tuple[float, int]
        ] = {}  # path -> (last_event_time, event_count)
        self._lock = threading.Lock()

    def _is_demo_file(self, path: str) -> bool:
        """Check if the path is a demo file."""
        return path.lower().endswith(".dem")

    def _is_file_ready(self, path: Path) -> bool:
        """
        Check if a file is ready for processing using multiple stability checks.

        Performs multiple size checks over time to ensure the file isn't
        being actively written to.
        """
        if not path.exists():
            return False

        try:
            # Initial size check
            initial_size = path.stat().st_size
            if initial_size < self.min_file_size:
                logger.debug(f"File too small ({initial_size} bytes): {path.name}")
                return False

            # Perform multiple stability checks
            sizes = [initial_size]
            for _ in range(self.STABILITY_CHECKS):
                time.sleep(self.STABILITY_CHECK_INTERVAL)
                if not path.exists():
                    return False
                sizes.append(path.stat().st_size)

            # File is ready if all sizes are identical
            if len(set(sizes)) == 1:
                logger.debug(f"File stable at {initial_size} bytes: {path.name}")
                return True
            else:
                logger.debug(f"File size changing ({sizes}): {path.name}")
                return False

        except OSError as e:
            logger.debug(f"Error checking file readiness: {e}")
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
        """Handle file modification events with coalescing."""
        if event.is_directory:
            return

        if not self._is_demo_file(event.src_path):
            return

        with self._lock:
            if event.src_path in self._pending_files:
                # Coalesce: update timestamp and increment event count
                last_time, event_count = self._pending_files[event.src_path]
                self._pending_files[event.src_path] = (time.time(), event_count + 1)
                logger.debug(f"Coalesced event #{event_count + 1} for: {event.src_path}")
            else:
                # New file not created through on_created (e.g., file moved/copied)
                logger.debug(f"Demo file modified (new): {event.src_path}")
                self._schedule_processing(event.src_path, "modified")

    def _schedule_processing(self, file_path: str, event_type: str) -> None:
        """Schedule a file for processing after debounce period with coalescing."""
        with self._lock:
            self._pending_files[file_path] = (time.time(), 1)

        # Start a thread to wait for debounce and then process
        def process_after_debounce():
            time.sleep(self.debounce_seconds)

            with self._lock:
                pending = self._pending_files.get(file_path)
                if pending is None:
                    return

                last_modified, event_count = pending

                # Check if file was modified during debounce
                elapsed = time.time() - last_modified
                if elapsed < self.debounce_seconds:
                    # Re-schedule - file still being written
                    logger.debug(f"Re-scheduling (still active, {event_count} events): {file_path}")
                    threading.Thread(target=process_after_debounce, daemon=True).start()
                    return

                del self._pending_files[file_path]

            path = Path(file_path)

            # Check cache first
            if self.demo_cache and self.demo_cache.is_analyzed(path):
                logger.info(f"Skipped (cached): {path.name}")
                return

            if self._is_file_ready(path):
                event = DemoFileEvent(file_path=path, event_type=event_type, timestamp=time.time())
                self.event_queue.put(event)
                logger.info(f"Demo ready (coalesced {event_count} events): {path.name}")

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
            watcher.mark_analyzed(event.file_path)  # Cache for skip next time

        watcher.start()
        # Keep running...
        watcher.stop()
    """

    def __init__(
        self,
        watch_folder: Path | None = None,
        recursive: bool = False,
        use_cache: bool = True,
        debounce_seconds: float = 2.0,
    ):
        """
        Initialize the replay watcher.

        Args:
            watch_folder: Folder to watch (defaults to CS2 replays folder)
            recursive: Whether to watch subdirectories
            use_cache: Whether to use demo cache to skip already-analyzed files
            debounce_seconds: Time to wait for file to stabilize (default 2.0s)
        """
        if watch_folder is None:
            watch_folder = get_default_replays_folder()

        self.watch_folder = Path(watch_folder)
        self.recursive = recursive
        self.debounce_seconds = debounce_seconds

        # Initialize cache if enabled
        self._cache: DemoCache | None = DemoCache() if use_cache else None

        self._event_queue: queue.Queue[DemoFileEvent] = queue.Queue()
        self._observer: Observer | None = None
        self._callbacks: list[Callable[[DemoFileEvent], None]] = []
        self._running = False
        self._processor_thread: threading.Thread | None = None

    @property
    def cache(self) -> DemoCache | None:
        """Get the demo cache instance."""
        return self._cache

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
                watcher.mark_analyzed(event.file_path)
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

    def mark_analyzed(self, path: Path) -> None:
        """
        Mark a demo file as analyzed (add to cache).

        Call this after successfully processing a demo to skip it next time.

        Args:
            path: Path to the analyzed demo file
        """
        if self._cache:
            self._cache.mark_analyzed(path)

    def is_analyzed(self, path: Path) -> bool:
        """
        Check if a demo file has already been analyzed.

        Args:
            path: Path to check

        Returns:
            True if the demo is in cache and unchanged
        """
        if self._cache:
            return self._cache.is_analyzed(path)
        return False

    def clear_cache(self) -> None:
        """Clear all cached demo entries."""
        if self._cache:
            self._cache.clear()

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

        # Set up the observer with cache
        handler = DemoFileHandler(
            self._event_queue,
            debounce_seconds=self.debounce_seconds,
            demo_cache=self._cache,
        )
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
    folder: Path | None = None,
    callback: Callable[[DemoFileEvent], None] | None = None,
    blocking: bool = True,
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
