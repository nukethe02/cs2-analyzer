"""
Performance profiling and timing utilities for OpenSight.

Provides:
- stage_timer: Context manager for timing individual stages
- Profiler: cProfile-based profiling for production workloads
- SlowJobLogger: Log details about slow demo analyses
- TimingStats: Collect and report timing statistics
"""

from __future__ import annotations

import cProfile
import io
import logging
import pstats
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Default threshold for slow job logging (seconds)
DEFAULT_SLOW_THRESHOLD_SECONDS = 60.0


@dataclass
class StageTime:
    """Timing result for a single stage."""

    name: str
    duration_seconds: float
    start_time: float
    end_time: float

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return self.duration_seconds * 1000


@dataclass
class TimingStats:
    """Collection of timing statistics for a full analysis run."""

    stages: list[StageTime] = field(default_factory=list)
    total_seconds: float = 0.0
    file_path: str | None = None
    file_size_bytes: int | None = None
    tick_count: int | None = None
    metrics_enabled: list[str] = field(default_factory=list)

    def add_stage(self, name: str, duration: float, start: float, end: float) -> None:
        """Add a timing stage."""
        self.stages.append(
            StageTime(name=name, duration_seconds=duration, start_time=start, end_time=end)
        )

    @property
    def total_ms(self) -> float:
        """Total time in milliseconds."""
        return self.total_seconds * 1000

    @property
    def file_size_mb(self) -> float | None:
        """File size in megabytes."""
        if self.file_size_bytes is None:
            return None
        return self.file_size_bytes / (1024 * 1024)

    def get_stage(self, name: str) -> StageTime | None:
        """Get timing for a specific stage."""
        for stage in self.stages:
            if stage.name == name:
                return stage
        return None

    def summary(self) -> dict[str, Any]:
        """Generate summary dictionary."""
        return {
            "total_seconds": round(self.total_seconds, 3),
            "total_ms": round(self.total_ms, 1),
            "file_size_mb": round(self.file_size_mb, 2) if self.file_size_mb else None,
            "tick_count": self.tick_count,
            "metrics_enabled": self.metrics_enabled,
            "stages": {
                s.name: {
                    "seconds": round(s.duration_seconds, 3),
                    "ms": round(s.duration_ms, 1),
                    "percent": (
                        round(s.duration_seconds / self.total_seconds * 100, 1)
                        if self.total_seconds > 0
                        else 0
                    ),
                }
                for s in self.stages
            },
        }

    def format_report(self, show_percentages: bool = True) -> str:
        """Format a human-readable timing report."""
        lines = []
        lines.append("=" * 60)
        lines.append("TIMING REPORT")
        lines.append("=" * 60)

        if self.file_path:
            lines.append(f"File: {self.file_path}")
        if self.file_size_mb:
            lines.append(f"Size: {self.file_size_mb:.2f} MB")
        if self.tick_count:
            lines.append(f"Ticks: {self.tick_count:,}")
        if self.metrics_enabled:
            lines.append(f"Metrics: {', '.join(self.metrics_enabled)}")

        lines.append("-" * 60)
        lines.append(f"{'Stage':<35} {'Time':>10} {'%':>8}")
        lines.append("-" * 60)

        for stage in self.stages:
            if self.total_seconds > 0:
                pct = stage.duration_seconds / self.total_seconds * 100
            else:
                pct = 0
            pct_str = f"{pct:>7.1f}%" if show_percentages else ""
            lines.append(f"{stage.name:<35} {stage.duration_seconds:>9.3f}s {pct_str}")

        lines.append("-" * 60)
        lines.append(f"{'TOTAL':<35} {self.total_seconds:>9.3f}s")
        lines.append("=" * 60)

        return "\n".join(lines)


class TimingCollector:
    """Collects timing data during analysis.

    Thread-safe collector for aggregating timing measurements
    from multiple stages during demo analysis.
    """

    def __init__(self) -> None:
        self._stats = TimingStats()
        self._start_time: float | None = None
        self._enabled = True

    @property
    def enabled(self) -> bool:
        """Check if timing collection is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable timing collection."""
        self._enabled = value

    def start(self) -> None:
        """Start overall timing."""
        self._start_time = time.perf_counter()

    def finish(self) -> None:
        """Finish overall timing and calculate total."""
        if self._start_time is not None:
            self._stats.total_seconds = time.perf_counter() - self._start_time

    def set_file_info(
        self, path: str | None = None, size_bytes: int | None = None, tick_count: int | None = None
    ) -> None:
        """Set file information for the analysis."""
        if path:
            self._stats.file_path = str(path)
        if size_bytes is not None:
            self._stats.file_size_bytes = size_bytes
        if tick_count is not None:
            self._stats.tick_count = tick_count

    def add_metric(self, metric_name: str) -> None:
        """Record that a metric was enabled."""
        if metric_name not in self._stats.metrics_enabled:
            self._stats.metrics_enabled.append(metric_name)

    @contextmanager
    def time_stage(self, name: str) -> Generator[None, None, None]:
        """Context manager to time a specific stage."""
        if not self._enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            duration = end - start
            self._stats.add_stage(name, duration, start, end)
            logger.debug(f"{name} took {duration:.3f}s")

    def record_stage(self, name: str, duration: float) -> None:
        """Manually record a stage timing."""
        if self._enabled:
            now = time.perf_counter()
            self._stats.add_stage(name, duration, now - duration, now)

    def get_stats(self) -> TimingStats:
        """Get the collected timing statistics."""
        return self._stats

    def reset(self) -> None:
        """Reset the collector for a new analysis."""
        self._stats = TimingStats()
        self._start_time = None


# Global timing collector (can be replaced per-request/per-analysis)
_global_collector: TimingCollector | None = None


def get_timing_collector() -> TimingCollector | None:
    """Get the current global timing collector."""
    return _global_collector


def set_timing_collector(collector: TimingCollector | None) -> None:
    """Set the global timing collector."""
    global _global_collector
    _global_collector = collector


@contextmanager
def stage_timer(name: str) -> Generator[None, None, None]:
    """Context manager to time a named stage.

    Usage:
        with stage_timer("parsing"):
            data = parser.parse()

        with stage_timer("ttd_calculation"):
            ttd_results = compute_ttd(data)

    Timing is logged to the debug level and added to the global
    collector if one is set.

    Args:
        name: Human-readable name for this stage

    Yields:
        None
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        logger.debug(f"{name} took {duration:.3f}s")

        # Record to global collector if available
        collector = get_timing_collector()
        if collector and collector.enabled:
            collector.record_stage(name, duration)


def timed(name: str | None = None) -> Callable:
    """Decorator to time a function.

    Usage:
        @timed("parse_kills")
        def parse_kills(self):
            ...

        @timed()  # Uses function name
        def compute_ttd(self):
            ...

    Args:
        name: Optional name for the timing (defaults to function name)

    Returns:
        Decorated function that records timing
    """

    def decorator(func: Callable) -> Callable:
        stage_name = name or func.__name__

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with stage_timer(stage_name):
                return func(*args, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator


class Profiler:
    """cProfile-based profiler for production workloads.

    Provides detailed CPU profiling with hotspot analysis.

    Usage:
        profiler = Profiler()
        profiler.start()
        # ... do work ...
        profiler.stop()
        profiler.print_stats(limit=20)

    Or as context manager:
        with Profiler() as p:
            # ... do work ...
        p.print_stats()
    """

    def __init__(self, sort_by: str = "cumulative") -> None:
        """Initialize profiler.

        Args:
            sort_by: How to sort stats. Options: cumulative, time, calls
        """
        self._profiler: cProfile.Profile | None = None
        self._stats: pstats.Stats | None = None
        self._sort_by = sort_by

    def start(self) -> None:
        """Start profiling."""
        self._profiler = cProfile.Profile()
        self._profiler.enable()

    def stop(self) -> None:
        """Stop profiling and generate stats."""
        if self._profiler:
            self._profiler.disable()
            stream = io.StringIO()
            self._stats = pstats.Stats(self._profiler, stream=stream)
            self._stats.sort_stats(self._sort_by)

    def __enter__(self) -> Profiler:
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.stop()

    def print_stats(self, limit: int = 30) -> None:
        """Print profiling statistics.

        Args:
            limit: Maximum number of entries to show
        """
        if self._stats:
            self._stats.print_stats(limit)

    def get_stats_string(self, limit: int = 30) -> str:
        """Get profiling statistics as a string.

        Args:
            limit: Maximum number of entries to show

        Returns:
            Formatted string with profiling stats
        """
        if not self._stats:
            return "No profiling data available"

        stream = io.StringIO()
        stats = pstats.Stats(self._profiler, stream=stream)
        stats.sort_stats(self._sort_by)
        stats.print_stats(limit)
        return stream.getvalue()

    def get_hotspots(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get the top hotspots as structured data.

        Args:
            limit: Maximum number of hotspots to return

        Returns:
            List of hotspot dictionaries with function info and timing
        """
        if not self._stats:
            return []

        hotspots = []
        # pstats stores data as:
        # {(file, line, func): (pcalls, ncalls, tottime, cumtime, callers)}
        for (file, line, func), stat_tuple in list(self._stats.stats.items())[:limit]:
            # stat_tuple format:
            # (primitive_calls, total_calls, total_time, cumulative_time, callers)
            pcalls, ncalls, tottime, cumtime, _callers = stat_tuple
            hotspots.append(
                {
                    "file": file,
                    "line": line,
                    "function": func,
                    "calls": ncalls,
                    "total_time": tottime,
                    "cumulative_time": cumtime,
                    "time_per_call": tottime / ncalls if ncalls > 0 else 0,
                }
            )

        return hotspots

    def save(self, path: Path) -> None:
        """Save profiling data to a file.

        Args:
            path: Path to save the profiling data
        """
        if self._profiler:
            self._profiler.dump_stats(str(path))


class SlowJobLogger:
    """Logger for demos that take too long to analyze.

    Automatically logs details about slow analyses to help identify
    optimization opportunities.

    Usage:
        slow_logger = SlowJobLogger(threshold_seconds=60)

        with slow_logger.track(demo_path, file_size, tick_count):
            # ... analyze demo ...
            slow_logger.add_metric("ttd", enabled=True)
            slow_logger.add_metric("cp", enabled=True)
    """

    def __init__(
        self,
        threshold_seconds: float = DEFAULT_SLOW_THRESHOLD_SECONDS,
        log_level: int = logging.WARNING,
    ) -> None:
        """Initialize slow job logger.

        Args:
            threshold_seconds: Threshold above which to log job details
            log_level: Logging level for slow job reports
        """
        self.threshold_seconds = threshold_seconds
        self.log_level = log_level
        self._current_job: dict[str, Any] | None = None
        self._start_time: float | None = None

    @contextmanager
    def track(
        self,
        file_path: str | None = None,
        file_size_bytes: int | None = None,
        tick_count: int | None = None,
    ) -> Generator[None, None, None]:
        """Track a job for slow logging.

        Args:
            file_path: Path to the demo file
            file_size_bytes: Size of the demo file in bytes
            tick_count: Number of ticks in the demo

        Yields:
            None
        """
        self._current_job = {
            "file_path": str(file_path) if file_path else None,
            "file_size_bytes": file_size_bytes,
            "file_size_mb": file_size_bytes / (1024 * 1024) if file_size_bytes else None,
            "tick_count": tick_count,
            "metrics_enabled": [],
        }
        self._start_time = time.perf_counter()

        try:
            yield
        finally:
            duration = time.perf_counter() - self._start_time

            if duration > self.threshold_seconds:
                self._log_slow_job(duration)

            self._current_job = None
            self._start_time = None

    def add_metric(self, metric_name: str, enabled: bool = True) -> None:
        """Record that a metric was enabled for the current job.

        Args:
            metric_name: Name of the metric
            enabled: Whether the metric was enabled
        """
        if self._current_job and enabled:
            self._current_job["metrics_enabled"].append(metric_name)

    def _log_slow_job(self, duration: float) -> None:
        """Log details about a slow job."""
        if not self._current_job:
            return

        job = self._current_job
        message_parts = [
            f"SLOW JOB DETECTED: {duration:.2f}s (threshold: {self.threshold_seconds}s)"
        ]

        if job["file_path"]:
            message_parts.append(f"  File: {job['file_path']}")
        if job["file_size_mb"]:
            message_parts.append(f"  Size: {job['file_size_mb']:.2f} MB")
        if job["tick_count"]:
            message_parts.append(f"  Ticks: {job['tick_count']:,}")
        if job["metrics_enabled"]:
            message_parts.append(f"  Metrics: {', '.join(job['metrics_enabled'])}")

        # Calculate throughput if we have size
        if job["file_size_bytes"] and duration > 0:
            mb_per_sec = job["file_size_mb"] / duration
            message_parts.append(f"  Throughput: {mb_per_sec:.2f} MB/s")

        logger.log(self.log_level, "\n".join(message_parts))


def create_timing_context(
    enable_timing: bool = True,
    enable_slow_logging: bool = True,
    slow_threshold_seconds: float = DEFAULT_SLOW_THRESHOLD_SECONDS,
) -> tuple[TimingCollector, SlowJobLogger]:
    """Create timing and slow logging context for an analysis.

    Args:
        enable_timing: Whether to enable timing collection
        enable_slow_logging: Whether to enable slow job logging
        slow_threshold_seconds: Threshold for slow job logging

    Returns:
        Tuple of (TimingCollector, SlowJobLogger)
    """
    collector = TimingCollector()
    collector.enabled = enable_timing

    slow_logger = SlowJobLogger(
        threshold_seconds=slow_threshold_seconds,
        log_level=logging.WARNING if enable_slow_logging else logging.DEBUG,
    )

    return collector, slow_logger


# Convenience exports
__all__ = [
    "StageTime",
    "TimingStats",
    "TimingCollector",
    "stage_timer",
    "timed",
    "get_timing_collector",
    "set_timing_collector",
    "Profiler",
    "SlowJobLogger",
    "create_timing_context",
    "DEFAULT_SLOW_THRESHOLD_SECONDS",
]
