"""
Utility functions and performance helpers for OpenSight CS2 Analyzer.

This module provides:
- Performance timing decorators
- Memory monitoring utilities
- Data validation helpers
- Common utility functions
"""

from functools import wraps
from typing import Any, Callable, Optional, TypeVar
import logging
import time
import gc

logger = logging.getLogger(__name__)

# Type variable for generic decorators
F = TypeVar('F', bound=Callable[..., Any])


def timed(func: F) -> F:
    """
    Decorator to measure and log function execution time.

    Usage:
        @timed
        def my_function():
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        logger.info(f"{func.__name__} completed in {elapsed:.3f}s")
        return result
    return wrapper  # type: ignore


def memory_efficient(clear_gc: bool = True):
    """
    Decorator for memory-intensive functions.
    Runs garbage collection after execution.

    Args:
        clear_gc: Whether to run gc.collect() after function

    Usage:
        @memory_efficient()
        def process_large_data():
            ...
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if clear_gc:
                gc.collect()
            return result
        return wrapper  # type: ignore
    return decorator


class PerformanceMonitor:
    """
    Context manager for monitoring performance of code blocks.

    Usage:
        with PerformanceMonitor("parsing demo"):
            parse_demo(...)
    """

    def __init__(self, operation_name: str, log_level: int = logging.INFO):
        self.operation_name = operation_name
        self.log_level = log_level
        self.start_time: Optional[float] = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - (self.start_time or 0)
        if exc_type is not None:
            logger.error(f"{self.operation_name} failed after {elapsed:.3f}s: {exc_val}")
        else:
            logger.log(self.log_level, f"{self.operation_name} completed in {elapsed:.3f}s")
        return False


def validate_steamid(steam_id: Any) -> bool:
    """
    Validate that a steam ID is valid.

    Args:
        steam_id: Value to validate

    Returns:
        True if valid steam ID, False otherwise
    """
    if steam_id is None:
        return False
    try:
        sid = int(steam_id)
        # Steam IDs are 64-bit, but in practice should be > 0
        return sid > 0
    except (ValueError, TypeError):
        return False


def validate_round_number(round_num: Any, max_rounds: int = 60) -> bool:
    """
    Validate that a round number is within expected range.

    Args:
        round_num: Value to validate
        max_rounds: Maximum expected rounds (default 60 for OT)

    Returns:
        True if valid round number, False otherwise
    """
    if round_num is None:
        return False
    try:
        rnum = int(round_num)
        return 0 <= rnum <= max_rounds
    except (ValueError, TypeError):
        return False


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is 0.

    Args:
        numerator: Top of fraction
        denominator: Bottom of fraction
        default: Value to return if denominator is 0

    Returns:
        Result of division or default
    """
    if denominator == 0:
        return default
    return numerator / denominator


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp a value to a range.

    Args:
        value: Value to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Clamped value
    """
    return max(min_val, min(value, max_val))


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds to a human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string like "2m 30s" or "1.5s"
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format a decimal as a percentage string.

    Args:
        value: Value between 0-1 (or already a percentage)
        decimals: Number of decimal places

    Returns:
        Formatted percentage string
    """
    if value < 1.0:
        value *= 100
    return f"{value:.{decimals}f}%"
