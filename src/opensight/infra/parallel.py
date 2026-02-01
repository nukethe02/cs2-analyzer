"""
Parallel Processing Module for Batch Demo Analysis

Implements:
- Multiprocessing for analyzing multiple demos simultaneously
- Process pool management with configurable workers
- Progress tracking and result aggregation
- Memory-efficient chunked processing
- Background worker support for async analysis
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default to CPU count - 1, minimum 1
DEFAULT_WORKERS = max(1, (os.cpu_count() or 4) - 1)
MAX_WORKERS = os.cpu_count() or 8


@dataclass
class DemoAnalysisTask:
    """A single demo analysis task."""

    demo_path: Path
    task_id: str = ""
    priority: int = 0  # Higher = more priority
    options: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.task_id:
            self.task_id = hashlib.md5(
                str(self.demo_path).encode(), usedforsecurity=False
            ).hexdigest()[:12]


@dataclass
class DemoAnalysisResult:
    """Result of a single demo analysis."""

    task_id: str
    demo_path: str
    success: bool
    duration_seconds: float
    error_message: str | None = None
    analysis_data: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BatchAnalysisProgress:
    """Progress tracking for batch analysis."""

    total_tasks: int
    completed_tasks: int = 0
    failed_tasks: int = 0
    current_task: str = ""
    start_time: float = field(default_factory=time.time)

    @property
    def progress_percent(self) -> float:
        if self.total_tasks == 0:
            return 100.0
        return round((self.completed_tasks / self.total_tasks) * 100, 1)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def estimated_remaining_seconds(self) -> float:
        if self.completed_tasks == 0:
            return 0.0
        avg_time_per_task = self.elapsed_seconds / self.completed_tasks
        remaining_tasks = self.total_tasks - self.completed_tasks
        return avg_time_per_task * remaining_tasks


@dataclass
class BatchAnalysisResult:
    """Result of batch analysis."""

    total_demos: int
    successful: int
    failed: int
    total_duration_seconds: float
    results: list[DemoAnalysisResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_demos == 0:
            return 0.0
        return round((self.successful / self.total_demos) * 100, 1)

    def to_dict(self) -> dict:
        return {
            "total_demos": self.total_demos,
            "successful": self.successful,
            "failed": self.failed,
            "success_rate": self.success_rate,
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "results": [r.to_dict() for r in self.results],
        }


def _analyze_single_demo(task: DemoAnalysisTask) -> DemoAnalysisResult:
    """
    Worker function to analyze a single demo.
    This runs in a separate process.
    """
    start_time = time.time()

    try:
        # Import here to avoid pickle issues with multiprocessing
        from opensight.analysis.analytics import DemoAnalyzer
        from opensight.core.parser import DemoParser

        # Parse the demo
        parser = DemoParser(task.demo_path)

        # Check for tick-level data option
        include_ticks = task.options.get("include_ticks", False)
        demo_data = parser.parse(include_ticks=include_ticks)

        # Run analysis
        analyzer = DemoAnalyzer(demo_data)
        analysis = analyzer.analyze()

        # Convert to serializable dict
        analysis_dict = {
            "map_name": analysis.map_name,
            "total_rounds": analysis.total_rounds,
            "team1_score": analysis.team1_score,
            "team2_score": analysis.team2_score,
            "player_count": len(analysis.players),
            "players": {
                str(sid): {
                    "name": p.name,
                    "team": p.team,
                    "kills": p.kills,
                    "deaths": p.deaths,
                    "assists": p.assists,
                    "adr": p.adr,
                    "hltv_rating": p.hltv_rating,
                    "kast_percentage": p.kast_percentage,
                }
                for sid, p in analysis.players.items()
            },
        }

        duration = time.time() - start_time
        return DemoAnalysisResult(
            task_id=task.task_id,
            demo_path=str(task.demo_path),
            success=True,
            duration_seconds=duration,
            analysis_data=analysis_dict,
        )

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Failed to analyze {task.demo_path}: {e}")
        return DemoAnalysisResult(
            task_id=task.task_id,
            demo_path=str(task.demo_path),
            success=False,
            duration_seconds=duration,
            error_message=str(e),
        )


class ParallelDemoAnalyzer:
    """
    Parallel demo analyzer using multiprocessing.

    Usage:
        analyzer = ParallelDemoAnalyzer(workers=4)
        results = analyzer.analyze_batch([Path("demo1.dem"), Path("demo2.dem")])
    """

    def __init__(
        self,
        workers: int = DEFAULT_WORKERS,
        use_processes: bool = True,
        progress_callback: Callable[[BatchAnalysisProgress], None] | None = None,
    ):
        """
        Initialize the parallel analyzer.

        Args:
            workers: Number of worker processes/threads
            use_processes: If True, use ProcessPoolExecutor; if False, use ThreadPoolExecutor
            progress_callback: Optional callback for progress updates
        """
        self.workers = min(workers, MAX_WORKERS)
        self.use_processes = use_processes
        self.progress_callback = progress_callback

        logger.info(f"ParallelDemoAnalyzer initialized with {self.workers} workers")

    def analyze_batch(
        self,
        demo_paths: list[Path],
        include_ticks: bool = False,
        timeout_per_demo: int = 300,  # 5 minutes per demo
    ) -> BatchAnalysisResult:
        """
        Analyze multiple demos in parallel.

        Args:
            demo_paths: List of paths to demo files
            include_ticks: Whether to include tick-level data (slower but needed for replay)
            timeout_per_demo: Timeout in seconds per demo

        Returns:
            BatchAnalysisResult with all results
        """
        if not demo_paths:
            return BatchAnalysisResult(
                total_demos=0,
                successful=0,
                failed=0,
                total_duration_seconds=0.0,
            )

        # Create tasks
        tasks = [
            DemoAnalysisTask(
                demo_path=path,
                options={"include_ticks": include_ticks},
            )
            for path in demo_paths
        ]

        # Initialize progress
        progress = BatchAnalysisProgress(total_tasks=len(tasks))
        start_time = time.time()

        results: list[DemoAnalysisResult] = []

        # Choose executor type
        ExecutorClass = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor

        logger.info(f"Starting batch analysis of {len(tasks)} demos with {self.workers} workers")

        with ExecutorClass(max_workers=self.workers) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(_analyze_single_demo, task): task for task in tasks}

            # Collect results as they complete
            for future in as_completed(future_to_task, timeout=timeout_per_demo * len(tasks)):
                task = future_to_task[future]
                progress.current_task = str(task.demo_path)

                try:
                    result = future.result(timeout=timeout_per_demo)
                    results.append(result)

                    if result.success:
                        progress.completed_tasks += 1
                    else:
                        progress.failed_tasks += 1
                        progress.completed_tasks += 1

                except Exception as e:
                    logger.error(f"Task {task.task_id} failed: {e}")
                    results.append(
                        DemoAnalysisResult(
                            task_id=task.task_id,
                            demo_path=str(task.demo_path),
                            success=False,
                            duration_seconds=0.0,
                            error_message=str(e),
                        )
                    )
                    progress.failed_tasks += 1
                    progress.completed_tasks += 1

                # Call progress callback
                if self.progress_callback:
                    self.progress_callback(progress)

        total_duration = time.time() - start_time
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        logger.info(
            f"Batch analysis complete: {successful}/{len(results)} successful in {total_duration:.1f}s"
        )

        return BatchAnalysisResult(
            total_demos=len(results),
            successful=successful,
            failed=failed,
            total_duration_seconds=total_duration,
            results=results,
        )

    def analyze_directory(
        self,
        directory: Path,
        recursive: bool = True,
        include_ticks: bool = False,
    ) -> BatchAnalysisResult:
        """
        Analyze all demo files in a directory.

        Args:
            directory: Directory to scan for demos
            recursive: Whether to scan subdirectories
            include_ticks: Whether to include tick-level data

        Returns:
            BatchAnalysisResult with all results
        """
        pattern = "**/*.dem" if recursive else "*.dem"
        demo_paths = list(directory.glob(pattern))

        # Also include .dem.gz files
        gz_pattern = "**/*.dem.gz" if recursive else "*.dem.gz"
        demo_paths.extend(directory.glob(gz_pattern))

        logger.info(f"Found {len(demo_paths)} demo files in {directory}")

        return self.analyze_batch(demo_paths, include_ticks=include_ticks)


class BackgroundAnalyzer:
    """
    Background worker for async demo analysis.

    Runs analysis in background threads/processes and stores results.
    """

    def __init__(self, result_dir: Path | None = None):
        """
        Initialize the background analyzer.

        Args:
            result_dir: Directory to store results (default: ~/.opensight/results)
        """
        self.result_dir = result_dir or Path.home() / ".opensight" / "results"
        self.result_dir.mkdir(parents=True, exist_ok=True)

        self._executor: ThreadPoolExecutor | None = None
        self._pending_tasks: dict[str, Any] = {}

    def start(self, workers: int = 2):
        """Start the background worker pool."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=workers)
            logger.info(f"Background analyzer started with {workers} workers")

    def stop(self):
        """Stop the background worker pool."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
            logger.info("Background analyzer stopped")

    def submit(self, demo_path: Path, include_ticks: bool = False) -> str:
        """
        Submit a demo for background analysis.

        Args:
            demo_path: Path to demo file
            include_ticks: Whether to include tick-level data

        Returns:
            Task ID for tracking
        """
        if self._executor is None:
            self.start()

        task = DemoAnalysisTask(
            demo_path=demo_path,
            options={"include_ticks": include_ticks},
        )

        future = self._executor.submit(_analyze_single_demo, task)
        self._pending_tasks[task.task_id] = future

        # Add callback to save result
        future.add_done_callback(lambda f: self._save_result(task.task_id, f))

        logger.info(f"Submitted background task {task.task_id} for {demo_path}")
        return task.task_id

    def _save_result(self, task_id: str, future):
        """Save result to disk when complete."""
        try:
            result = future.result()
            result_path = self.result_dir / f"{task_id}.json"

            with open(result_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)

            logger.info(f"Saved result for task {task_id}")

        except Exception as e:
            logger.error(f"Failed to save result for {task_id}: {e}")

        finally:
            # Remove from pending
            self._pending_tasks.pop(task_id, None)

    def get_status(self, task_id: str) -> dict:
        """
        Get status of a background task.

        Returns:
            Dict with status, result if complete
        """
        # Check if still pending
        if task_id in self._pending_tasks:
            future = self._pending_tasks[task_id]
            if future.done():
                try:
                    result = future.result()
                    return {"status": "complete", "result": result.to_dict()}
                except Exception as e:
                    return {"status": "failed", "error": str(e)}
            return {"status": "pending"}

        # Check for saved result
        result_path = self.result_dir / f"{task_id}.json"
        if result_path.exists():
            with open(result_path) as f:
                return {"status": "complete", "result": json.load(f)}

        return {"status": "not_found"}

    def list_results(self) -> list[dict]:
        """List all saved results."""
        results = []
        for path in self.result_dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                    results.append(
                        {
                            "task_id": path.stem,
                            "demo_path": data.get("demo_path"),
                            "success": data.get("success"),
                        }
                    )
            except Exception:
                continue
        return results


# Convenience functions


def analyze_demos_parallel(
    demo_paths: list[Path],
    workers: int = DEFAULT_WORKERS,
    include_ticks: bool = False,
) -> BatchAnalysisResult:
    """
    Convenience function to analyze multiple demos in parallel.

    Args:
        demo_paths: List of demo file paths
        workers: Number of parallel workers
        include_ticks: Include tick-level data for replay

    Returns:
        BatchAnalysisResult
    """
    analyzer = ParallelDemoAnalyzer(workers=workers)
    return analyzer.analyze_batch(demo_paths, include_ticks=include_ticks)


def analyze_directory_parallel(
    directory: Path,
    workers: int = DEFAULT_WORKERS,
    recursive: bool = True,
) -> BatchAnalysisResult:
    """
    Convenience function to analyze all demos in a directory.

    Args:
        directory: Directory containing demo files
        workers: Number of parallel workers
        recursive: Scan subdirectories

    Returns:
        BatchAnalysisResult
    """
    analyzer = ParallelDemoAnalyzer(workers=workers)
    return analyzer.analyze_directory(directory, recursive=recursive)


# Alias for backward compatibility
analyze_batch = analyze_demos_parallel
