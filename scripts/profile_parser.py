#!/usr/bin/env python3
"""
Parser Performance Profiler

Identifies the slowest functions in parser.py to guide optimization efforts.
Uses cProfile for detailed timing analysis.

Usage:
    python scripts/profile_parser.py path/to/demo.dem
    python scripts/profile_parser.py path/to/demo.dem --top 10
    python scripts/profile_parser.py path/to/demo.dem --output profile_results.txt
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
from io import StringIO
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def profile_parser(demo_path: str, top_n: int = 20, output_file: str | None = None) -> dict:
    """
    Profile the demo parser and return timing statistics.

    Args:
        demo_path: Path to the demo file to parse
        top_n: Number of top functions to report
        output_file: Optional file to write detailed results

    Returns:
        Dict with profiling summary
    """
    from opensight.core.parser import DemoParser

    demo_path = Path(demo_path)
    if not demo_path.exists():
        raise FileNotFoundError(f"Demo file not found: {demo_path}")

    print(f"\n{'=' * 60}")
    print(f"Profiling Parser: {demo_path.name}")
    print(f"File size: {demo_path.stat().st_size / (1024 * 1024):.1f} MB")
    print(f"{'=' * 60}\n")

    # Profile the parsing
    profiler = cProfile.Profile()

    profiler.enable()
    parser = DemoParser(demo_path)
    data = parser.parse(include_ticks=False, comprehensive=True)
    profiler.disable()

    # Collect stats
    stats_stream = StringIO()
    stats = pstats.Stats(profiler, stream=stats_stream)
    stats.strip_dirs()
    stats.sort_stats("cumulative")

    # Print top functions by cumulative time
    print(f"\nTop {top_n} Functions by Cumulative Time:")
    print("-" * 80)
    stats.print_stats(top_n)

    # Also sort by total time (time spent IN the function, not including subcalls)
    print(f"\n\nTop {top_n} Functions by Total Time (self time):")
    print("-" * 80)
    stats.sort_stats("tottime")
    stats.print_stats(top_n)

    # Focus on parser.py functions
    print("\n\nParser.py Functions Only (sorted by cumtime):")
    print("-" * 80)
    stats.sort_stats("cumulative")
    stats.print_stats("parser.py", top_n)

    # Identify iterrows() hotspots
    print("\n\nIterrows Hotspots (DataFrame iteration):")
    print("-" * 80)
    stats.print_stats("iterrows", top_n)

    # Collect summary data
    summary = {
        "demo_file": str(demo_path),
        "file_size_mb": demo_path.stat().st_size / (1024 * 1024),
        "num_rounds": data.num_rounds,
        "num_kills": len(data.kills),
        "num_damages": len(data.damages),
        "num_players": len(data.player_stats),
    }

    # Get total parse time
    total_stats = stats.get_stats_profile()
    if hasattr(total_stats, "total_tt"):
        summary["total_time_seconds"] = total_stats.total_tt
    else:
        # Fallback: sum all cumulative times at top level
        summary["total_time_seconds"] = (
            sum(
                stat[3]
                for stat in stats.stats.values()  # cumtime
            )
            / len(stats.stats)
            if stats.stats
            else 0
        )

    # Extract top bottlenecks
    stats.sort_stats("cumulative")
    bottlenecks = []
    for (filename, lineno, funcname), (_cc, nc, tt, ct, _callers) in list(stats.stats.items())[:10]:
        if "parser.py" in filename or "analytics.py" in filename or "metrics.py" in filename:
            bottlenecks.append(
                {
                    "function": funcname,
                    "file": Path(filename).name,
                    "line": lineno,
                    "calls": nc,
                    "total_time": tt,
                    "cumulative_time": ct,
                }
            )

    summary["bottlenecks"] = bottlenecks

    # Write detailed output if requested
    if output_file:
        with open(output_file, "w") as f:
            f.write(stats_stream.getvalue())
            f.write("\n\n" + "=" * 60 + "\n")
            f.write("SUMMARY\n")
            f.write("=" * 60 + "\n")
            for key, value in summary.items():
                if key != "bottlenecks":
                    f.write(f"{key}: {value}\n")
            f.write("\nTOP BOTTLENECKS:\n")
            for b in summary["bottlenecks"]:
                f.write(
                    f"  {b['function']} ({b['file']}:{b['line']}): "
                    f"{b['cumulative_time']:.3f}s cumulative, {b['calls']} calls\n"
                )
        print(f"\nDetailed results written to: {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("PROFILING SUMMARY")
    print("=" * 60)
    print(f"Demo: {summary['demo_file']}")
    print(f"Size: {summary['file_size_mb']:.1f} MB")
    print(f"Rounds: {summary['num_rounds']}")
    print(f"Kills: {summary['num_kills']}")
    print(f"Damages: {summary['num_damages']}")
    print(f"Players: {summary['num_players']}")
    print()
    print("TOP BOTTLENECKS TO OPTIMIZE:")
    for i, b in enumerate(summary["bottlenecks"][:5], 1):
        print(f"  {i}. {b['function']} ({b['file']}:{b['line']})")
        print(f"     Cumulative: {b['cumulative_time']:.3f}s, Calls: {b['calls']}")

    return summary


def benchmark_function(demo_path: str, func_name: str, iterations: int = 5) -> dict:
    """
    Benchmark a specific function for before/after comparison.

    Args:
        demo_path: Path to demo file
        func_name: Name of function to benchmark (e.g., "_build_kills")
        iterations: Number of iterations for averaging

    Returns:
        Dict with benchmark results
    """
    import time

    from opensight.core.parser import DemoParser

    demo_path = Path(demo_path)
    parser = DemoParser(demo_path)

    # First pass to warm up and get the data
    parser.parse(include_ticks=False, comprehensive=True)

    times = []
    for _ in range(iterations):
        # Re-parse for each iteration (reset parser state)
        parser._data = None
        start = time.perf_counter()
        parser.parse(include_ticks=False, comprehensive=True)
        end = time.perf_counter()
        times.append(end - start)

    return {
        "function": func_name,
        "iterations": iterations,
        "times": times,
        "mean_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
    }


def compare_implementations(
    demo_path: str,
    old_func: callable,
    new_func: callable,
    test_data,
    iterations: int = 10,
) -> dict:
    """
    Compare two implementations of a function.

    Args:
        demo_path: Path to demo (for context)
        old_func: Original function
        new_func: Optimized function
        test_data: Data to pass to both functions
        iterations: Number of iterations

    Returns:
        Dict with comparison results
    """
    import time

    old_times = []
    new_times = []

    for _ in range(iterations):
        # Time old implementation
        start = time.perf_counter()
        old_func(test_data)
        old_times.append(time.perf_counter() - start)

        # Time new implementation
        start = time.perf_counter()
        new_func(test_data)
        new_times.append(time.perf_counter() - start)

    old_mean = sum(old_times) / len(old_times)
    new_mean = sum(new_times) / len(new_times)
    speedup = old_mean / new_mean if new_mean > 0 else float("inf")

    return {
        "old_mean": old_mean,
        "new_mean": new_mean,
        "speedup": speedup,
        "old_times": old_times,
        "new_times": new_times,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile the CS2 demo parser to identify performance bottlenecks"
    )
    parser.add_argument("demo", help="Path to demo file (.dem)")
    parser.add_argument(
        "--top", "-t", type=int, default=20, help="Number of top functions to show (default: 20)"
    )
    parser.add_argument("--output", "-o", help="Output file for detailed results")
    parser.add_argument(
        "--benchmark", "-b", action="store_true", help="Run benchmark mode (multiple iterations)"
    )
    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=5,
        help="Number of iterations for benchmark mode (default: 5)",
    )

    args = parser.parse_args()

    if args.benchmark:
        print("Running benchmark mode...")
        results = benchmark_function(args.demo, "parse", args.iterations)
        print(f"\nBenchmark Results ({results['iterations']} iterations):")
        print(f"  Mean: {results['mean_time']:.3f}s")
        print(f"  Min:  {results['min_time']:.3f}s")
        print(f"  Max:  {results['max_time']:.3f}s")
    else:
        profile_parser(args.demo, args.top, args.output)
