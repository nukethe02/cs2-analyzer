"""
OpenSight CLI - Command Line Interface for CS2 Analytics

Provides commands for:
- Analyzing demo files
- Decoding share codes
- Watching for new replays
- Generating reports

IMPORTANT: This module uses lazy imports to ensure that lightweight commands
like 'info' and 'decode' don't load heavy dependencies (demoparser2, pandas, numpy).
"""

import json
import logging
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

from opensight import __version__

# NOTE: Heavy dependencies (demoparser2, pandas, numpy) are loaded lazily via
# _import_parser(), _import_metrics(), _import_watcher() to keep 'info' and
# 'decode' commands fast and lightweight.


def _import_parser():
    """Lazy import for DemoParser (loads demoparser2, pandas, numpy)."""
    from opensight.core.parser import DemoData, DemoParser

    return DemoParser, DemoData


def _import_metrics():
    """Lazy import for metrics functions (loads analytics)."""
    from opensight.analysis.metrics import (
        calculate_comprehensive_metrics,
        calculate_crosshair_placement,
        calculate_economy_metrics,
        calculate_engagement_metrics,
        calculate_opening_metrics,
        calculate_trade_metrics,
        calculate_ttd,
        calculate_utility_metrics,
    )

    return {
        "calculate_ttd": calculate_ttd,
        "calculate_crosshair_placement": calculate_crosshair_placement,
        "calculate_engagement_metrics": calculate_engagement_metrics,
        "calculate_economy_metrics": calculate_economy_metrics,
        "calculate_utility_metrics": calculate_utility_metrics,
        "calculate_trade_metrics": calculate_trade_metrics,
        "calculate_opening_metrics": calculate_opening_metrics,
        "calculate_comprehensive_metrics": calculate_comprehensive_metrics,
    }


def _import_watcher():
    """Lazy import for watcher (loads watchdog)."""
    from opensight.infra.watcher import DemoFileEvent, ReplayWatcher, get_default_replays_folder

    return ReplayWatcher, DemoFileEvent, get_default_replays_folder


def _import_export():
    """Lazy import for export functions."""
    from opensight.visualization.export import export_analysis

    return export_analysis


app = typer.Typer(
    name="opensight",
    help="Local CS2 analytics framework - professional-grade metrics without cloud dependencies",
    add_completion=False,
)
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        console.print(f"[bold blue]OpenSight[/bold blue] v{__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-V", help="Enable verbose output"),
) -> None:
    """OpenSight - Local CS2 Analytics Framework"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@app.command()
def analyze(
    demo_path: Path = typer.Argument(
        ...,
        help="Path to the .dem file to analyze",
        exists=True,
        dir_okay=False,
        resolve_path=True,
    ),
    player: str | None = typer.Option(
        None, "--player", "-p", help="Filter results to a specific player (name or Steam ID)"
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for results (format detected from extension: .json, .csv, .xlsx, .html)",
    ),
    metrics: str = typer.Option(
        "all",
        "--metrics",
        "-m",
        help="Metrics to calculate: all, ttd, cp, engagement, economy, utility, trades, opening",
    ),
    profile: bool = typer.Option(
        False, "--profile", hidden=True, help="Show timing per stage for performance profiling"
    ),
) -> None:
    """
    Analyze a CS2 demo file and display metrics.

    Calculates professional-grade analytics including:
    - Time to Damage (TTD)
    - Crosshair Placement (CP)
    - Kill/Death statistics
    - Damage per round

    Performance options:
    - --parse-mode minimal: Parse only kills/damages for TTD/CP (faster)
    - --cp-sample-rate 4: Sample every 4th tick (32 Hz instead of 128 Hz)
    - --no-optimize: Disable dtype optimization (more memory)
    """
    console.print("\n[bold blue]OpenSight[/bold blue] - Analyzing demo...\n")

    timings: dict[str, float] = {}

    # Lazy import heavy dependencies
    DemoParser, DemoData = _import_parser()
    metrics_funcs = _import_metrics()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=not profile,
    ) as progress:
        # Parse the demo
        task = progress.add_task("Parsing demo file...", total=100)
        try:
            start_time = time.perf_counter()
            progress.update(task, completed=10, description="Loading parser...")

            parser = DemoParser(demo_path)
            progress.update(task, completed=30, description="Parsing events...")

            data = parser.parse()
            progress.update(task, completed=100, description="Demo parsed!")

            timings["parsing"] = time.perf_counter() - start_time
        except Exception as e:
            console.print(f"[red]Error parsing demo:[/red] {e}")
            raise typer.Exit(1)

    # Display basic info
    info_table = Table(title="Demo Information", show_header=False)
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="green")
    info_table.add_row("Map", data.map_name)
    info_table.add_row("Duration", f"{data.duration_seconds:.1f} seconds")
    info_table.add_row("Tick Rate", str(data.tick_rate))
    info_table.add_row("Players", str(len(data.player_names)))
    console.print(info_table)
    console.print()

    # Filter by player if specified
    steam_id: int | None = None
    if player:
        # Try to find player by name or Steam ID
        try:
            steam_id = int(player)
        except ValueError:
            # Search by name
            for sid, name in data.player_names.items():
                if player.lower() in name.lower():
                    steam_id = sid
                    break
            if steam_id is None:
                console.print(f"[yellow]Warning:[/yellow] Player '{player}' not found")

    # Calculate and display metrics with profiling
    start_time = time.perf_counter()

    if metrics in ("all", "engagement"):
        _display_engagement_metrics(data, steam_id, metrics_funcs)
        if profile:
            timings["engagement"] = time.perf_counter() - start_time
            start_time = time.perf_counter()

    if metrics in ("all", "ttd"):
        _display_ttd_metrics(data, steam_id, metrics_funcs)
        if profile:
            timings["ttd"] = time.perf_counter() - start_time
            start_time = time.perf_counter()

    if metrics in ("all", "cp"):
        _display_cp_metrics(data, steam_id, metrics_funcs)
        if profile:
            timings["cp"] = time.perf_counter() - start_time
            start_time = time.perf_counter()

    if metrics in ("all", "economy"):
        _display_economy_metrics(data, steam_id, metrics_funcs)
        if profile:
            timings["economy"] = time.perf_counter() - start_time
            start_time = time.perf_counter()

    if metrics in ("all", "utility"):
        _display_utility_metrics(data, steam_id, metrics_funcs)
        if profile:
            timings["utility"] = time.perf_counter() - start_time
            start_time = time.perf_counter()

    if metrics in ("all", "trades"):
        _display_trade_metrics(data, steam_id, metrics_funcs)
        if profile:
            timings["trades"] = time.perf_counter() - start_time
            start_time = time.perf_counter()

    if metrics in ("all", "opening"):
        _display_opening_metrics(data, steam_id, metrics_funcs)
        if profile:
            timings["opening"] = time.perf_counter() - start_time
            start_time = time.perf_counter()

    # Export if requested
    if output:
        export_analysis = _import_export()
        comprehensive = metrics_funcs["calculate_comprehensive_metrics"](data, steam_id)
        export_analysis(data, comprehensive, output)
        console.print(f"\n[green]Results exported to:[/green] {output}")
        if profile:
            timings["export"] = time.perf_counter() - start_time

    # Display profiling results
    if profile:
        console.print()
        profile_table = Table(title="Performance Profile", show_header=True)
        profile_table.add_column("Stage", style="cyan")
        profile_table.add_column("Time", justify="right", style="yellow")
        profile_table.add_column("Percentage", justify="right")

        total_time = sum(timings.values())
        for stage, duration in timings.items():
            pct = (duration / total_time * 100) if total_time > 0 else 0
            profile_table.add_row(stage.capitalize(), f"{duration:.3f}s", f"{pct:.1f}%")
        profile_table.add_row(
            "[bold]Total[/bold]", f"[bold]{total_time:.3f}s[/bold]", "[bold]100%[/bold]"
        )
        console.print(profile_table)


def _display_engagement_metrics(data, steam_id: int | None, metrics_funcs: dict) -> None:
    """Display engagement metrics table."""
    calculate_engagement_metrics = metrics_funcs["calculate_engagement_metrics"]
    metrics_result = calculate_engagement_metrics(data, steam_id)

    if not metrics_result:
        console.print("[yellow]No engagement metrics available[/yellow]")
        return

    table = Table(title="Engagement Metrics")
    table.add_column("Player", style="cyan")
    table.add_column("K", justify="right")
    table.add_column("D", justify="right")
    table.add_column("K/D", justify="right")
    table.add_column("HS%", justify="right")
    table.add_column("DPR", justify="right")

    for _sid, m in sorted(metrics_result.items(), key=lambda x: x[1].total_kills, reverse=True):
        kd = m.total_kills / m.total_deaths if m.total_deaths > 0 else m.total_kills
        table.add_row(
            m.player_name,
            str(m.total_kills),
            str(m.total_deaths),
            f"{kd:.2f}",
            f"{m.headshot_percentage:.1f}%",
            f"{m.damage_per_round:.1f}",
        )

    console.print(table)
    console.print()


def _display_ttd_metrics(data, steam_id: int | None, metrics_funcs: dict) -> None:
    """Display Time to Damage metrics table."""
    calculate_ttd = metrics_funcs["calculate_ttd"]
    ttd_results = calculate_ttd(data, steam_id)

    if not ttd_results:
        console.print("[yellow]No TTD metrics available[/yellow]")
        return

    table = Table(title="Time to Damage (TTD)")
    table.add_column("Player", style="cyan")
    table.add_column("Engagements", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")

    for _sid, ttd in sorted(ttd_results.items(), key=lambda x: x[1].mean_ttd_ms):
        table.add_row(
            ttd.player_name,
            str(ttd.engagement_count),
            f"{ttd.mean_ttd_ms:.0f}ms",
            f"{ttd.median_ttd_ms:.0f}ms",
            f"{ttd.min_ttd_ms:.0f}ms",
            f"{ttd.max_ttd_ms:.0f}ms",
        )

    console.print(table)
    console.print()


def _display_cp_metrics(data, steam_id: int | None, metrics_funcs: dict) -> None:
    """Display Crosshair Placement metrics table."""
    calculate_crosshair_placement = metrics_funcs["calculate_crosshair_placement"]
    cp_results = calculate_crosshair_placement(data, steam_id)

    if not cp_results:
        console.print("[yellow]No crosshair placement metrics available[/yellow]")
        return

    table = Table(title="Crosshair Placement")
    table.add_column("Player", style="cyan")
    table.add_column("Samples", justify="right")
    table.add_column("Mean Angle", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("90th %ile", justify="right")
    table.add_column("Score", justify="right")

    for _sid, cp in sorted(cp_results.items(), key=lambda x: x[1].placement_score, reverse=True):
        table.add_row(
            cp.player_name,
            str(cp.sample_count),
            f"{cp.mean_angle_deg:.1f}°",
            f"{cp.median_angle_deg:.1f}°",
            f"{cp.percentile_90_deg:.1f}°",
            f"{cp.placement_score:.1f}",
        )

    console.print(table)
    console.print()


def _display_economy_metrics(data, steam_id: int | None, metrics_funcs: dict) -> None:
    """Display economy metrics table."""
    calculate_economy_metrics = metrics_funcs["calculate_economy_metrics"]
    econ_results = calculate_economy_metrics(data, steam_id)

    if not econ_results:
        console.print("[yellow]No economy metrics available[/yellow]")
        return

    table = Table(title="Economy Metrics")
    table.add_column("Player", style="cyan")
    table.add_column("Money Spent", justify="right")
    table.add_column("Efficiency", justify="right")
    table.add_column("Eco Kills", justify="right")
    table.add_column("Force Kills", justify="right")
    table.add_column("Full Buy Kills", justify="right")
    table.add_column("Favorite Weapon", justify="right")

    for sid, econ in sorted(econ_results.items(), key=lambda x: x[1].weapon_efficiency, reverse=True):
        table.add_row(
            econ.player_name,
            f"${econ.total_money_spent:,}",
            f"{econ.weapon_efficiency:.2f}",
            str(econ.eco_round_kills),
            str(econ.force_buy_kills),
            str(econ.full_buy_kills),
            econ.favorite_weapon,
        )

    console.print(table)
    console.print()


@app.command()
def decode(
    share_code: str = typer.Argument(
        ..., help="CS2 share code (e.g., CSGO-xxxxx-xxxxx-xxxxx-xxxxx-xxxxx)"
    ),
) -> None:
    """
    Decode a CS2 share code to extract match metadata.

    The share code contains encoded information about the match
    including match ID, outcome ID, and token.

    This command uses zero heavy dependencies - just pure math.
    """
    # Import sharecode module directly - it has no heavy dependencies
    from opensight.integrations.sharecode import decode_sharecode

    try:
        info = decode_sharecode(share_code)

        panel = Panel(
            f"[cyan]Match ID:[/cyan] {info.match_id}\n"
            f"[cyan]Outcome ID:[/cyan] {info.outcome_id}\n"
            f"[cyan]Token:[/cyan] {info.token}",
            title="[bold blue]Share Code Decoded[/bold blue]",
            expand=False,
        )
        console.print(panel)

    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def watch(
    folder: Path | None = typer.Option(
        None,
        "--folder",
        "-f",
        help="Folder to watch (defaults to CS2 replays folder)",
        exists=False,
        dir_okay=True,
        file_okay=False,
    ),
    auto_analyze: bool = typer.Option(
        True, "--analyze/--no-analyze", "-a", help="Automatically analyze new demos"
    ),
    use_cache: bool = typer.Option(
        True, "--cache/--no-cache", help="Use cache to skip already-analyzed demos"
    ),
    debounce: float = typer.Option(
        2.0, "--debounce", "-d", help="Debounce time in seconds (wait for file to stabilize)"
    ),
) -> None:
    """
    Watch for new CS2 replay files and optionally analyze them.

    Monitors the specified folder (or default CS2 replays folder)
    for new .dem files and triggers analysis automatically.

    Features:
    - Debounces file events to wait for write completion
    - Caches analyzed demos to skip re-processing
    - Coalesces multiple file events into single analysis
    """
    # Lazy import watcher dependencies
    ReplayWatcher, DemoFileEvent, get_default_replays_folder = _import_watcher()

    if folder is None:
        folder = get_default_replays_folder()

    console.print("\n[bold blue]OpenSight[/bold blue] - Watching for replays\n")
    console.print(f"[cyan]Folder:[/cyan] {folder}")
    console.print(f"[cyan]Auto-analyze:[/cyan] {'Yes' if auto_analyze else 'No'}")
    console.print(f"[cyan]Cache:[/cyan] {'Enabled' if use_cache else 'Disabled'}")
    console.print(f"[cyan]Debounce:[/cyan] {debounce}s")
    console.print("\nPress [bold]Ctrl+C[/bold] to stop...\n")

    watcher = ReplayWatcher(folder, use_cache=use_cache, debounce_seconds=debounce)

    # Check for existing demos
    existing = watcher.scan_existing()
    if existing:
        cached_count = sum(1 for p in existing if watcher.is_analyzed(p))
        console.print(
            f"[yellow]Found {len(existing)} existing demo(s) ({cached_count} cached)[/yellow]\n"
        )

    # Lazy import for analysis
    if auto_analyze:
        DemoParser, _ = _import_parser()
        metrics_funcs = _import_metrics()
        calculate_engagement_metrics = metrics_funcs["calculate_engagement_metrics"]

    @watcher.on_new_demo
    def handle_new_demo(event: DemoFileEvent) -> None:
        console.print(f"\n[green]New demo detected:[/green] {event.filename}")

        if auto_analyze:
            console.print("Starting analysis...")
            try:
                parser = DemoParser(event.file_path)
                data = parser.parse()

                console.print(f"  Map: {data.map_name}")
                console.print(f"  Duration: {data.duration_seconds:.1f}s")
                console.print(f"  Players: {len(data.player_names)}")

                # Quick stats
                metrics_result = calculate_engagement_metrics(data)
                for _sid, m in sorted(
                    metrics_result.items(), key=lambda x: x[1].total_kills, reverse=True
                )[:3]:
                    console.print(f"  Top: {m.player_name} - {m.total_kills}K/{m.total_deaths}D")

                # Mark as analyzed so we skip it next time
                watcher.mark_analyzed(event.file_path)
                console.print("  [dim]Cached for future runs[/dim]")

            except Exception as e:
                console.print(f"[red]Analysis failed:[/red] {e}")

    try:
        watcher.start(blocking=True)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping watcher...[/yellow]")
        watcher.stop()


@app.command()
def info() -> None:
    """
    Display information about OpenSight and the environment.

    This command uses zero heavy dependencies - no demoparser2 loading.
    """
    import platform as plat

    console.print(f"\n[bold blue]OpenSight[/bold blue] v{__version__}\n")

    table = Table(show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Python", plat.python_version())
    table.add_row("Platform", plat.system())
    table.add_row("Architecture", plat.machine())

    # Check for dependencies WITHOUT importing them fully
    deps = []

    # Check demoparser2 availability without importing (avoid loading Rust binary)
    try:
        import importlib.util

        spec = importlib.util.find_spec("demoparser2")
        if spec is not None:
            deps.append(("demoparser2", "[green]available[/green]"))
        else:
            deps.append(("demoparser2", "[red]not installed[/red]"))
    except Exception:
        deps.append(("demoparser2", "[red]not installed[/red]"))

    # Check awpy availability
    try:
        spec = importlib.util.find_spec("awpy")
        if spec is not None:
            deps.append(("awpy", "[green]available[/green]"))
        else:
            deps.append(("awpy", "[yellow]not installed[/yellow]"))
    except Exception:
        deps.append(("awpy", "[yellow]not installed[/yellow]"))

    for name, version in deps:
        table.add_row(name, version)

    # Check replays folder using platform-specific path (no heavy imports)
    try:
        _, _, get_default_replays_folder = _import_watcher()
        replays_folder = get_default_replays_folder()
        folder_status = (
            "[green]exists[/green]" if replays_folder.exists() else "[yellow]not found[/yellow]"
        )
        table.add_row("Replays Folder", f"{replays_folder} ({folder_status})")
    except Exception:
        table.add_row("Replays Folder", "[yellow]unable to determine[/yellow]")

    console.print(table)


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
