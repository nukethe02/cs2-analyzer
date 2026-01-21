"""
OpenSight CLI - Command Line Interface for CS2 Analytics

Provides commands for:
- Analyzing demo files
- Decoding share codes
- Watching for new replays
- Generating reports
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from opensight import __version__
from opensight.sharecode import decode_sharecode, ShareCodeInfo
from opensight.parser import DemoParser, DemoData
from opensight.metrics import (
    calculate_ttd,
    calculate_crosshair_placement,
    calculate_engagement_metrics,
    calculate_economy_metrics,
    calculate_utility_metrics,
    calculate_trade_metrics,
    calculate_opening_metrics,
    calculate_comprehensive_metrics,
)
from opensight.watcher import ReplayWatcher, DemoFileEvent, get_default_replays_folder
from opensight.export import export_analysis

app = typer.Typer(
    name="opensight",
    help="Local CS2 analytics framework - professional-grade metrics without cloud dependencies",
    add_completion=False,
)
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
        help="Show version and exit"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-V",
        help="Enable verbose output"
    )
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
    player: Optional[str] = typer.Option(
        None,
        "--player",
        "-p",
        help="Filter results to a specific player (name or Steam ID)"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for results (format detected from extension: .json, .csv, .xlsx, .html)"
    ),
    metrics: str = typer.Option(
        "all",
        "--metrics",
        "-m",
        help="Metrics to calculate: all, ttd, cp, engagement, economy, utility, trades, opening"
    ),
) -> None:
    """
    Analyze a CS2 demo file and display metrics.

    Calculates professional-grade analytics including:
    - Time to Damage (TTD)
    - Crosshair Placement (CP)
    - Kill/Death statistics
    - Damage per round
    """
    console.print(f"\n[bold blue]OpenSight[/bold blue] - Analyzing demo...\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Parse the demo
        task = progress.add_task("Parsing demo file...", total=None)
        try:
            parser = DemoParser(demo_path)
            data = parser.parse()
        except Exception as e:
            console.print(f"[red]Error parsing demo:[/red] {e}")
            raise typer.Exit(1)

        progress.update(task, description="Demo parsed successfully!")

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
    steam_id: Optional[int] = None
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

    # Calculate and display metrics
    if metrics in ("all", "engagement"):
        _display_engagement_metrics(data, steam_id)

    if metrics in ("all", "ttd"):
        _display_ttd_metrics(data, steam_id)

    if metrics in ("all", "cp"):
        _display_cp_metrics(data, steam_id)

    if metrics in ("all", "economy"):
        _display_economy_metrics(data, steam_id)

    if metrics in ("all", "utility"):
        _display_utility_metrics(data, steam_id)

    if metrics in ("all", "trades"):
        _display_trade_metrics(data, steam_id)

    if metrics in ("all", "opening"):
        _display_opening_metrics(data, steam_id)

    # Export if requested
    if output:
        comprehensive = calculate_comprehensive_metrics(data, steam_id)
        export_analysis(data, comprehensive, output)
        console.print(f"\n[green]Results exported to:[/green] {output}")


def _display_engagement_metrics(data: DemoData, steam_id: Optional[int]) -> None:
    """Display engagement metrics table."""
    metrics = calculate_engagement_metrics(data, steam_id)

    if not metrics:
        console.print("[yellow]No engagement metrics available[/yellow]")
        return

    table = Table(title="Engagement Metrics")
    table.add_column("Player", style="cyan")
    table.add_column("K", justify="right")
    table.add_column("D", justify="right")
    table.add_column("K/D", justify="right")
    table.add_column("HS%", justify="right")
    table.add_column("DPR", justify="right")

    for sid, m in sorted(metrics.items(), key=lambda x: x[1].total_kills, reverse=True):
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


def _display_ttd_metrics(data: DemoData, steam_id: Optional[int]) -> None:
    """Display Time to Damage metrics table."""
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

    for sid, ttd in sorted(ttd_results.items(), key=lambda x: x[1].mean_ttd_ms):
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


def _display_cp_metrics(data: DemoData, steam_id: Optional[int]) -> None:
    """Display Crosshair Placement metrics table."""
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

    for sid, cp in sorted(cp_results.items(), key=lambda x: x[1].placement_score, reverse=True):
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


def _display_economy_metrics(data: DemoData, steam_id: Optional[int]) -> None:
    """Display economy metrics table."""
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
        ...,
        help="CS2 share code (e.g., CSGO-xxxxx-xxxxx-xxxxx-xxxxx-xxxxx)"
    )
) -> None:
    """
    Decode a CS2 share code to extract match metadata.

    The share code contains encoded information about the match
    including match ID, outcome ID, and token.
    """
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
    folder: Optional[Path] = typer.Option(
        None,
        "--folder",
        "-f",
        help="Folder to watch (defaults to CS2 replays folder)",
        exists=False,
        dir_okay=True,
        file_okay=False,
    ),
    auto_analyze: bool = typer.Option(
        True,
        "--analyze/--no-analyze",
        "-a",
        help="Automatically analyze new demos"
    ),
) -> None:
    """
    Watch for new CS2 replay files and optionally analyze them.

    Monitors the specified folder (or default CS2 replays folder)
    for new .dem files and triggers analysis automatically.
    """
    if folder is None:
        folder = get_default_replays_folder()

    console.print(f"\n[bold blue]OpenSight[/bold blue] - Watching for replays\n")
    console.print(f"[cyan]Folder:[/cyan] {folder}")
    console.print(f"[cyan]Auto-analyze:[/cyan] {'Yes' if auto_analyze else 'No'}")
    console.print("\nPress [bold]Ctrl+C[/bold] to stop...\n")

    watcher = ReplayWatcher(folder)

    # Check for existing demos
    existing = watcher.scan_existing()
    if existing:
        console.print(f"[yellow]Found {len(existing)} existing demo(s)[/yellow]\n")

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
                metrics = calculate_engagement_metrics(data)
                for sid, m in sorted(metrics.items(), key=lambda x: x[1].total_kills, reverse=True)[:3]:
                    console.print(f"  Top: {m.player_name} - {m.total_kills}K/{m.total_deaths}D")

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
    """
    import platform as plat

    console.print(f"\n[bold blue]OpenSight[/bold blue] v{__version__}\n")

    table = Table(show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Python", plat.python_version())
    table.add_row("Platform", plat.system())
    table.add_row("Architecture", plat.machine())

    # Check for dependencies
    deps = []
    try:
        import demoparser2
        deps.append(("demoparser2", getattr(demoparser2, "__version__", "installed")))
    except ImportError:
        deps.append(("demoparser2", "[red]not installed[/red]"))

    try:
        import awpy
        deps.append(("awpy", getattr(awpy, "__version__", "installed")))
    except ImportError:
        deps.append(("awpy", "[red]not installed[/red]"))

    for name, version in deps:
        table.add_row(name, version)

    replays_folder = get_default_replays_folder()
    folder_status = "[green]exists[/green]" if replays_folder.exists() else "[yellow]not found[/yellow]"
    table.add_row("Replays Folder", f"{replays_folder} ({folder_status})")

    console.print(table)


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
