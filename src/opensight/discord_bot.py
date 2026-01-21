"""
Discord Bot Integration for OpenSight CS2 Analyzer

Provides real-time demo analysis and coaching insights through Discord.

Commands:
    !analyze <url>     - Analyze a demo from URL
    !decode <code>     - Decode a CS2 share code
    !stats <steamid>   - Get player stats from last analysis
    !coaching          - Get AI coaching tips from last analysis
    !help              - Show available commands

Environment Variables:
    DISCORD_BOT_TOKEN  - Your Discord bot token
    OPENSIGHT_API_URL  - OpenSight API URL (default: http://localhost:7860)

Usage:
    pip install discord.py aiohttp
    export DISCORD_BOT_TOKEN="your-token"
    python -m opensight.discord_bot
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Check for discord.py availability
try:
    import discord
    from discord.ext import commands
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    logger.warning("discord.py not installed. Install with: pip install discord.py")

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logger.warning("aiohttp not installed. Install with: pip install aiohttp")


# Configuration
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
OPENSIGHT_API_URL = os.getenv("OPENSIGHT_API_URL", "http://localhost:7860")


class OpenSightBot(commands.Bot if DISCORD_AVAILABLE else object):
    """Discord bot for CS2 demo analysis."""

    def __init__(self) -> None:
        if not DISCORD_AVAILABLE:
            raise ImportError("discord.py is required. Install with: pip install discord.py")

        intents = discord.Intents.default()
        intents.message_content = True

        super().__init__(
            command_prefix="!",
            intents=intents,
            description="OpenSight CS2 Demo Analyzer Bot"
        )

        self.api_url = OPENSIGHT_API_URL
        self.last_analysis: dict[int, dict] = {}  # channel_id -> analysis data
        self._session: aiohttp.ClientSession | None = None

    async def setup_hook(self) -> None:
        """Called when the bot is starting up."""
        self._session = aiohttp.ClientSession()
        logger.info(f"Bot connected to API at {self.api_url}")

    async def close(self) -> None:
        """Clean up resources."""
        if self._session:
            await self._session.close()
        await super().close()

    async def on_ready(self) -> None:
        """Called when the bot is ready."""
        logger.info(f"Logged in as {self.user} (ID: {self.user.id})")
        logger.info(f"Connected to {len(self.guilds)} guilds")

    async def _api_request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any
    ) -> dict | None:
        """Make a request to the OpenSight API."""
        if not self._session:
            self._session = aiohttp.ClientSession()

        url = f"{self.api_url}{endpoint}"
        try:
            async with self._session.request(method, url, **kwargs) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error = await response.text()
                    logger.error(f"API error {response.status}: {error}")
                    return None
        except Exception as e:
            logger.error(f"API request failed: {e}")
            return None


def create_bot() -> OpenSightBot:
    """Create and configure the Discord bot."""
    if not DISCORD_AVAILABLE:
        raise ImportError("discord.py is required. Install with: pip install discord.py")
    if not AIOHTTP_AVAILABLE:
        raise ImportError("aiohttp is required. Install with: pip install aiohttp")

    bot = OpenSightBot()

    @bot.command(name="ping")
    async def ping(ctx: commands.Context) -> None:
        """Check if the bot is responsive."""
        latency = round(bot.latency * 1000)
        await ctx.send(f"🏓 Pong! Latency: {latency}ms")

    @bot.command(name="health")
    async def health(ctx: commands.Context) -> None:
        """Check OpenSight API health."""
        async with ctx.typing():
            result = await bot._api_request("GET", "/health")
            if result:
                await ctx.send(f"✅ OpenSight API is healthy\nVersion: {result.get('version', 'unknown')}")
            else:
                await ctx.send("❌ OpenSight API is not responding")

    @bot.command(name="decode")
    async def decode(ctx: commands.Context, code: str) -> None:
        """Decode a CS2 share code.

        Usage: !decode CSGO-xxxxx-xxxxx-xxxxx-xxxxx-xxxxx
        """
        async with ctx.typing():
            result = await bot._api_request(
                "POST",
                "/decode",
                json={"code": code}
            )

            if result:
                embed = discord.Embed(
                    title="🔓 Share Code Decoded",
                    color=discord.Color.green()
                )
                embed.add_field(name="Match ID", value=str(result.get("match_id", "N/A")), inline=True)
                embed.add_field(name="Outcome ID", value=str(result.get("outcome_id", "N/A")), inline=True)
                embed.add_field(name="Token", value=str(result.get("token", "N/A")), inline=True)
                await ctx.send(embed=embed)
            else:
                await ctx.send("❌ Failed to decode share code. Make sure it's a valid CS2 share code.")

    @bot.command(name="analyze")
    async def analyze(ctx: commands.Context, url: str | None = None) -> None:
        """Analyze a CS2 demo file.

        Usage:
            !analyze <url>  - Analyze demo from URL
            Attach a .dem file to analyze it directly
        """
        # Check for attached files
        if ctx.message.attachments:
            attachment = ctx.message.attachments[0]
            if not attachment.filename.endswith(('.dem', '.dem.gz')):
                await ctx.send("❌ Please attach a .dem or .dem.gz file")
                return

            await ctx.send(f"📥 Downloading {attachment.filename}...")

            async with ctx.typing():
                # Download the file
                file_data = await attachment.read()

                # Create form data
                form = aiohttp.FormData()
                form.add_field(
                    'file',
                    file_data,
                    filename=attachment.filename,
                    content_type='application/octet-stream'
                )

                await ctx.send("🔍 Analyzing demo... This may take a minute.")

                result = await bot._api_request("POST", "/analyze", data=form)

                if result:
                    bot.last_analysis[ctx.channel.id] = result
                    await _send_analysis_summary(ctx, result)
                else:
                    await ctx.send("❌ Analysis failed. Please try again.")
        elif url:
            await ctx.send("❌ URL analysis not yet supported. Please attach the demo file directly.")
        else:
            await ctx.send("❌ Please attach a .dem file or provide a URL")

    @bot.command(name="stats")
    async def stats(ctx: commands.Context, player_name: str | None = None) -> None:
        """Get player stats from the last analysis.

        Usage: !stats [player_name]
        """
        analysis = bot.last_analysis.get(ctx.channel.id)
        if not analysis:
            await ctx.send("❌ No analysis data. Run !analyze first.")
            return

        players = analysis.get("players", {})
        if not players:
            await ctx.send("❌ No player data available")
            return

        if player_name:
            # Find specific player
            player = None
            for p in players.values():
                if player_name.lower() in p.get("name", "").lower():
                    player = p
                    break

            if player:
                await _send_player_stats(ctx, player)
            else:
                await ctx.send(f"❌ Player '{player_name}' not found")
        else:
            # Send summary of all players
            embed = discord.Embed(
                title=f"📊 Match Stats - {analysis.get('map_name', 'Unknown Map')}",
                color=discord.Color.blue()
            )

            # Sort by rating
            sorted_players = sorted(
                players.values(),
                key=lambda p: p.get("hltv_rating", 0),
                reverse=True
            )

            for p in sorted_players[:10]:
                name = p.get("name", "Unknown")
                rating = p.get("hltv_rating", 0)
                kd = f"{p.get('kills', 0)}/{p.get('deaths', 0)}"
                adr = p.get("adr", 0)

                embed.add_field(
                    name=f"{name} ({p.get('team', '?')})",
                    value=f"Rating: **{rating:.2f}** | K/D: {kd} | ADR: {adr:.0f}",
                    inline=False
                )

            await ctx.send(embed=embed)

    @bot.command(name="coaching")
    async def coaching(ctx: commands.Context, player_name: str | None = None) -> None:
        """Get AI coaching insights from the last analysis.

        Usage: !coaching [player_name]
        """
        analysis = bot.last_analysis.get(ctx.channel.id)
        if not analysis:
            await ctx.send("❌ No analysis data. Run !analyze first.")
            return

        coaching_data = analysis.get("coaching", [])
        if not coaching_data:
            await ctx.send("ℹ️ No coaching insights available for this demo")
            return

        embed = discord.Embed(
            title="🎯 AI Coaching Insights",
            color=discord.Color.gold()
        )

        for insight in coaching_data[:5]:
            embed.add_field(
                name=insight.get("category", "Tip"),
                value=insight.get("message", "No details"),
                inline=False
            )

        await ctx.send(embed=embed)

    @bot.command(name="maps")
    async def maps(ctx: commands.Context) -> None:
        """List available map information."""
        async with ctx.typing():
            result = await bot._api_request("GET", "/maps")
            if result:
                maps_list = result.get("maps", [])
                await ctx.send(f"🗺️ Available maps: {', '.join(maps_list)}")
            else:
                await ctx.send("❌ Could not fetch map list")

    async def _send_analysis_summary(ctx: commands.Context, analysis: dict) -> None:
        """Send a summary of the analysis."""
        embed = discord.Embed(
            title=f"✅ Analysis Complete - {analysis.get('map_name', 'Unknown')}",
            color=discord.Color.green()
        )

        embed.add_field(
            name="Match Info",
            value=f"Rounds: {analysis.get('total_rounds', '?')} | "
                  f"Score: {analysis.get('team1_score', 0)}-{analysis.get('team2_score', 0)}",
            inline=False
        )

        players = analysis.get("players", {})
        if players:
            mvp = max(players.values(), key=lambda p: p.get("hltv_rating", 0))
            embed.add_field(
                name="MVP",
                value=f"{mvp.get('name', 'Unknown')} - {mvp.get('hltv_rating', 0):.2f} rating",
                inline=True
            )

        embed.set_footer(text="Use !stats or !coaching for more details")
        await ctx.send(embed=embed)

    async def _send_player_stats(ctx: commands.Context, player: dict) -> None:
        """Send detailed stats for a player."""
        embed = discord.Embed(
            title=f"📊 {player.get('name', 'Unknown')} ({player.get('team', '?')})",
            color=discord.Color.blue()
        )

        # Combat stats
        embed.add_field(
            name="Combat",
            value=f"K/D/A: {player.get('kills', 0)}/{player.get('deaths', 0)}/{player.get('assists', 0)}\n"
                  f"HS%: {player.get('headshot_percentage', 0):.0f}%\n"
                  f"ADR: {player.get('adr', 0):.1f}",
            inline=True
        )

        # Rating
        embed.add_field(
            name="Rating",
            value=f"HLTV: **{player.get('hltv_rating', 0):.2f}**\n"
                  f"KAST: {player.get('kast_percentage', 0):.0f}%",
            inline=True
        )

        # Advanced stats if available
        ttd = player.get("ttd_mean")
        cp = player.get("crosshair_placement_mean")
        if ttd or cp:
            advanced = ""
            if ttd:
                advanced += f"TTD: {ttd:.0f}ms\n"
            if cp:
                advanced += f"CP: {cp:.1f}°"
            embed.add_field(name="Advanced", value=advanced or "N/A", inline=True)

        await ctx.send(embed=embed)

    return bot


def run_bot() -> None:
    """Run the Discord bot."""
    if not DISCORD_BOT_TOKEN:
        raise ValueError(
            "DISCORD_BOT_TOKEN environment variable not set.\n"
            "Get a token from https://discord.com/developers/applications"
        )

    bot = create_bot()
    bot.run(DISCORD_BOT_TOKEN)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    run_bot()
