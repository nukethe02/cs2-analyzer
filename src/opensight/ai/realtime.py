"""
Real-Time Coaching Mode for CS2 Demo Analyzer.

Provides live coaching during practice servers and scrims through
WebSocket streaming of parsed data with voice/text alerts.
"""

import json
import queue
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# ============================================================================
# Alert Types and Priorities
# ============================================================================


class AlertPriority(Enum):
    """Priority levels for coaching alerts."""

    CRITICAL = 1  # Immediate attention needed
    HIGH = 2  # Important feedback
    MEDIUM = 3  # General coaching tip
    LOW = 4  # Nice to know
    INFO = 5  # Informational


class AlertType(Enum):
    """Types of real-time coaching alerts."""

    # Economy alerts
    ECO_ROUND = "eco_round"
    FORCE_BUY = "force_buy"
    SAVE_ROUND = "save_round"
    LOW_MONEY = "low_money"

    # Tactical alerts
    ROTATE_NEEDED = "rotate_needed"
    FLASH_INCOMING = "flash_incoming"
    ENEMY_SPOTTED = "enemy_spotted"
    PLANT_REMINDER = "plant_reminder"
    TIME_WARNING = "time_warning"

    # Performance alerts
    TRADE_OPPORTUNITY = "trade_opportunity"
    TEAMMATE_DOWN = "teammate_down"
    CLUTCH_SITUATION = "clutch_situation"
    MULTI_KILL = "multi_kill"

    # Mistake alerts
    TEAM_FLASH = "team_flash"
    OVERAGGRESSION = "overaggression"
    POOR_POSITION = "poor_position"
    MISSED_TRADE = "missed_trade"
    WASTED_UTILITY = "wasted_utility"

    # Positive reinforcement
    GOOD_TRADE = "good_trade"
    NICE_FLASH = "nice_flash"
    SMART_PLAY = "smart_play"
    ROUND_WON = "round_won"


@dataclass
class CoachingAlert:
    """A single coaching alert."""

    alert_id: str
    alert_type: AlertType
    priority: AlertPriority
    message: str
    details: str = ""
    timestamp: float = 0.0
    round_num: int = 0
    tick: int = 0
    target_player: str | None = None  # Steam ID of relevant player
    position: tuple[float, float, float] | None = None
    expires_in_ms: int = 5000  # Alert expires after this time
    voice_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "priority": self.priority.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
            "round_num": self.round_num,
            "tick": self.tick,
            "target_player": self.target_player,
            "position": self.position,
            "expires_in_ms": self.expires_in_ms,
            "voice_enabled": self.voice_enabled,
        }


# ============================================================================
# Game State Tracking
# ============================================================================


@dataclass
class PlayerState:
    """Real-time state of a player."""

    steamid: str
    name: str
    team: str  # "ct" or "t"
    alive: bool = True
    health: int = 100
    armor: int = 0
    money: int = 800
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    has_defuser: bool = False
    has_helmet: bool = False
    primary_weapon: str = ""
    secondary_weapon: str = ""
    grenades: list[str] = field(default_factory=list)

    # Round stats
    kills_this_round: int = 0
    damage_this_round: int = 0
    flashes_thrown: int = 0
    enemies_flashed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "steamid": self.steamid,
            "name": self.name,
            "team": self.team,
            "alive": self.alive,
            "health": self.health,
            "armor": self.armor,
            "money": self.money,
            "position": {"x": self.x, "y": self.y, "z": self.z},
            "angles": {"pitch": self.pitch, "yaw": self.yaw},
            "has_defuser": self.has_defuser,
            "has_helmet": self.has_helmet,
            "weapons": {
                "primary": self.primary_weapon,
                "secondary": self.secondary_weapon,
                "grenades": self.grenades,
            },
            "round_stats": {
                "kills": self.kills_this_round,
                "damage": self.damage_this_round,
                "flashes_thrown": self.flashes_thrown,
                "enemies_flashed": self.enemies_flashed,
            },
        }


@dataclass
class RoundState:
    """Real-time state of the current round."""

    round_num: int
    phase: str = "freezetime"  # freezetime, live, planted, over
    time_remaining: float = 115.0
    bomb_planted: bool = False
    bomb_site: str = ""
    bomb_time_remaining: float = 40.0

    ct_alive: int = 5
    t_alive: int = 5
    ct_score: int = 0
    t_score: int = 0

    ct_money: int = 0
    t_money: int = 0

    first_kill_team: str | None = None
    first_death_team: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_num": self.round_num,
            "phase": self.phase,
            "time_remaining": round(self.time_remaining, 1),
            "bomb": {
                "planted": self.bomb_planted,
                "site": self.bomb_site,
                "time_remaining": round(self.bomb_time_remaining, 1) if self.bomb_planted else None,
            },
            "alive": {"ct": self.ct_alive, "t": self.t_alive},
            "score": {"ct": self.ct_score, "t": self.t_score},
            "economy": {"ct_total": self.ct_money, "t_total": self.t_money},
            "first_blood": {"kill_team": self.first_kill_team, "death_team": self.first_death_team},
        }


@dataclass
class GameState:
    """Complete real-time game state."""

    map_name: str
    tick: int = 0
    timestamp: float = 0.0

    players: dict[str, PlayerState] = field(default_factory=dict)
    round_state: RoundState = field(default_factory=lambda: RoundState(round_num=1))

    # Event history for pattern detection
    recent_kills: deque = field(default_factory=lambda: deque(maxlen=20))
    recent_damages: deque = field(default_factory=lambda: deque(maxlen=50))
    recent_grenades: deque = field(default_factory=lambda: deque(maxlen=30))

    # Tracked player (for coaching focus)
    focus_player: str | None = None  # Steam ID

    def to_dict(self) -> dict[str, Any]:
        return {
            "map_name": self.map_name,
            "tick": self.tick,
            "timestamp": self.timestamp,
            "players": {sid: p.to_dict() for sid, p in self.players.items()},
            "round": self.round_state.to_dict(),
            "focus_player": self.focus_player,
        }


# ============================================================================
# Alert Detection Engine
# ============================================================================


class AlertDetector:
    """
    Detects situations requiring coaching alerts.
    """

    def __init__(self, focus_team: str = "ct"):
        self.focus_team = focus_team  # "ct" or "t"
        self.alert_cooldowns: dict[str, float] = {}  # alert_type -> last_triggered
        self.cooldown_seconds = 10.0  # Don't repeat same alert within this time

        self._alert_counter = 0

    def detect_alerts(
        self, game_state: GameState, prev_state: GameState | None = None
    ) -> list[CoachingAlert]:
        """
        Detect alerts from current game state.

        Args:
            game_state: Current game state
            prev_state: Previous game state for comparison

        Returns:
            List of detected alerts
        """
        alerts = []

        # Economy alerts
        alerts.extend(self._detect_economy_alerts(game_state))

        # Time-based alerts
        alerts.extend(self._detect_time_alerts(game_state))

        # Tactical situation alerts
        alerts.extend(self._detect_tactical_alerts(game_state, prev_state))

        # Performance alerts
        alerts.extend(self._detect_performance_alerts(game_state, prev_state))

        # Filter by cooldown
        alerts = self._filter_by_cooldown(alerts)

        return alerts

    def _detect_economy_alerts(self, state: GameState) -> list[CoachingAlert]:
        """Detect economy-related alerts."""
        alerts = []
        round_state = state.round_state

        # Only during freeze time
        if round_state.phase != "freezetime":
            return alerts

        # Get team money
        team_players = [p for p in state.players.values() if p.team == self.focus_team]
        team_money = sum(p.money for p in team_players)
        avg_money = team_money / max(1, len(team_players))

        # Low money alert
        if avg_money < 2000 and round_state.round_num > 1:
            alerts.append(
                self._create_alert(
                    AlertType.LOW_MONEY,
                    AlertPriority.MEDIUM,
                    f"Low economy - ${int(avg_money)} average",
                    "Consider eco or force buy together",
                    round_state.round_num,
                    state.tick,
                )
            )

        # Eco round suggestion
        if avg_money < 1500:
            alerts.append(
                self._create_alert(
                    AlertType.ECO_ROUND,
                    AlertPriority.HIGH,
                    "Eco round recommended",
                    "Save for next round full buy",
                    round_state.round_num,
                    state.tick,
                )
            )

        return alerts

    def _detect_time_alerts(self, state: GameState) -> list[CoachingAlert]:
        """Detect time-related alerts."""
        alerts = []
        round_state = state.round_state

        if round_state.phase != "live":
            return alerts

        # Time warning for T side
        if self.focus_team == "t":
            if round_state.time_remaining < 30 and not round_state.bomb_planted:
                alerts.append(
                    self._create_alert(
                        AlertType.TIME_WARNING,
                        AlertPriority.HIGH,
                        f"{int(round_state.time_remaining)} seconds!",
                        "Execute now or plant bomb",
                        round_state.round_num,
                        state.tick,
                        expires_in_ms=3000,
                    )
                )

            if round_state.time_remaining < 15 and not round_state.bomb_planted:
                alerts.append(
                    self._create_alert(
                        AlertType.PLANT_REMINDER,
                        AlertPriority.CRITICAL,
                        "PLANT NOW!",
                        "Time running out!",
                        round_state.round_num,
                        state.tick,
                        expires_in_ms=2000,
                    )
                )

        return alerts

    def _detect_tactical_alerts(
        self, state: GameState, prev_state: GameState | None
    ) -> list[CoachingAlert]:
        """Detect tactical situation alerts."""
        alerts = []
        round_state = state.round_state

        if round_state.phase not in ["live", "planted"]:
            return alerts

        # Clutch situation
        team_alive = round_state.ct_alive if self.focus_team == "ct" else round_state.t_alive
        enemy_alive = round_state.t_alive if self.focus_team == "ct" else round_state.ct_alive

        if team_alive == 1 and enemy_alive >= 1:
            alerts.append(
                self._create_alert(
                    AlertType.CLUTCH_SITUATION,
                    AlertPriority.HIGH,
                    f"1v{enemy_alive} Clutch!",
                    "Take your time, isolate duels",
                    round_state.round_num,
                    state.tick,
                    expires_in_ms=10000,
                )
            )

        # Rotate needed (CT specific)
        if self.focus_team == "ct" and round_state.bomb_planted:
            alerts.append(
                self._create_alert(
                    AlertType.ROTATE_NEEDED,
                    AlertPriority.HIGH,
                    f"Bomb planted {round_state.bomb_site}!",
                    f"Rotate and retake {round_state.bomb_site}",
                    round_state.round_num,
                    state.tick,
                    expires_in_ms=5000,
                )
            )

        return alerts

    def _detect_performance_alerts(
        self, state: GameState, prev_state: GameState | None
    ) -> list[CoachingAlert]:
        """Detect performance-related alerts."""
        alerts = []

        if not prev_state:
            return alerts

        # Check for kills (positive reinforcement)
        focus_players = [p for p in state.players.values() if p.team == self.focus_team]

        for player in focus_players:
            prev_player = prev_state.players.get(player.steamid)
            if not prev_player:
                continue

            # Multi-kill
            kills_gained = player.kills_this_round - prev_player.kills_this_round
            if kills_gained >= 2:
                alerts.append(
                    self._create_alert(
                        AlertType.MULTI_KILL,
                        AlertPriority.LOW,
                        f"Nice! {player.name} double kill!",
                        "",
                        state.round_state.round_num,
                        state.tick,
                        target_player=player.steamid,
                        expires_in_ms=3000,
                    )
                )

            # Good flash
            if player.enemies_flashed > prev_player.enemies_flashed:
                flashed = player.enemies_flashed - prev_player.enemies_flashed
                alerts.append(
                    self._create_alert(
                        AlertType.NICE_FLASH,
                        AlertPriority.LOW,
                        f"Good flash! {flashed} enemies blinded",
                        "",
                        state.round_state.round_num,
                        state.tick,
                        target_player=player.steamid,
                        expires_in_ms=2000,
                    )
                )

        # Check for teammate deaths (trade opportunity)
        for recent_kill in list(state.recent_kills)[-3:]:
            victim_team = state.players.get(recent_kill.get("victim_steamid"), {})
            if hasattr(victim_team, "team") and victim_team.team == self.focus_team:
                # Teammate died - check for trade opportunity
                killer_steamid = recent_kill.get("attacker_steamid")
                killer = state.players.get(killer_steamid)

                if killer and killer.alive:
                    alerts.append(
                        self._create_alert(
                            AlertType.TRADE_OPPORTUNITY,
                            AlertPriority.HIGH,
                            f"Trade {recent_kill.get('victim_name', 'teammate')}!",
                            "Enemy at known position",
                            state.round_state.round_num,
                            state.tick,
                            expires_in_ms=4000,
                        )
                    )

        return alerts

    def _create_alert(
        self,
        alert_type: AlertType,
        priority: AlertPriority,
        message: str,
        details: str,
        round_num: int,
        tick: int,
        target_player: str | None = None,
        position: tuple[float, float, float] | None = None,
        expires_in_ms: int = 5000,
    ) -> CoachingAlert:
        """Create a coaching alert."""
        self._alert_counter += 1
        return CoachingAlert(
            alert_id=f"alert_{self._alert_counter}_{int(time.time() * 1000)}",
            alert_type=alert_type,
            priority=priority,
            message=message,
            details=details,
            timestamp=time.time(),
            round_num=round_num,
            tick=tick,
            target_player=target_player,
            position=position,
            expires_in_ms=expires_in_ms,
        )

    def _filter_by_cooldown(self, alerts: list[CoachingAlert]) -> list[CoachingAlert]:
        """Filter alerts by cooldown to prevent spam."""
        filtered = []
        current_time = time.time()

        for alert in alerts:
            key = alert.alert_type.value
            last_triggered = self.alert_cooldowns.get(key, 0)

            if current_time - last_triggered >= self.cooldown_seconds:
                filtered.append(alert)
                self.alert_cooldowns[key] = current_time

        return filtered


# ============================================================================
# Real-Time Coaching Session
# ============================================================================


class RealtimeCoachingSession:
    """
    Manages a real-time coaching session.
    """

    def __init__(self, session_id: str, focus_player: str | None = None, focus_team: str = "ct"):
        self.session_id = session_id
        self.focus_player = focus_player
        self.focus_team = focus_team

        self.game_state = GameState(map_name="unknown")
        self.prev_state: GameState | None = None

        self.alert_detector = AlertDetector(focus_team=focus_team)
        self.alert_history: deque = deque(maxlen=100)

        # Callbacks for alerts
        self.alert_callbacks: list[Callable[[CoachingAlert], None]] = []

        # Session stats
        self.started_at = datetime.now()
        self.alerts_sent = 0
        self.rounds_coached = 0

        # Running state
        self.active = False
        self._alert_queue: queue.Queue = queue.Queue()

    def start(self) -> None:
        """Start the coaching session."""
        self.active = True
        self.started_at = datetime.now()

    def stop(self) -> None:
        """Stop the coaching session."""
        self.active = False

    def update_state(self, state_update: dict[str, Any]) -> list[CoachingAlert]:
        """
        Update game state from incoming data.

        Args:
            state_update: State update from game/parser

        Returns:
            List of triggered alerts
        """
        if not self.active:
            return []

        # Store previous state
        self.prev_state = GameState(
            map_name=self.game_state.map_name,
            tick=self.game_state.tick,
            players={
                sid: PlayerState(
                    **{
                        k: v
                        for k, v in p.to_dict().items()
                        if k not in ["position", "angles", "weapons", "round_stats"]
                    }
                )
                for sid, p in self.game_state.players.items()
            },
            round_state=RoundState(**self.game_state.round_state.to_dict()),
        )

        # Apply update
        self._apply_state_update(state_update)

        # Detect alerts
        alerts = self.alert_detector.detect_alerts(self.game_state, self.prev_state)

        # Process alerts
        for alert in alerts:
            self.alerts_sent += 1
            self.alert_history.append(alert)
            self._alert_queue.put(alert)

            # Call registered callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception:
                    pass

        return alerts

    def _apply_state_update(self, update: dict[str, Any]) -> None:
        """Apply state update to game state."""
        # Update tick
        if "tick" in update:
            self.game_state.tick = update["tick"]

        if "timestamp" in update:
            self.game_state.timestamp = update["timestamp"]

        if "map_name" in update:
            self.game_state.map_name = update["map_name"]

        # Update round state
        if "round" in update:
            round_data = update["round"]
            rs = self.game_state.round_state

            if "round_num" in round_data:
                if round_data["round_num"] != rs.round_num:
                    self.rounds_coached += 1
                rs.round_num = round_data["round_num"]

            if "phase" in round_data:
                rs.phase = round_data["phase"]
            if "time_remaining" in round_data:
                rs.time_remaining = round_data["time_remaining"]
            if "bomb_planted" in round_data:
                rs.bomb_planted = round_data["bomb_planted"]
            if "bomb_site" in round_data:
                rs.bomb_site = round_data["bomb_site"]
            if "ct_alive" in round_data:
                rs.ct_alive = round_data["ct_alive"]
            if "t_alive" in round_data:
                rs.t_alive = round_data["t_alive"]
            if "ct_score" in round_data:
                rs.ct_score = round_data["ct_score"]
            if "t_score" in round_data:
                rs.t_score = round_data["t_score"]

        # Update players
        if "players" in update:
            for steamid, player_data in update["players"].items():
                if steamid not in self.game_state.players:
                    self.game_state.players[steamid] = PlayerState(
                        steamid=steamid,
                        name=player_data.get("name", "Player"),
                        team=player_data.get("team", "ct"),
                    )

                player = self.game_state.players[steamid]

                if "name" in player_data:
                    player.name = player_data["name"]
                if "team" in player_data:
                    player.team = player_data["team"]
                if "alive" in player_data:
                    player.alive = player_data["alive"]
                if "health" in player_data:
                    player.health = player_data["health"]
                if "armor" in player_data:
                    player.armor = player_data["armor"]
                if "money" in player_data:
                    player.money = player_data["money"]
                if "x" in player_data:
                    player.x = player_data["x"]
                if "y" in player_data:
                    player.y = player_data["y"]
                if "z" in player_data:
                    player.z = player_data["z"]
                if "kills" in player_data:
                    player.kills_this_round = player_data["kills"]
                if "damage" in player_data:
                    player.damage_this_round = player_data["damage"]
                if "enemies_flashed" in player_data:
                    player.enemies_flashed = player_data["enemies_flashed"]

        # Record events
        if "kill" in update:
            self.game_state.recent_kills.append(update["kill"])

        if "damage" in update:
            self.game_state.recent_damages.append(update["damage"])

        if "grenade" in update:
            self.game_state.recent_grenades.append(update["grenade"])

    def get_next_alert(self, timeout: float = 0.1) -> CoachingAlert | None:
        """Get the next alert from the queue."""
        try:
            return self._alert_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def register_callback(self, callback: Callable[[CoachingAlert], None]) -> None:
        """Register a callback for alerts."""
        self.alert_callbacks.append(callback)

    def get_session_stats(self) -> dict[str, Any]:
        """Get session statistics."""
        duration = (datetime.now() - self.started_at).total_seconds()
        return {
            "session_id": self.session_id,
            "active": self.active,
            "started_at": self.started_at.isoformat(),
            "duration_seconds": round(duration, 1),
            "alerts_sent": self.alerts_sent,
            "rounds_coached": self.rounds_coached,
            "alerts_per_round": round(self.alerts_sent / max(1, self.rounds_coached), 2),
            "focus_player": self.focus_player,
            "focus_team": self.focus_team,
            "current_round": self.game_state.round_state.round_num,
            "score": {
                "ct": self.game_state.round_state.ct_score,
                "t": self.game_state.round_state.t_score,
            },
        }

    def get_state(self) -> dict[str, Any]:
        """Get current game state."""
        return self.game_state.to_dict()


# ============================================================================
# WebSocket Manager
# ============================================================================


class RealtimeCoachingManager:
    """
    Manages real-time coaching sessions and WebSocket connections.
    """

    def __init__(self):
        self.sessions: dict[str, RealtimeCoachingSession] = {}
        self._session_counter = 0

    def create_session(
        self, focus_player: str | None = None, focus_team: str = "ct"
    ) -> RealtimeCoachingSession:
        """Create a new coaching session."""
        self._session_counter += 1
        session_id = f"session_{self._session_counter}_{int(time.time())}"

        session = RealtimeCoachingSession(
            session_id=session_id, focus_player=focus_player, focus_team=focus_team
        )

        self.sessions[session_id] = session
        session.start()

        return session

    def get_session(self, session_id: str) -> RealtimeCoachingSession | None:
        """Get an existing session."""
        return self.sessions.get(session_id)

    def end_session(self, session_id: str) -> dict[str, Any] | None:
        """End a session and return stats."""
        session = self.sessions.pop(session_id, None)
        if session:
            session.stop()
            return session.get_session_stats()
        return None

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all active sessions."""
        return [
            {
                "session_id": s.session_id,
                "active": s.active,
                "focus_team": s.focus_team,
                "rounds_coached": s.rounds_coached,
            }
            for s in self.sessions.values()
        ]


# ============================================================================
# Text-to-Speech Alert Formatter
# ============================================================================


class VoiceAlertFormatter:
    """
    Formats alerts for text-to-speech output.
    """

    # Priority-specific prefixes
    PRIORITY_PREFIXES = {
        AlertPriority.CRITICAL: "Urgent! ",
        AlertPriority.HIGH: "",
        AlertPriority.MEDIUM: "",
        AlertPriority.LOW: "",
        AlertPriority.INFO: "",
    }

    # Alert type specific formatting
    ALERT_FORMATS = {
        AlertType.CLUTCH_SITUATION: "{message}",
        AlertType.TIME_WARNING: "{message}",
        AlertType.PLANT_REMINDER: "{message}",
        AlertType.ROTATE_NEEDED: "{message}",
        AlertType.TRADE_OPPORTUNITY: "{message}",
        AlertType.ECO_ROUND: "Eco round",
        AlertType.MULTI_KILL: "Nice!",
        AlertType.NICE_FLASH: "Good flash!",
    }

    @classmethod
    def format_for_voice(cls, alert: CoachingAlert) -> str:
        """
        Format an alert for voice output.

        Args:
            alert: The coaching alert

        Returns:
            Text suitable for TTS
        """
        prefix = cls.PRIORITY_PREFIXES.get(alert.priority, "")

        # Use custom format if available
        if alert.alert_type in cls.ALERT_FORMATS:
            message = cls.ALERT_FORMATS[alert.alert_type].format(
                message=alert.message, details=alert.details
            )
        else:
            message = alert.message

        return f"{prefix}{message}"

    @classmethod
    def format_for_display(cls, alert: CoachingAlert) -> dict[str, Any]:
        """
        Format an alert for visual display.

        Args:
            alert: The coaching alert

        Returns:
            Formatted display data
        """
        return {
            "id": alert.alert_id,
            "type": alert.alert_type.value,
            "priority": alert.priority.name,
            "title": alert.message,
            "subtitle": alert.details,
            "expires_at": alert.timestamp + alert.expires_in_ms / 1000,
            "color": cls._get_priority_color(alert.priority),
            "icon": cls._get_alert_icon(alert.alert_type),
        }

    @classmethod
    def _get_priority_color(cls, priority: AlertPriority) -> str:
        """Get color for priority level."""
        colors = {
            AlertPriority.CRITICAL: "#ff0000",
            AlertPriority.HIGH: "#ff8800",
            AlertPriority.MEDIUM: "#ffff00",
            AlertPriority.LOW: "#00ff00",
            AlertPriority.INFO: "#0088ff",
        }
        return colors.get(priority, "#ffffff")

    @classmethod
    def _get_alert_icon(cls, alert_type: AlertType) -> str:
        """Get icon for alert type."""
        icons = {
            AlertType.CLUTCH_SITUATION: "⚔️",
            AlertType.TIME_WARNING: "⏱️",
            AlertType.PLANT_REMINDER: "💣",
            AlertType.ROTATE_NEEDED: "🔄",
            AlertType.TRADE_OPPORTUNITY: "🎯",
            AlertType.ECO_ROUND: "💰",
            AlertType.TEAM_FLASH: "⚠️",
            AlertType.MULTI_KILL: "🔥",
            AlertType.NICE_FLASH: "✨",
            AlertType.ROUND_WON: "🏆",
        }
        return icons.get(alert_type, "📢")


# ============================================================================
# WebSocket Protocol Messages
# ============================================================================


class WSMessageType(Enum):
    """WebSocket message types."""

    # Client -> Server
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    STATE_UPDATE = "state_update"
    CONFIGURE = "configure"

    # Server -> Client
    ALERT = "alert"
    STATE = "state"
    ERROR = "error"
    SESSION_INFO = "session_info"


@dataclass
class WSMessage:
    """WebSocket message structure."""

    msg_type: WSMessageType
    data: dict[str, Any]
    session_id: str | None = None

    def to_json(self) -> str:
        return json.dumps(
            {"type": self.msg_type.value, "session_id": self.session_id, "data": self.data}
        )

    @classmethod
    def from_json(cls, json_str: str) -> "WSMessage":
        data = json.loads(json_str)
        return cls(
            msg_type=WSMessageType(data["type"]),
            data=data.get("data", {}),
            session_id=data.get("session_id"),
        )


# ============================================================================
# Convenience Functions
# ============================================================================

_default_manager: RealtimeCoachingManager | None = None


def get_manager() -> RealtimeCoachingManager:
    """Get or create the default coaching manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = RealtimeCoachingManager()
    return _default_manager


def create_coaching_session(
    focus_player: str | None = None, focus_team: str = "ct"
) -> dict[str, Any]:
    """
    Create a new real-time coaching session.

    Args:
        focus_player: Steam ID of player to focus coaching on
        focus_team: Team to coach ("ct" or "t")

    Returns:
        Session info
    """
    manager = get_manager()
    session = manager.create_session(focus_player, focus_team)
    return session.get_session_stats()


def update_game_state(session_id: str, state_update: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Update game state and get triggered alerts.

    Args:
        session_id: Session ID
        state_update: Game state update

    Returns:
        List of triggered alerts
    """
    manager = get_manager()
    session = manager.get_session(session_id)

    if not session:
        return []

    alerts = session.update_state(state_update)
    return [a.to_dict() for a in alerts]


def get_session_info(session_id: str) -> dict[str, Any] | None:
    """Get session information."""
    manager = get_manager()
    session = manager.get_session(session_id)
    return session.get_session_stats() if session else None


def end_coaching_session(session_id: str) -> dict[str, Any] | None:
    """End a coaching session."""
    return get_manager().end_session(session_id)


def format_alert_for_voice(alert_data: dict[str, Any]) -> str:
    """Format an alert for voice output."""
    alert = CoachingAlert(
        alert_id=alert_data.get("alert_id", ""),
        alert_type=AlertType(alert_data.get("alert_type", "info")),
        priority=AlertPriority(alert_data.get("priority", 5)),
        message=alert_data.get("message", ""),
        details=alert_data.get("details", ""),
    )
    return VoiceAlertFormatter.format_for_voice(alert)
