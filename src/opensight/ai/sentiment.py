"""
Sentiment Analysis Module for CS2 Demo Analyzer.

Analyzes in-game voice communications (if available) to gauge
team morale and communication effectiveness using NLP techniques.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

# ============================================================================
# Sentiment Types and Definitions
# ============================================================================


class SentimentType(Enum):
    """Types of sentiment classifications."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    TOXIC = "toxic"
    ENCOURAGING = "encouraging"
    FRUSTRATED = "frustrated"
    CALM = "calm"
    EXCITED = "excited"


class CommunicationType(Enum):
    """Types of in-game communication."""

    CALLOUT = "callout"
    STRATEGY = "strategy"
    ENCOURAGEMENT = "encouragement"
    CRITICISM = "criticism"
    FRUSTRATION = "frustration"
    CELEBRATION = "celebration"
    QUESTION = "question"
    RESPONSE = "response"
    SILENCE = "silence"


class MoraleLevel(Enum):
    """Team morale levels."""

    EXCELLENT = "excellent"
    GOOD = "good"
    NEUTRAL = "neutral"
    POOR = "poor"
    CRITICAL = "critical"


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class VoiceMessage:
    """A single voice communication instance."""

    message_id: str
    steamid: str
    player_name: str
    text: str  # Transcribed text
    timestamp: float
    round_num: int
    duration_ms: float

    # Analysis results
    sentiment: SentimentType = SentimentType.NEUTRAL
    sentiment_score: float = 0.0  # -1 to 1
    comm_type: CommunicationType = CommunicationType.CALLOUT
    contains_callout: bool = False
    is_constructive: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "steamid": self.steamid,
            "player_name": self.player_name,
            "text": self.text,
            "timestamp": self.timestamp,
            "round_num": self.round_num,
            "duration_ms": self.duration_ms,
            "sentiment": self.sentiment.value,
            "sentiment_score": round(self.sentiment_score, 3),
            "comm_type": self.comm_type.value,
            "contains_callout": self.contains_callout,
            "is_constructive": self.is_constructive,
        }


@dataclass
class PlayerCommunicationStats:
    """Communication statistics for a player."""

    steamid: str
    player_name: str

    # Message counts
    total_messages: int = 0
    callouts_made: int = 0
    strategy_calls: int = 0
    encouragements: int = 0
    criticisms: int = 0

    # Sentiment scores
    avg_sentiment: float = 0.0
    positive_rate: float = 0.0
    negative_rate: float = 0.0

    # Communication quality
    talk_time_seconds: float = 0.0
    words_per_minute: float = 0.0
    callout_accuracy: float = 0.0  # If we can verify callouts
    response_rate: float = 0.0  # How often they respond to comms

    # Flags
    is_tilted: bool = False
    is_leader: bool = False
    needs_support: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "steamid": self.steamid,
            "player_name": self.player_name,
            "message_counts": {
                "total": self.total_messages,
                "callouts": self.callouts_made,
                "strategy": self.strategy_calls,
                "encouragements": self.encouragements,
                "criticisms": self.criticisms,
            },
            "sentiment": {
                "average": round(self.avg_sentiment, 3),
                "positive_rate": round(self.positive_rate, 2),
                "negative_rate": round(self.negative_rate, 2),
            },
            "quality": {
                "talk_time_seconds": round(self.talk_time_seconds, 1),
                "words_per_minute": round(self.words_per_minute, 1),
                "callout_accuracy": round(self.callout_accuracy, 2),
                "response_rate": round(self.response_rate, 2),
            },
            "flags": {
                "is_tilted": self.is_tilted,
                "is_leader": self.is_leader,
                "needs_support": self.needs_support,
            },
        }


@dataclass
class TeamMoraleReport:
    """Overall team morale and communication report."""

    team_name: str
    demo_id: str
    map_name: str
    timestamp: str

    # Overall morale
    overall_morale: MoraleLevel = MoraleLevel.NEUTRAL
    morale_score: float = 0.5  # 0 to 1

    # Communication metrics
    total_comms: int = 0
    callouts_per_round: float = 0.0
    avg_response_time_ms: float = 0.0
    communication_gaps: list[dict[str, Any]] = field(default_factory=list)

    # Sentiment breakdown
    positive_percentage: float = 0.0
    negative_percentage: float = 0.0
    toxic_incidents: int = 0

    # Team dynamics
    dominant_communicator: str | None = None
    quiet_players: list[str] = field(default_factory=list)
    tension_moments: list[dict[str, Any]] = field(default_factory=list)

    # Per-player stats
    player_stats: list[PlayerCommunicationStats] = field(default_factory=list)

    # Round-by-round morale
    morale_timeline: list[dict[str, Any]] = field(default_factory=list)

    # Recommendations
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "team_name": self.team_name,
            "demo_id": self.demo_id,
            "map_name": self.map_name,
            "timestamp": self.timestamp,
            "overall": {"morale": self.overall_morale.value, "score": round(self.morale_score, 2)},
            "communication": {
                "total_comms": self.total_comms,
                "callouts_per_round": round(self.callouts_per_round, 2),
                "avg_response_time_ms": round(self.avg_response_time_ms, 1),
                "communication_gaps": self.communication_gaps,
            },
            "sentiment": {
                "positive_percentage": round(self.positive_percentage, 1),
                "negative_percentage": round(self.negative_percentage, 1),
                "toxic_incidents": self.toxic_incidents,
            },
            "dynamics": {
                "dominant_communicator": self.dominant_communicator,
                "quiet_players": self.quiet_players,
                "tension_moments": self.tension_moments,
            },
            "player_stats": [p.to_dict() for p in self.player_stats],
            "morale_timeline": self.morale_timeline,
            "recommendations": self.recommendations,
        }


# ============================================================================
# Sentiment Analysis Engine
# ============================================================================


class SentimentAnalyzer:
    """
    Analyzes sentiment from voice comm transcripts.
    Uses rule-based analysis with CS2-specific vocabulary.
    """

    # CS2-specific positive words/phrases
    POSITIVE_PATTERNS = [
        r"\b(nice|good|great|amazing|insane|sick|clean|nutty|huge)\b",
        r"\b(well played|wp|good job|gj|let's go|lets go|gg|good shit)\b",
        r"\b(we got this|we can do this|believe|clutch|trust)\b",
        r"\b(comeback|momentum|confident|easy|ez)\b",
    ]

    # CS2-specific negative words/phrases
    NEGATIVE_PATTERNS = [
        r"\b(bad|terrible|awful|trash|garbage|stupid|dumb|idiot)\b",
        r"\b(wtf|what the fuck|are you serious|seriously|really)\b",
        r"\b(useless|no help|where were you|why didn't you)\b",
        r"\b(fuck|shit|damn|crap|hate)\b",
        r"\b(lost|done|over|gg go next|ff|forfeit)\b",
    ]

    # Toxic patterns (severe negative)
    TOXIC_PATTERNS = [
        r"\b(kys|kill yourself|uninstall|delete game)\b",
        r"\b(retard|retarded|braindead|bot|boosted)\b",
        r"\b(throwing|thrower|troll|trolling)\b",
        r"(stfu|shut up|shut the fuck up|muted)",
        r"\b(reported|reporting|ban|banned)\b",
    ]

    # Encouraging patterns
    ENCOURAGEMENT_PATTERNS = [
        r"\b(you got this|believe|try again|next round)\b",
        r"\b(it's okay|its okay|no worries|all good|no problem)\b",
        r"\b(unlucky|close|almost|nearly)\b",
        r"\b(focus|reset|fresh round|let's reset)\b",
    ]

    # Frustrated patterns
    FRUSTRATION_PATTERNS = [
        r"^(ugh|argh|omg|bruh)\b",
        r"\b(every time|always|never|again)\b",
        r"[!?]{2,}",
        r"\b(how|why|what)\b.*[!?]{2,}",
    ]

    # CS2 callout patterns (location-based)
    CALLOUT_PATTERNS = [
        # General locations
        r"\b(long|short|mid|cat|catwalk|connector|heaven|hell|pit|ramp|stairs)\b",
        # Dust2
        r"\b(tunnels|t spawn|ct spawn|xbox|car|barrels|goose|ninja|elevator)\b",
        # Mirage
        r"\b(palace|apps|apartments|window|jungle|connector|bench|ticket|van)\b",
        # Inferno
        r"\b(banana|library|graveyard|pit|balcony|boiler|dark|sandbags|coffins)\b",
        # Numbers
        r"\b(one|two|three|four|five|1|2|3|4|5)\s*(on|at|in)\b",
        # Enemy counts
        r"\b(pushing|rotating|flanking|peeking|holding)\b",
    ]

    # Strategy patterns
    STRATEGY_PATTERNS = [
        r"\b(let's|lets)\s+(go|push|rush|exec|execute|take|smoke|flash)\b",
        r"\b(wait|hold|save|eco|force|buy|drop)\b",
        r"\b(A|B)\s+(site|split|exec|execute|take)\b",
        r"\b(default|spread|stack|play for picks)\b",
        r"\b(rotate|retake|fall back|anchor)\b",
    ]

    def __init__(self):
        # Compile patterns for efficiency
        self.positive_re = [re.compile(p, re.IGNORECASE) for p in self.POSITIVE_PATTERNS]
        self.negative_re = [re.compile(p, re.IGNORECASE) for p in self.NEGATIVE_PATTERNS]
        self.toxic_re = [re.compile(p, re.IGNORECASE) for p in self.TOXIC_PATTERNS]
        self.encouragement_re = [re.compile(p, re.IGNORECASE) for p in self.ENCOURAGEMENT_PATTERNS]
        self.frustration_re = [re.compile(p, re.IGNORECASE) for p in self.FRUSTRATION_PATTERNS]
        self.callout_re = [re.compile(p, re.IGNORECASE) for p in self.CALLOUT_PATTERNS]
        self.strategy_re = [re.compile(p, re.IGNORECASE) for p in self.STRATEGY_PATTERNS]

    def analyze_message(self, message: VoiceMessage) -> VoiceMessage:
        """
        Analyze a single voice message for sentiment.

        Args:
            message: Voice message to analyze

        Returns:
            Analyzed message with sentiment data
        """
        text = message.text.lower()

        # Count pattern matches
        positive_score = sum(len(p.findall(text)) for p in self.positive_re)
        negative_score = sum(len(p.findall(text)) for p in self.negative_re)
        toxic_score = sum(len(p.findall(text)) for p in self.toxic_re)
        encouragement_score = sum(len(p.findall(text)) for p in self.encouragement_re)
        frustration_score = sum(len(p.findall(text)) for p in self.frustration_re)
        callout_score = sum(len(p.findall(text)) for p in self.callout_re)
        strategy_score = sum(len(p.findall(text)) for p in self.strategy_re)

        # Calculate overall sentiment (-1 to 1)
        positive_total = positive_score + encouragement_score
        negative_total = negative_score + toxic_score * 2 + frustration_score

        if positive_total + negative_total > 0:
            message.sentiment_score = (positive_total - negative_total) / (
                positive_total + negative_total
            )
        else:
            message.sentiment_score = 0.0

        # Determine sentiment type
        if toxic_score > 0:
            message.sentiment = SentimentType.TOXIC
        elif message.sentiment_score > 0.3:
            if encouragement_score > positive_score:
                message.sentiment = SentimentType.ENCOURAGING
            else:
                message.sentiment = SentimentType.POSITIVE
        elif message.sentiment_score < -0.3:
            if frustration_score > negative_score:
                message.sentiment = SentimentType.FRUSTRATED
            else:
                message.sentiment = SentimentType.NEGATIVE
        else:
            message.sentiment = SentimentType.NEUTRAL

        # Determine communication type
        if callout_score >= 2:
            message.comm_type = CommunicationType.CALLOUT
            message.contains_callout = True
        elif strategy_score >= 1:
            message.comm_type = CommunicationType.STRATEGY
        elif encouragement_score >= 1:
            message.comm_type = CommunicationType.ENCOURAGEMENT
        elif negative_score >= 2 or toxic_score >= 1:
            message.comm_type = CommunicationType.CRITICISM
        elif frustration_score >= 1:
            message.comm_type = CommunicationType.FRUSTRATION
        elif positive_score >= 2:
            message.comm_type = CommunicationType.CELEBRATION
        elif "?" in text:
            message.comm_type = CommunicationType.QUESTION
        else:
            message.comm_type = CommunicationType.RESPONSE

        # Determine if constructive
        message.is_constructive = (
            callout_score > 0
            or strategy_score > 0
            or encouragement_score > 0
            or (positive_score > negative_score and toxic_score == 0)
        )

        return message


# ============================================================================
# Communication Analysis Engine
# ============================================================================


class CommunicationAnalyzer:
    """
    Analyzes team communication patterns and effectiveness.
    """

    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()

    def analyze_team_comms(
        self, messages: list[dict[str, Any]], team_steamids: list[str], demo_data: dict[str, Any]
    ) -> TeamMoraleReport:
        """
        Analyze team communications from a demo.

        Args:
            messages: List of voice message dictionaries
            team_steamids: Steam IDs of team members
            demo_data: Demo metadata

        Returns:
            Complete morale report
        """
        # Filter to team messages
        team_messages = [m for m in messages if m.get("steamid") in team_steamids]

        # Convert to VoiceMessage objects and analyze
        analyzed_messages = []
        for i, msg in enumerate(team_messages):
            voice_msg = VoiceMessage(
                message_id=f"msg_{i}",
                steamid=msg.get("steamid", ""),
                player_name=msg.get("player_name", "Player"),
                text=msg.get("text", ""),
                timestamp=msg.get("timestamp", 0),
                round_num=msg.get("round_num", 0),
                duration_ms=msg.get("duration_ms", 0),
            )
            analyzed = self.sentiment_analyzer.analyze_message(voice_msg)
            analyzed_messages.append(analyzed)

        # Create report
        report = TeamMoraleReport(
            team_name=demo_data.get("team_name", "Unknown Team"),
            demo_id=demo_data.get("demo_id", ""),
            map_name=demo_data.get("map_name", "unknown"),
            timestamp=datetime.now().isoformat(),
        )

        # Calculate metrics
        report.total_comms = len(analyzed_messages)

        if analyzed_messages:
            # Sentiment breakdown
            sentiments = [m.sentiment_score for m in analyzed_messages]
            positive = len([s for s in sentiments if s > 0.2])
            negative = len([s for s in sentiments if s < -0.2])

            report.positive_percentage = (positive / len(sentiments)) * 100
            report.negative_percentage = (negative / len(sentiments)) * 100
            report.toxic_incidents = len(
                [m for m in analyzed_messages if m.sentiment == SentimentType.TOXIC]
            )

            # Callouts per round
            callouts = len([m for m in analyzed_messages if m.contains_callout])
            rounds = demo_data.get("num_rounds", 1)
            report.callouts_per_round = callouts / max(rounds, 1)

        # Calculate player stats
        report.player_stats = self._calculate_player_stats(analyzed_messages, team_steamids)

        # Determine morale level
        report.morale_score = self._calculate_morale_score(report)
        report.overall_morale = self._score_to_morale(report.morale_score)

        # Build timeline
        report.morale_timeline = self._build_morale_timeline(
            analyzed_messages, demo_data.get("num_rounds", 1)
        )

        # Identify dynamics
        self._identify_team_dynamics(report)

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        return report

    def _calculate_player_stats(
        self, messages: list[VoiceMessage], team_steamids: list[str]
    ) -> list[PlayerCommunicationStats]:
        """Calculate per-player communication statistics."""
        stats_by_player: dict[str, PlayerCommunicationStats] = {}

        # Initialize stats for all team members
        for steamid in team_steamids:
            stats_by_player[steamid] = PlayerCommunicationStats(
                steamid=steamid,
                player_name=steamid,  # Will be updated
            )

        # Process messages
        for msg in messages:
            steamid = msg.steamid
            if steamid not in stats_by_player:
                continue

            stats = stats_by_player[steamid]
            stats.player_name = msg.player_name
            stats.total_messages += 1
            stats.talk_time_seconds += msg.duration_ms / 1000

            # Count by type
            if msg.comm_type == CommunicationType.CALLOUT:
                stats.callouts_made += 1
            elif msg.comm_type == CommunicationType.STRATEGY:
                stats.strategy_calls += 1
            elif msg.comm_type == CommunicationType.ENCOURAGEMENT:
                stats.encouragements += 1
            elif msg.comm_type == CommunicationType.CRITICISM:
                stats.criticisms += 1

        # Calculate averages
        for steamid, stats in stats_by_player.items():
            player_messages = [m for m in messages if m.steamid == steamid]

            if player_messages:
                # Average sentiment
                stats.avg_sentiment = sum(m.sentiment_score for m in player_messages) / len(
                    player_messages
                )

                # Sentiment rates
                positive = len([m for m in player_messages if m.sentiment_score > 0.2])
                negative = len([m for m in player_messages if m.sentiment_score < -0.2])
                stats.positive_rate = positive / len(player_messages)
                stats.negative_rate = negative / len(player_messages)

                # Words per minute (rough estimate)
                total_words = sum(len(m.text.split()) for m in player_messages)
                if stats.talk_time_seconds > 0:
                    stats.words_per_minute = (total_words / stats.talk_time_seconds) * 60

            # Set flags
            if stats.avg_sentiment < -0.3 and stats.criticisms > stats.encouragements:
                stats.is_tilted = True

            if stats.strategy_calls >= 3:
                stats.is_leader = True

            if stats.avg_sentiment < -0.2 and stats.total_messages > 3:
                stats.needs_support = True

        return list(stats_by_player.values())

    def _calculate_morale_score(self, report: TeamMoraleReport) -> float:
        """Calculate overall morale score (0 to 1)."""
        score = 0.5  # Start neutral

        # Sentiment impact
        sentiment_diff = report.positive_percentage - report.negative_percentage
        score += sentiment_diff / 200  # Max ±0.25

        # Toxic incidents are severe
        score -= report.toxic_incidents * 0.1

        # Communication volume bonus
        if report.callouts_per_round >= 3:
            score += 0.1
        elif report.callouts_per_round >= 1:
            score += 0.05

        # Player stats impact
        tilted_players = len([p for p in report.player_stats if p.is_tilted])
        score -= tilted_players * 0.1

        leaders = len([p for p in report.player_stats if p.is_leader])
        score += min(leaders * 0.05, 0.1)

        # Clamp to 0-1
        return max(0.0, min(1.0, score))

    def _score_to_morale(self, score: float) -> MoraleLevel:
        """Convert morale score to level."""
        if score >= 0.75:
            return MoraleLevel.EXCELLENT
        elif score >= 0.55:
            return MoraleLevel.GOOD
        elif score >= 0.40:
            return MoraleLevel.NEUTRAL
        elif score >= 0.25:
            return MoraleLevel.POOR
        else:
            return MoraleLevel.CRITICAL

    def _build_morale_timeline(
        self, messages: list[VoiceMessage], num_rounds: int
    ) -> list[dict[str, Any]]:
        """Build round-by-round morale timeline."""
        timeline = []

        for round_num in range(1, num_rounds + 1):
            round_messages = [m for m in messages if m.round_num == round_num]

            if round_messages:
                avg_sentiment = sum(m.sentiment_score for m in round_messages) / len(round_messages)
                positive = len([m for m in round_messages if m.sentiment_score > 0.2])
                negative = len([m for m in round_messages if m.sentiment_score < -0.2])
            else:
                avg_sentiment = 0
                positive = 0
                negative = 0

            timeline.append(
                {
                    "round": round_num,
                    "messages": len(round_messages),
                    "avg_sentiment": round(avg_sentiment, 3),
                    "positive_count": positive,
                    "negative_count": negative,
                    "morale_trend": (
                        "up"
                        if avg_sentiment > 0.1
                        else "down"
                        if avg_sentiment < -0.1
                        else "stable"
                    ),
                }
            )

        return timeline

    def _identify_team_dynamics(self, report: TeamMoraleReport) -> None:
        """Identify team dynamics and tension points."""
        # Find dominant communicator
        if report.player_stats:
            sorted_by_comms = sorted(
                report.player_stats, key=lambda p: p.total_messages, reverse=True
            )
            if sorted_by_comms[0].total_messages > 0:
                report.dominant_communicator = sorted_by_comms[0].player_name

            # Find quiet players
            avg_messages = sum(p.total_messages for p in report.player_stats) / len(
                report.player_stats
            )
            report.quiet_players = [
                p.player_name for p in report.player_stats if p.total_messages < avg_messages * 0.3
            ]

        # Identify tension moments from timeline
        for _i, round_data in enumerate(report.morale_timeline):
            if round_data["avg_sentiment"] < -0.5:
                report.tension_moments.append(
                    {
                        "round": round_data["round"],
                        "severity": "high" if round_data["avg_sentiment"] < -0.7 else "medium",
                        "negative_count": round_data["negative_count"],
                    }
                )

    def _generate_recommendations(self, report: TeamMoraleReport) -> list[str]:
        """Generate coaching recommendations based on analysis."""
        recommendations = []

        # Morale-based recommendations
        if report.overall_morale in [MoraleLevel.POOR, MoraleLevel.CRITICAL]:
            recommendations.append(
                "Team morale is low. Consider taking a break or focusing on positive callouts."
            )

        # Communication volume
        if report.callouts_per_round < 1.5:
            recommendations.append("Callout frequency is low. Encourage more information sharing.")

        # Toxic behavior
        if report.toxic_incidents > 0:
            recommendations.append(
                f"Address {report.toxic_incidents} toxic incident(s). Team atmosphere affects performance."
            )

        # Quiet players
        if report.quiet_players:
            names = ", ".join(report.quiet_players[:3])
            recommendations.append(f"Encourage participation from quieter players: {names}")

        # Tilted players
        tilted = [p.player_name for p in report.player_stats if p.is_tilted]
        if tilted:
            names = ", ".join(tilted[:2])
            recommendations.append(
                f"Players showing frustration: {names}. Offer support or reset mentally."
            )

        # Leadership
        leaders = [p for p in report.player_stats if p.is_leader]
        if not leaders:
            recommendations.append(
                "No clear shot caller identified. Designate an IGL for better coordination."
            )

        # Positive reinforcement gap
        encouragers = [p for p in report.player_stats if p.encouragements >= 2]
        if not encouragers and report.total_comms > 10:
            recommendations.append(
                "Low positive reinforcement. Encourage teammates after good plays."
            )

        return recommendations[:6]  # Limit to 6 recommendations


# ============================================================================
# Voice Transcription Integration (Placeholder)
# ============================================================================


class VoiceTranscriber:
    """
    Placeholder for voice transcription integration.
    In production, this would connect to speech-to-text services.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key
        self.available = False

    def transcribe_audio(self, audio_data: bytes, sample_rate: int = 16000) -> str | None:
        """
        Transcribe audio to text.

        Args:
            audio_data: Raw audio bytes
            sample_rate: Audio sample rate

        Returns:
            Transcribed text or None
        """
        # Placeholder - would integrate with:
        # - OpenAI Whisper
        # - Google Speech-to-Text
        # - Azure Speech Services
        # - Local Whisper model
        return None

    def transcribe_demo_audio(
        self, demo_path: Path, output_path: Path | None = None
    ) -> list[dict[str, Any]]:
        """
        Extract and transcribe all voice comms from a demo.

        Args:
            demo_path: Path to demo file
            output_path: Optional path to save transcription

        Returns:
            List of transcribed messages with timestamps
        """
        # Placeholder - demo audio extraction not directly supported
        # Would need to integrate with demo recording/playback
        return []


# ============================================================================
# Chat Log Analysis (Alternative when voice not available)
# ============================================================================


class ChatLogAnalyzer:
    """
    Analyzes in-game text chat when voice comms aren't available.
    """

    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()

    def analyze_chat_log(
        self, chat_messages: list[dict[str, Any]], team_steamids: list[str]
    ) -> dict[str, Any]:
        """
        Analyze text chat messages.

        Args:
            chat_messages: List of chat message dictionaries
            team_steamids: Steam IDs of team members

        Returns:
            Analysis results
        """
        # Filter team messages
        team_messages = [m for m in chat_messages if m.get("steamid") in team_steamids]

        results = {
            "total_messages": len(team_messages),
            "by_player": {},
            "sentiment_summary": {"positive": 0, "negative": 0, "neutral": 0, "toxic": 0},
            "callouts_found": [],
            "notable_messages": [],
        }

        for msg in team_messages:
            # Create voice message for analysis (reusing sentiment logic)
            voice_msg = VoiceMessage(
                message_id="",
                steamid=msg.get("steamid", ""),
                player_name=msg.get("player_name", "Player"),
                text=msg.get("text", ""),
                timestamp=msg.get("timestamp", 0),
                round_num=msg.get("round_num", 0),
                duration_ms=0,
            )

            analyzed = self.sentiment_analyzer.analyze_message(voice_msg)

            # Update player stats
            steamid = msg.get("steamid", "")
            if steamid not in results["by_player"]:
                results["by_player"][steamid] = {
                    "name": msg.get("player_name", "Player"),
                    "messages": 0,
                    "sentiment_sum": 0,
                }

            results["by_player"][steamid]["messages"] += 1
            results["by_player"][steamid]["sentiment_sum"] += analyzed.sentiment_score

            # Update sentiment counts
            if analyzed.sentiment == SentimentType.TOXIC:
                results["sentiment_summary"]["toxic"] += 1
            elif analyzed.sentiment_score > 0.2:
                results["sentiment_summary"]["positive"] += 1
            elif analyzed.sentiment_score < -0.2:
                results["sentiment_summary"]["negative"] += 1
            else:
                results["sentiment_summary"]["neutral"] += 1

            # Track callouts
            if analyzed.contains_callout:
                results["callouts_found"].append(
                    {
                        "round": msg.get("round_num", 0),
                        "player": msg.get("player_name", "Player"),
                        "text": msg.get("text", ""),
                    }
                )

            # Track notable messages (very positive or negative)
            if abs(analyzed.sentiment_score) > 0.5:
                results["notable_messages"].append(
                    {
                        "round": msg.get("round_num", 0),
                        "player": msg.get("player_name", "Player"),
                        "text": msg.get("text", ""),
                        "sentiment": analyzed.sentiment.value,
                        "score": analyzed.sentiment_score,
                    }
                )

        return results


# ============================================================================
# Convenience Functions
# ============================================================================

_default_analyzer: CommunicationAnalyzer | None = None


def get_analyzer() -> CommunicationAnalyzer:
    """Get or create the default communication analyzer."""
    global _default_analyzer
    if _default_analyzer is None:
        _default_analyzer = CommunicationAnalyzer()
    return _default_analyzer


def analyze_team_morale(
    messages: list[dict[str, Any]], team_steamids: list[str], demo_data: dict[str, Any]
) -> dict[str, Any]:
    """
    Analyze team morale from voice communications.

    Args:
        messages: List of voice message dictionaries
        team_steamids: Steam IDs of team members
        demo_data: Demo metadata

    Returns:
        Morale report dictionary
    """
    analyzer = get_analyzer()
    report = analyzer.analyze_team_comms(messages, team_steamids, demo_data)
    return report.to_dict()


def analyze_single_message(text: str) -> dict[str, Any]:
    """
    Analyze sentiment of a single message.

    Args:
        text: Message text

    Returns:
        Sentiment analysis results
    """
    msg = VoiceMessage(
        message_id="single",
        steamid="",
        player_name="",
        text=text,
        timestamp=0,
        round_num=0,
        duration_ms=0,
    )

    analyzer = SentimentAnalyzer()
    analyzed = analyzer.analyze_message(msg)

    return {
        "text": text,
        "sentiment": analyzed.sentiment.value,
        "score": analyzed.sentiment_score,
        "type": analyzed.comm_type.value,
        "contains_callout": analyzed.contains_callout,
        "is_constructive": analyzed.is_constructive,
    }


def analyze_chat_log(
    chat_messages: list[dict[str, Any]], team_steamids: list[str]
) -> dict[str, Any]:
    """
    Analyze text chat log for sentiment.

    Args:
        chat_messages: List of chat message dictionaries
        team_steamids: Steam IDs of team members

    Returns:
        Chat analysis results
    """
    analyzer = ChatLogAnalyzer()
    return analyzer.analyze_chat_log(chat_messages, team_steamids)
