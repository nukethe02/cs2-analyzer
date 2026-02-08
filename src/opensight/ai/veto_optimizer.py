"""
Map veto optimizer for CS2 best-of-1 and best-of-3 formats.

Combines map win rates with recency weighting and opponent data
to generate optimal ban/pick sequences.

Entirely heuristic — no LLM needed. Pure computation.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

ACTIVE_MAP_POOL = [
    "de_mirage",
    "de_inferno",
    "de_nuke",
    "de_ancient",
    "de_anubis",
    "de_dust2",
    "de_vertigo",
]


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class MapStrength:
    """Team's performance on a specific map."""

    map_name: str
    matches_played: int = 0
    win_rate: float = 0.5  # 0.0-1.0
    ct_win_rate: float = 0.5
    t_win_rate: float = 0.5
    avg_rounds_won: float = 0.0
    recency_weighted_win_rate: float = 0.5
    confidence: float = 0.0  # higher with more matches played

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "map_name": self.map_name,
            "matches_played": self.matches_played,
            "win_rate": round(self.win_rate, 3),
            "ct_win_rate": round(self.ct_win_rate, 3),
            "t_win_rate": round(self.t_win_rate, 3),
            "avg_rounds_won": round(self.avg_rounds_won, 1),
            "recency_weighted_win_rate": round(self.recency_weighted_win_rate, 3),
            "confidence": round(self.confidence, 3),
        }


@dataclass
class VetoRecommendation:
    """Recommended ban/pick for one step of the veto."""

    action: str  # "ban" or "pick"
    map_name: str
    reason: str
    your_win_rate: float = 0.5
    opponent_win_rate: float = 0.5
    net_advantage: float = 0.0  # your WR - opponent WR

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "action": self.action,
            "map_name": self.map_name,
            "reason": self.reason,
            "your_win_rate": round(self.your_win_rate, 3),
            "opponent_win_rate": round(self.opponent_win_rate, 3),
            "net_advantage": round(self.net_advantage, 3),
        }


@dataclass
class VetoAnalysis:
    """Complete veto analysis for a matchup."""

    your_map_pool: list[MapStrength] = field(default_factory=list)
    opponent_map_pool: list[MapStrength] = field(default_factory=list)
    recommended_veto_sequence: list[VetoRecommendation] = field(default_factory=list)
    best_map_for_you: str = ""
    worst_map_for_you: str = ""
    predicted_decider_map: str = ""
    confidence: str = "low"
    format: str = "bo1"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "your_map_pool": [ms.to_dict() for ms in self.your_map_pool],
            "opponent_map_pool": [ms.to_dict() for ms in self.opponent_map_pool],
            "recommended_veto_sequence": [
                vr.to_dict() for vr in self.recommended_veto_sequence
            ],
            "best_map_for_you": self.best_map_for_you,
            "worst_map_for_you": self.worst_map_for_you,
            "predicted_decider_map": self.predicted_decider_map,
            "confidence": self.confidence,
            "format": self.format,
        }


# =============================================================================
# VetoOptimizer
# =============================================================================


class VetoOptimizer:
    """
    Generates optimal map veto sequences for CS2 matches.

    Supports BO1 (6 bans, 1 remaining) and BO3 (2 bans, 2 picks, 2 bans, 1 decider).
    """

    def analyze(
        self,
        your_demos: list[dict],
        opponent_data: list[dict] | dict,
        format: str = "bo1",
    ) -> VetoAnalysis:
        """
        Generate veto recommendations for an upcoming match.

        Args:
            your_demos: List of orchestrator results from your team's matches
            opponent_data: Either a list of orchestrator results OR a scouting
                dict (TeamScoutReport.to_dict() format)
            format: "bo1" or "bo3"

        Returns:
            VetoAnalysis with map strengths and recommended veto sequence
        """
        # Compute your map strengths
        your_maps = self._compute_map_strengths(your_demos)

        # Compute opponent map strengths
        if isinstance(opponent_data, dict):
            opp_maps = self._map_strengths_from_scouting(opponent_data)
        else:
            opp_maps = self._compute_map_strengths(opponent_data)

        # Fill in any missing maps from the active pool with defaults
        your_maps = self._fill_pool(your_maps)
        opp_maps = self._fill_pool(opp_maps)

        # Sort by win rate descending for readability
        your_maps.sort(key=lambda m: m.recency_weighted_win_rate, reverse=True)
        opp_maps.sort(key=lambda m: m.recency_weighted_win_rate, reverse=True)

        # Generate veto sequence
        if format == "bo3":
            sequence = self._generate_bo3_veto(your_maps, opp_maps)
        else:
            sequence = self._generate_bo1_veto(your_maps, opp_maps)

        # Determine best/worst maps
        best = max(your_maps, key=lambda m: m.recency_weighted_win_rate)
        worst = min(your_maps, key=lambda m: m.recency_weighted_win_rate)

        # Predict decider (last map in sequence or the non-banned map)
        decider = self._predict_decider(your_maps, opp_maps, format)

        # Overall confidence
        total_your_matches = sum(m.matches_played for m in your_maps)
        total_opp_matches = sum(m.matches_played for m in opp_maps)
        if total_your_matches >= 10 and total_opp_matches >= 10:
            confidence = "high"
        elif total_your_matches >= 5 or total_opp_matches >= 5:
            confidence = "medium"
        else:
            confidence = "low"

        return VetoAnalysis(
            your_map_pool=your_maps,
            opponent_map_pool=opp_maps,
            recommended_veto_sequence=sequence,
            best_map_for_you=best.map_name,
            worst_map_for_you=worst.map_name,
            predicted_decider_map=decider,
            confidence=confidence,
            format=format,
        )

    # =========================================================================
    # Map Strength Computation
    # =========================================================================

    def _compute_map_strengths(self, demos: list[dict]) -> list[MapStrength]:
        """
        Compute win rates per map from orchestrator result data.

        Assumes team1 (CT first half, score_ct) perspective.
        """
        # Group demos by map
        map_demos: dict[str, list[dict]] = {}
        for demo in demos:
            demo_info = demo.get("demo_info") or {}
            map_name = demo_info.get("map", "")
            if not map_name:
                continue
            map_demos.setdefault(map_name, []).append(demo)

        strengths: list[MapStrength] = []

        for map_name, demos_list in map_demos.items():
            wins = 0
            ct_round_wins = 0
            ct_round_total = 0
            t_round_wins = 0
            t_round_total = 0
            total_rounds_won = 0

            for demo in demos_list:
                demo_info = demo.get("demo_info") or {}
                score_ct = demo_info.get("score_ct", 0)
                score_t = demo_info.get("score_t", 0)

                # Team1 (CT first half) is "our" team
                total_rounds_won += score_ct
                if score_ct > score_t:
                    wins += 1

                # CT/T side split from round_timeline
                timeline = demo.get("round_timeline") or []
                for rdata in timeline:
                    winner = rdata.get("winner", "")
                    round_num = rdata.get("round_num", 0)
                    is_first_half = round_num <= 12

                    if is_first_half:
                        # Our team is CT in first half
                        ct_round_total += 1
                        if winner == "CT":
                            ct_round_wins += 1
                    else:
                        # Our team is T in second half
                        t_round_total += 1
                        if winner == "T":
                            t_round_wins += 1

            n = len(demos_list)
            win_rate = wins / n if n > 0 else 0.5
            ct_wr = ct_round_wins / ct_round_total if ct_round_total > 0 else 0.5
            t_wr = t_round_wins / t_round_total if t_round_total > 0 else 0.5
            avg_rounds = total_rounds_won / n if n > 0 else 0.0

            recency_wr = self._recency_weight(demos_list)
            confidence = 1 - math.exp(-n / 3)

            strengths.append(
                MapStrength(
                    map_name=map_name,
                    matches_played=n,
                    win_rate=win_rate,
                    ct_win_rate=ct_wr,
                    t_win_rate=t_wr,
                    avg_rounds_won=avg_rounds,
                    recency_weighted_win_rate=recency_wr,
                    confidence=confidence,
                )
            )

        return strengths

    def _map_strengths_from_scouting(self, scouting: dict) -> list[MapStrength]:
        """
        Estimate map strengths from scouting data (TeamScoutReport.to_dict()).

        Since scouting data doesn't contain explicit win rates, we estimate
        based on which maps the opponent plays on (maps in their map_tendencies
        are assumed to be their stronger maps).
        """
        map_tendencies = scouting.get("map_tendencies") or []

        # Demos analyzed gives us confidence
        demos_analyzed = scouting.get("demos_analyzed", 0)
        base_confidence = 1 - math.exp(-demos_analyzed / 3) if demos_analyzed > 0 else 0.0

        strengths: list[MapStrength] = []
        for mt in map_tendencies:
            map_name = mt.get("map_name", "")
            if not map_name:
                continue

            # Estimate strength from tendencies
            t_side = mt.get("t_side") or {}
            ct_side = mt.get("ct_side") or {}

            t_aggression = t_side.get("aggression", 50)
            ct_aggression = ct_side.get("aggression", 50)

            # Higher aggression on T-side + low CT aggression = T-sided team
            # Estimate a moderate win rate for maps they play on
            estimated_wr = 0.55  # They chose to play this map

            strengths.append(
                MapStrength(
                    map_name=map_name,
                    matches_played=demos_analyzed,
                    win_rate=estimated_wr,
                    ct_win_rate=0.5 + (ct_aggression - 50) * 0.002,
                    t_win_rate=0.5 + (t_aggression - 50) * 0.002,
                    avg_rounds_won=12.0,  # approximate
                    recency_weighted_win_rate=estimated_wr,
                    confidence=base_confidence * 0.5,  # lower confidence from scouting
                )
            )

        return strengths

    def _fill_pool(self, strengths: list[MapStrength]) -> list[MapStrength]:
        """
        Ensure all active pool maps are represented.
        Missing maps get 50% win rate with 0 confidence.
        """
        existing = {ms.map_name for ms in strengths}
        for map_name in ACTIVE_MAP_POOL:
            if map_name not in existing:
                strengths.append(MapStrength(map_name=map_name))
        return strengths

    def _recency_weight(
        self, demos: list[dict], half_life: int = 5
    ) -> float:
        """
        Apply exponential decay weighting — recent matches matter more.

        Args:
            demos: List of demo dicts (assumed roughly chronological)
            half_life: Number of matches for weight to halve

        Returns:
            Recency-weighted win rate (0.0-1.0)
        """
        if not demos:
            return 0.5

        total_weight = 0.0
        weighted_wins = 0.0

        # Iterate from most recent (end of list) to oldest
        for i, demo in enumerate(reversed(demos)):
            weight = math.exp(-i / half_life)
            total_weight += weight

            demo_info = demo.get("demo_info") or {}
            score_ct = demo_info.get("score_ct", 0)
            score_t = demo_info.get("score_t", 0)

            if score_ct > score_t:
                weighted_wins += weight

        return weighted_wins / total_weight if total_weight > 0 else 0.5

    # =========================================================================
    # BO1 Veto Generation
    # =========================================================================

    def _generate_bo1_veto(
        self,
        your_maps: list[MapStrength],
        opp_maps: list[MapStrength],
    ) -> list[VetoRecommendation]:
        """
        BO1 veto: each team bans 3 maps, remaining map is played.

        Standard CS2 sequence: A ban, B ban, A ban, B ban, A ban, B ban, remaining.
        Strategy: ban opponent's best maps that give them the biggest advantage.
        """
        your_lookup = {ms.map_name: ms for ms in your_maps}
        opp_lookup = {ms.map_name: ms for ms in opp_maps}

        remaining = set(ACTIVE_MAP_POOL)
        sequence: list[VetoRecommendation] = []

        for step in range(6):
            is_our_ban = step % 2 == 0  # We ban on even steps

            if is_our_ban:
                # Our ban: ban the map where opponent has biggest advantage
                best_ban = self._pick_ban_target(
                    remaining, your_lookup, opp_lookup, perspective="ours"
                )
            else:
                # Simulate opponent ban: they'd ban our best map
                best_ban = self._pick_ban_target(
                    remaining, your_lookup, opp_lookup, perspective="theirs"
                )

            if best_ban is None:
                break

            your_ms = your_lookup.get(best_ban, MapStrength(map_name=best_ban))
            opp_ms = opp_lookup.get(best_ban, MapStrength(map_name=best_ban))
            net = your_ms.recency_weighted_win_rate - opp_ms.recency_weighted_win_rate

            if is_our_ban:
                reason = (
                    f"Ban {best_ban} — opponent "
                    f"{opp_ms.recency_weighted_win_rate:.0%} WR"
                    f" ({opp_ms.matches_played} matches)"
                )
            else:
                reason = (
                    f"Opponent bans {best_ban} — your "
                    f"{your_ms.recency_weighted_win_rate:.0%} WR"
                    f" ({your_ms.matches_played} matches)"
                )

            sequence.append(
                VetoRecommendation(
                    action="ban",
                    map_name=best_ban,
                    reason=reason,
                    your_win_rate=your_ms.recency_weighted_win_rate,
                    opponent_win_rate=opp_ms.recency_weighted_win_rate,
                    net_advantage=net,
                )
            )
            remaining.discard(best_ban)

        # Remaining map is the decider
        if remaining:
            decider = remaining.pop()
            your_ms = your_lookup.get(decider, MapStrength(map_name=decider))
            opp_ms = opp_lookup.get(decider, MapStrength(map_name=decider))
            net = your_ms.recency_weighted_win_rate - opp_ms.recency_weighted_win_rate

            sequence.append(
                VetoRecommendation(
                    action="pick",
                    map_name=decider,
                    reason=f"Remaining map — your {your_ms.recency_weighted_win_rate:.0%} "
                    f"vs their {opp_ms.recency_weighted_win_rate:.0%}",
                    your_win_rate=your_ms.recency_weighted_win_rate,
                    opponent_win_rate=opp_ms.recency_weighted_win_rate,
                    net_advantage=net,
                )
            )

        return sequence

    # =========================================================================
    # BO3 Veto Generation
    # =========================================================================

    def _generate_bo3_veto(
        self,
        your_maps: list[MapStrength],
        opp_maps: list[MapStrength],
    ) -> list[VetoRecommendation]:
        """
        BO3 veto: ban-ban-pick-pick-ban-ban-decider.

        Standard CS2 sequence:
          1. Team A ban
          2. Team B ban
          3. Team A pick
          4. Team B pick
          5. Team A ban
          6. Team B ban
          7. Remaining map is decider
        """
        your_lookup = {ms.map_name: ms for ms in your_maps}
        opp_lookup = {ms.map_name: ms for ms in opp_maps}

        remaining = set(ACTIVE_MAP_POOL)
        sequence: list[VetoRecommendation] = []

        # Step 1: Our ban — ban opponent's best map
        target = self._pick_ban_target(remaining, your_lookup, opp_lookup, "ours")
        if target:
            self._add_step(sequence, "ban", target, your_lookup, opp_lookup,
                           "Ban opponent's strongest map")
            remaining.discard(target)

        # Step 2: Opponent ban — they ban our best map
        target = self._pick_ban_target(remaining, your_lookup, opp_lookup, "theirs")
        if target:
            self._add_step(sequence, "ban", target, your_lookup, opp_lookup,
                           "Opponent bans your strongest map")
            remaining.discard(target)

        # Step 3: Our pick — pick our best remaining map
        target = self._pick_best_map(remaining, your_lookup, opp_lookup, "ours")
        if target:
            self._add_step(sequence, "pick", target, your_lookup, opp_lookup,
                           "Pick your strongest remaining map")
            remaining.discard(target)

        # Step 4: Opponent pick — they pick their best remaining map
        target = self._pick_best_map(remaining, your_lookup, opp_lookup, "theirs")
        if target:
            self._add_step(sequence, "pick", target, your_lookup, opp_lookup,
                           "Opponent picks their strongest remaining map")
            remaining.discard(target)

        # Step 5: Our ban
        target = self._pick_ban_target(remaining, your_lookup, opp_lookup, "ours")
        if target:
            self._add_step(sequence, "ban", target, your_lookup, opp_lookup,
                           "Ban opponent's best remaining")
            remaining.discard(target)

        # Step 6: Opponent ban
        target = self._pick_ban_target(remaining, your_lookup, opp_lookup, "theirs")
        if target:
            self._add_step(sequence, "ban", target, your_lookup, opp_lookup,
                           "Opponent bans your best remaining")
            remaining.discard(target)

        # Step 7: Remaining map is decider
        if remaining:
            decider = remaining.pop()
            self._add_step(sequence, "pick", decider, your_lookup, opp_lookup,
                           "Decider map")

        return sequence

    # =========================================================================
    # Veto Helpers
    # =========================================================================

    def _pick_ban_target(
        self,
        remaining: set[str],
        your_lookup: dict[str, MapStrength],
        opp_lookup: dict[str, MapStrength],
        perspective: str,
    ) -> str | None:
        """
        Pick the best map to ban from remaining pool.

        perspective="ours": ban the map where opponent has biggest advantage
        perspective="theirs": simulate opponent banning our best map
        """
        if not remaining:
            return None

        if perspective == "ours":
            # Ban the map where opponent advantage is biggest
            # i.e., opp_wr - your_wr is maximized
            return max(
                remaining,
                key=lambda m: (
                    opp_lookup.get(m, MapStrength(map_name=m)).recency_weighted_win_rate
                    - your_lookup.get(m, MapStrength(map_name=m)).recency_weighted_win_rate
                ),
            )
        else:
            # Opponent bans our best map
            # i.e., your_wr - opp_wr is maximized
            return max(
                remaining,
                key=lambda m: (
                    your_lookup.get(m, MapStrength(map_name=m)).recency_weighted_win_rate
                    - opp_lookup.get(m, MapStrength(map_name=m)).recency_weighted_win_rate
                ),
            )

    def _pick_best_map(
        self,
        remaining: set[str],
        your_lookup: dict[str, MapStrength],
        opp_lookup: dict[str, MapStrength],
        perspective: str,
    ) -> str | None:
        """
        Pick the best map to play from remaining pool.

        perspective="ours": pick map with highest net advantage for us
        perspective="theirs": pick map with highest net advantage for opponent
        """
        if not remaining:
            return None

        if perspective == "ours":
            return max(
                remaining,
                key=lambda m: (
                    your_lookup.get(m, MapStrength(map_name=m)).recency_weighted_win_rate
                    - opp_lookup.get(m, MapStrength(map_name=m)).recency_weighted_win_rate
                ),
            )
        else:
            return max(
                remaining,
                key=lambda m: (
                    opp_lookup.get(m, MapStrength(map_name=m)).recency_weighted_win_rate
                    - your_lookup.get(m, MapStrength(map_name=m)).recency_weighted_win_rate
                ),
            )

    def _add_step(
        self,
        sequence: list[VetoRecommendation],
        action: str,
        map_name: str,
        your_lookup: dict[str, MapStrength],
        opp_lookup: dict[str, MapStrength],
        reason_prefix: str,
    ) -> None:
        """Add a veto step to the sequence."""
        your_ms = your_lookup.get(map_name, MapStrength(map_name=map_name))
        opp_ms = opp_lookup.get(map_name, MapStrength(map_name=map_name))
        net = your_ms.recency_weighted_win_rate - opp_ms.recency_weighted_win_rate

        reason = (
            f"{reason_prefix}: {map_name} — "
            f"your {your_ms.recency_weighted_win_rate:.0%} vs "
            f"their {opp_ms.recency_weighted_win_rate:.0%}"
        )

        sequence.append(
            VetoRecommendation(
                action=action,
                map_name=map_name,
                reason=reason,
                your_win_rate=your_ms.recency_weighted_win_rate,
                opponent_win_rate=opp_ms.recency_weighted_win_rate,
                net_advantage=net,
            )
        )

    def _predict_decider(
        self,
        your_maps: list[MapStrength],
        opp_maps: list[MapStrength],
        format: str,
    ) -> str:
        """
        Predict which map will be the decider after optimal vetoes.
        """
        your_lookup = {ms.map_name: ms for ms in your_maps}
        opp_lookup = {ms.map_name: ms for ms in opp_maps}

        remaining = set(ACTIVE_MAP_POOL)

        if format == "bo3":
            # Simulate: ban-ban-pick-pick-ban-ban-decider
            steps = [
                ("ours", "ban"),
                ("theirs", "ban"),
                ("ours", "pick"),
                ("theirs", "pick"),
                ("ours", "ban"),
                ("theirs", "ban"),
            ]
        else:
            # BO1: 3 bans each
            steps = [
                ("ours", "ban"),
                ("theirs", "ban"),
                ("ours", "ban"),
                ("theirs", "ban"),
                ("ours", "ban"),
                ("theirs", "ban"),
            ]

        for perspective, action in steps:
            if not remaining:
                break
            if action == "ban":
                target = self._pick_ban_target(
                    remaining, your_lookup, opp_lookup, perspective
                )
            else:
                target = self._pick_best_map(
                    remaining, your_lookup, opp_lookup, perspective
                )
            if target:
                remaining.discard(target)

        if remaining:
            return remaining.pop()
        return ""


# =============================================================================
# Module-level convenience
# =============================================================================

_optimizer_instance: VetoOptimizer | None = None


def get_veto_optimizer() -> VetoOptimizer:
    """Get or create singleton VetoOptimizer instance."""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = VetoOptimizer()
    return _optimizer_instance
