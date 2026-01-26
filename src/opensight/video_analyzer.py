"""
Video Analysis Tool for CS2 Gameplay

Analyzes gameplay videos to detect mistakes, provide performance feedback,
and suggest strategies for improvement using computer vision and ML algorithms.

Features:
- Crosshair tracking and placement analysis
- Action detection (shooting, movement, utility usage)
- Mistake detection (bad crosshair placement, poor positioning)
- Performance metrics extraction from video
- Personalized coaching feedback

Usage:
    from opensight.video_analyzer import VideoAnalyzer, analyze_gameplay_video

    # Quick analysis
    results = analyze_gameplay_video("gameplay.mp4")

    # Detailed analysis with custom settings
    analyzer = VideoAnalyzer("gameplay.mp4")
    results = analyzer.analyze(
        detect_crosshair=True,
        detect_actions=True,
        generate_coaching=True
    )
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Lazy import flags
_CV2_AVAILABLE: bool | None = None
_SKLEARN_AVAILABLE: bool | None = None


def _check_cv2_available() -> bool:
    """Lazily check if OpenCV is available."""
    global _CV2_AVAILABLE
    if _CV2_AVAILABLE is None:
        try:
            import cv2 as _  # noqa: F401

            _CV2_AVAILABLE = True
        except ImportError:
            _CV2_AVAILABLE = False
    return _CV2_AVAILABLE


def _check_sklearn_available() -> bool:
    """Lazily check if scikit-learn is available."""
    global _SKLEARN_AVAILABLE
    if _SKLEARN_AVAILABLE is None:
        try:
            import sklearn as _  # noqa: F401

            _SKLEARN_AVAILABLE = True
        except ImportError:
            _SKLEARN_AVAILABLE = False
    return _SKLEARN_AVAILABLE


# =============================================================================
# Data Classes
# =============================================================================


class ActionType(Enum):
    """Types of gameplay actions detected in video."""

    IDLE = "idle"
    MOVING = "moving"
    SHOOTING = "shooting"
    RELOADING = "reloading"
    SCOPING = "scoping"
    THROWING_UTILITY = "throwing_utility"
    PLANTING = "planting"
    DEFUSING = "defusing"
    SWITCHING_WEAPON = "switching_weapon"


class MistakeType(Enum):
    """Types of gameplay mistakes."""

    BAD_CROSSHAIR_PLACEMENT = "bad_crosshair_placement"
    POOR_POSITIONING = "poor_positioning"
    SLOW_REACTION = "slow_reaction"
    MISSED_SHOTS = "missed_shots"
    OVER_PEEKING = "over_peeking"
    UNNECESSARY_RELOAD = "unnecessary_reload"
    BAD_UTILITY_USAGE = "bad_utility_usage"
    POOR_MOVEMENT = "poor_movement"
    NOT_CHECKING_CORNERS = "not_checking_corners"
    RUNNING_WHILE_SHOOTING = "running_while_shooting"


class SkillLevel(Enum):
    """Player skill level classification."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    PROFESSIONAL = "professional"


@dataclass
class CrosshairPosition:
    """Crosshair position at a specific frame."""

    frame_num: int
    timestamp_ms: float
    x: int  # Pixel X coordinate
    y: int  # Pixel Y coordinate
    confidence: float  # Detection confidence (0-1)
    screen_region: str = ""  # 'center', 'upper', 'lower', 'left', 'right'


@dataclass
class DetectedAction:
    """A detected gameplay action."""

    action_type: ActionType
    start_frame: int
    end_frame: int
    start_time_ms: float
    end_time_ms: float
    confidence: float
    metadata: dict = field(default_factory=dict)


@dataclass
class DetectedMistake:
    """A detected gameplay mistake."""

    mistake_type: MistakeType
    frame_num: int
    timestamp_ms: float
    severity: str  # 'low', 'medium', 'high'
    description: str
    suggestion: str
    confidence: float


@dataclass
class PerformanceMetrics:
    """Performance metrics extracted from video analysis."""

    # Crosshair metrics
    crosshair_stability: float = 0.0  # 0-100, higher is more stable
    crosshair_placement_score: float = 0.0  # 0-100, head level consistency
    average_crosshair_height: float = 0.0  # Normalized height (0-1)
    crosshair_movement_smoothness: float = 0.0  # 0-100

    # Reaction metrics
    average_reaction_time_ms: float = 0.0
    fastest_reaction_ms: float = 0.0
    slowest_reaction_ms: float = 0.0

    # Action metrics
    shots_detected: int = 0
    movement_efficiency: float = 0.0  # 0-100
    utility_usage_count: int = 0

    # Overall scores
    aim_score: float = 0.0  # 0-100
    movement_score: float = 0.0  # 0-100
    game_sense_score: float = 0.0  # 0-100
    overall_score: float = 0.0  # 0-100

    # Skill classification
    estimated_skill_level: SkillLevel = SkillLevel.INTERMEDIATE


@dataclass
class CoachingTip:
    """A personalized coaching tip."""

    category: str  # 'aim', 'movement', 'positioning', 'utility', 'game_sense'
    priority: int  # 1-5, 1 being highest priority
    title: str
    description: str
    drill_suggestion: str = ""
    related_mistakes: list[MistakeType] = field(default_factory=list)


@dataclass
class VideoAnalysisResult:
    """Complete video analysis results."""

    # Video info
    video_path: str
    duration_seconds: float
    total_frames: int
    fps: float
    resolution: tuple[int, int]

    # Analysis results
    crosshair_positions: list[CrosshairPosition] = field(default_factory=list)
    detected_actions: list[DetectedAction] = field(default_factory=list)
    detected_mistakes: list[DetectedMistake] = field(default_factory=list)
    performance_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    coaching_tips: list[CoachingTip] = field(default_factory=list)

    # Processing info
    analysis_time_seconds: float = 0.0
    frames_analyzed: int = 0
    sample_rate: int = 1  # Every Nth frame analyzed


# =============================================================================
# Crosshair Tracker
# =============================================================================


class CrosshairTracker:
    """
    Tracks crosshair position across video frames.

    Uses computer vision techniques to detect and track the crosshair:
    1. Template matching for common crosshair styles
    2. Color-based detection (green/white crosshairs)
    3. Center-region analysis with edge detection
    """

    # Common crosshair colors (BGR format for OpenCV)
    CROSSHAIR_COLORS = [
        (0, 255, 0),  # Green
        (255, 255, 255),  # White
        (0, 255, 255),  # Yellow
        (0, 0, 255),  # Red
        (255, 0, 255),  # Magenta
    ]

    def __init__(self, frame_width: int, frame_height: int):
        """Initialize crosshair tracker with frame dimensions."""
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.center_x = frame_width // 2
        self.center_y = frame_height // 2

        # Region of interest around center (crosshair should be here)
        self.roi_size = min(frame_width, frame_height) // 8
        self.roi_x1 = self.center_x - self.roi_size
        self.roi_y1 = self.center_y - self.roi_size
        self.roi_x2 = self.center_x + self.roi_size
        self.roi_y2 = self.center_y + self.roi_size

        # Previous positions for smoothing
        self.position_history: list[tuple[int, int]] = []
        self.max_history = 10

    def detect(
        self, frame: np.ndarray, frame_num: int, timestamp_ms: float
    ) -> CrosshairPosition | None:
        """
        Detect crosshair position in a frame.

        Args:
            frame: BGR image as numpy array
            frame_num: Current frame number
            timestamp_ms: Timestamp in milliseconds

        Returns:
            CrosshairPosition if detected, None otherwise
        """
        if not _check_cv2_available():
            # Fallback to center position if OpenCV not available
            return CrosshairPosition(
                frame_num=frame_num,
                timestamp_ms=timestamp_ms,
                x=self.center_x,
                y=self.center_y,
                confidence=0.5,
                screen_region="center",
            )

        # Extract ROI around center
        roi = frame[self.roi_y1 : self.roi_y2, self.roi_x1 : self.roi_x2]

        best_pos = None
        best_confidence = 0.0

        # Try color-based detection for each crosshair color
        for color in self.CROSSHAIR_COLORS:
            pos, conf = self._detect_by_color(roi, color)
            if conf > best_confidence:
                best_confidence = conf
                best_pos = pos

        # Try edge-based detection if color detection fails
        if best_confidence < 0.5:
            pos, conf = self._detect_by_edges(roi)
            if conf > best_confidence:
                best_confidence = conf
                best_pos = pos

        # Default to center if no detection
        if best_pos is None:
            best_pos = (self.roi_size, self.roi_size)
            best_confidence = 0.3

        # Convert ROI coordinates to frame coordinates
        frame_x = self.roi_x1 + best_pos[0]
        frame_y = self.roi_y1 + best_pos[1]

        # Smooth position using history
        frame_x, frame_y = self._smooth_position(frame_x, frame_y)

        # Determine screen region
        screen_region = self._get_screen_region(frame_x, frame_y)

        return CrosshairPosition(
            frame_num=frame_num,
            timestamp_ms=timestamp_ms,
            x=frame_x,
            y=frame_y,
            confidence=best_confidence,
            screen_region=screen_region,
        )

    def _detect_by_color(
        self, roi: np.ndarray, target_color: tuple[int, int, int]
    ) -> tuple[tuple[int, int] | None, float]:
        """Detect crosshair by color matching."""
        import cv2

        # Create color mask with tolerance
        tolerance = 40
        lower = np.array([max(0, c - tolerance) for c in target_color])
        upper = np.array([min(255, c + tolerance) for c in target_color])

        mask = cv2.inRange(roi, lower, upper)

        # Find contours in mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, 0.0

        # Find contour closest to center of ROI
        roi_center = (roi.shape[1] // 2, roi.shape[0] // 2)
        best_contour = None
        best_dist = float("inf")

        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                dist = np.sqrt((cx - roi_center[0]) ** 2 + (cy - roi_center[1]) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_contour = (cx, cy)

        if best_contour is None:
            return None, 0.0

        # Calculate confidence based on distance from center
        max_dist = np.sqrt(roi.shape[0] ** 2 + roi.shape[1] ** 2) / 2
        confidence = 1.0 - (best_dist / max_dist)

        return best_contour, confidence

    def _detect_by_edges(self, roi: np.ndarray) -> tuple[tuple[int, int] | None, float]:
        """Detect crosshair using edge detection (for outline crosshairs)."""
        import cv2

        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Apply Hough line detection to find crosshair lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=10, maxLineGap=5)

        if lines is None or len(lines) < 2:
            return None, 0.0

        # Find intersection point of lines (should be crosshair center)
        roi_center = (roi.shape[1] // 2, roi.shape[0] // 2)

        # Simple approach: find average of line midpoints near center
        midpoints = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            mx, my = (x1 + x2) // 2, (y1 + y2) // 2
            dist = np.sqrt((mx - roi_center[0]) ** 2 + (my - roi_center[1]) ** 2)
            if dist < roi.shape[0] // 4:  # Only consider lines near center
                midpoints.append((mx, my))

        if not midpoints:
            return None, 0.0

        # Average midpoint
        avg_x = int(np.mean([p[0] for p in midpoints]))
        avg_y = int(np.mean([p[1] for p in midpoints]))

        confidence = min(len(midpoints) / 4.0, 1.0) * 0.7  # Cap at 0.7 for edge detection

        return (avg_x, avg_y), confidence

    def _smooth_position(self, x: int, y: int) -> tuple[int, int]:
        """Apply temporal smoothing to reduce jitter."""
        self.position_history.append((x, y))
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)

        if len(self.position_history) < 3:
            return x, y

        # Weighted average (recent positions weighted more)
        weights = np.linspace(0.5, 1.0, len(self.position_history))
        weights = weights / weights.sum()

        smooth_x = int(np.average([p[0] for p in self.position_history], weights=weights))
        smooth_y = int(np.average([p[1] for p in self.position_history], weights=weights))

        return smooth_x, smooth_y

    def _get_screen_region(self, x: int, y: int) -> str:
        """Determine which region of the screen the position is in."""
        # Divide screen into 9 regions
        x_ratio = x / self.frame_width
        y_ratio = y / self.frame_height

        if 0.4 <= x_ratio <= 0.6 and 0.4 <= y_ratio <= 0.6:
            return "center"
        elif y_ratio < 0.4:
            return "upper"
        elif y_ratio > 0.6:
            return "lower"
        elif x_ratio < 0.4:
            return "left"
        else:
            return "right"


# =============================================================================
# Action Detector
# =============================================================================


class ActionDetector:
    """
    Detects gameplay actions from video frames.

    Uses frame differencing, optical flow, and pattern recognition
    to identify actions like shooting, moving, and utility usage.
    """

    # Thresholds for action detection
    MOVEMENT_THRESHOLD = 0.05  # Frame difference threshold for movement
    SHOOTING_FLASH_THRESHOLD = 0.15  # Brightness spike for muzzle flash

    def __init__(self, fps: float):
        """Initialize action detector with video FPS."""
        self.fps = fps
        self.ms_per_frame = 1000.0 / fps

        self.prev_frame = None
        self.prev_gray = None
        self.current_action = ActionType.IDLE
        self.action_start_frame = 0

        # Brightness history for flash detection
        self.brightness_history: list[float] = []
        self.max_brightness_history = 10

    def detect(self, frame: np.ndarray, frame_num: int) -> DetectedAction | None:
        """
        Detect action in current frame.

        Args:
            frame: BGR image as numpy array
            frame_num: Current frame number

        Returns:
            DetectedAction if action state changed, None otherwise
        """
        if not _check_cv2_available():
            return None

        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate average brightness
        brightness = np.mean(gray) / 255.0
        self.brightness_history.append(brightness)
        if len(self.brightness_history) > self.max_brightness_history:
            self.brightness_history.pop(0)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_frame = frame
            return None

        # Calculate frame difference
        frame_diff = cv2.absdiff(gray, self.prev_gray)
        diff_score = np.mean(frame_diff) / 255.0

        # Detect shooting (muzzle flash causes brightness spike)
        shooting_detected = self._detect_shooting(brightness)

        # Determine current action
        new_action = ActionType.IDLE
        confidence = 0.7

        if shooting_detected:
            new_action = ActionType.SHOOTING
            confidence = 0.85
        elif diff_score > self.MOVEMENT_THRESHOLD:
            new_action = ActionType.MOVING
            confidence = 0.75

        # Check for action state change
        result = None
        if new_action != self.current_action:
            # End previous action
            if self.current_action != ActionType.IDLE:
                result = DetectedAction(
                    action_type=self.current_action,
                    start_frame=self.action_start_frame,
                    end_frame=frame_num,
                    start_time_ms=self.action_start_frame * self.ms_per_frame,
                    end_time_ms=frame_num * self.ms_per_frame,
                    confidence=confidence,
                )

            # Start new action
            self.current_action = new_action
            self.action_start_frame = frame_num

        self.prev_gray = gray
        self.prev_frame = frame

        return result

    def _detect_shooting(self, current_brightness: float) -> bool:
        """Detect shooting based on brightness spike (muzzle flash)."""
        if len(self.brightness_history) < 3:
            return False

        # Calculate average brightness excluding current
        avg_brightness = np.mean(self.brightness_history[:-1])

        # Check for sudden brightness increase
        brightness_diff = current_brightness - avg_brightness

        return brightness_diff > self.SHOOTING_FLASH_THRESHOLD

    def finalize(self, total_frames: int) -> DetectedAction | None:
        """Finalize any ongoing action at end of video."""
        if self.current_action != ActionType.IDLE:
            return DetectedAction(
                action_type=self.current_action,
                start_frame=self.action_start_frame,
                end_frame=total_frames,
                start_time_ms=self.action_start_frame * self.ms_per_frame,
                end_time_ms=total_frames * self.ms_per_frame,
                confidence=0.6,
            )
        return None


# =============================================================================
# Mistake Analyzer
# =============================================================================


class MistakeAnalyzer:
    """
    Analyzes gameplay for common mistakes.

    Uses crosshair positions, actions, and frame analysis to detect:
    - Bad crosshair placement (not at head level)
    - Running while shooting
    - Slow reactions
    - Over-peeking
    """

    # Head level range (normalized Y position, 0 = top, 1 = bottom)
    HEAD_LEVEL_MIN = 0.3
    HEAD_LEVEL_MAX = 0.5

    # Crosshair stability threshold (pixels)
    STABILITY_THRESHOLD = 50

    def __init__(self, frame_width: int, frame_height: int, fps: float):
        """Initialize mistake analyzer."""
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.ms_per_frame = 1000.0 / fps

        # Track recent crosshair positions for stability analysis
        self.recent_positions: list[CrosshairPosition] = []
        self.max_recent = 30  # ~0.5 seconds at 60fps

        self.detected_mistakes: list[DetectedMistake] = []

    def analyze_frame(
        self,
        frame_num: int,
        crosshair_pos: CrosshairPosition | None,
        current_action: ActionType,
        prev_action: ActionType | None,
    ) -> DetectedMistake | None:
        """
        Analyze a single frame for mistakes.

        Args:
            frame_num: Current frame number
            crosshair_pos: Detected crosshair position
            current_action: Current detected action
            prev_action: Previous frame's action

        Returns:
            DetectedMistake if found, None otherwise
        """
        if crosshair_pos is None:
            return None

        self.recent_positions.append(crosshair_pos)
        if len(self.recent_positions) > self.max_recent:
            self.recent_positions.pop(0)

        # Check for bad crosshair placement
        mistake = self._check_crosshair_placement(crosshair_pos)
        if mistake:
            return mistake

        # Check for running while shooting
        mistake = self._check_running_shooting(frame_num, current_action, prev_action)
        if mistake:
            return mistake

        # Check for poor crosshair stability
        mistake = self._check_crosshair_stability(frame_num)
        if mistake:
            return mistake

        return None

    def _check_crosshair_placement(self, pos: CrosshairPosition) -> DetectedMistake | None:
        """Check if crosshair is at proper head level."""
        normalized_y = pos.y / self.frame_height

        # Only flag if significantly off head level
        if normalized_y < self.HEAD_LEVEL_MIN - 0.1:
            return DetectedMistake(
                mistake_type=MistakeType.BAD_CROSSHAIR_PLACEMENT,
                frame_num=pos.frame_num,
                timestamp_ms=pos.timestamp_ms,
                severity="medium",
                description="Crosshair placed too high (above head level)",
                suggestion="Lower your crosshair to head level for faster target acquisition",
                confidence=0.7,
            )
        elif normalized_y > self.HEAD_LEVEL_MAX + 0.15:
            return DetectedMistake(
                mistake_type=MistakeType.BAD_CROSSHAIR_PLACEMENT,
                frame_num=pos.frame_num,
                timestamp_ms=pos.timestamp_ms,
                severity="high",
                description="Crosshair placed too low (at body/ground level)",
                suggestion="Raise your crosshair to head level - this is critical for CS2",
                confidence=0.8,
            )

        return None

    def _check_running_shooting(
        self, frame_num: int, current_action: ActionType, prev_action: ActionType | None
    ) -> DetectedMistake | None:
        """Check if player is shooting while moving."""
        # This is a simplified check - in reality would need more context
        if current_action == ActionType.SHOOTING and prev_action == ActionType.MOVING:
            return DetectedMistake(
                mistake_type=MistakeType.RUNNING_WHILE_SHOOTING,
                frame_num=frame_num,
                timestamp_ms=frame_num * self.ms_per_frame,
                severity="high",
                description="Firing while transitioning from movement",
                suggestion="Counter-strafe before shooting for accurate fire",
                confidence=0.6,
            )
        return None

    def _check_crosshair_stability(self, frame_num: int) -> DetectedMistake | None:
        """Check crosshair stability over recent frames."""
        if len(self.recent_positions) < 10:
            return None

        # Calculate position variance
        xs = [p.x for p in self.recent_positions[-10:]]
        ys = [p.y for p in self.recent_positions[-10:]]

        x_std = np.std(xs)
        y_std = np.std(ys)
        total_std = np.sqrt(x_std**2 + y_std**2)

        # High variance indicates shaky aim
        if total_std > self.STABILITY_THRESHOLD * 2:
            return DetectedMistake(
                mistake_type=MistakeType.POOR_MOVEMENT,
                frame_num=frame_num,
                timestamp_ms=frame_num * self.ms_per_frame,
                severity="low",
                description="Crosshair movement appears unstable/shaky",
                suggestion="Practice smooth crosshair control in aim training maps",
                confidence=0.5,
            )

        return None


# =============================================================================
# Coaching Feedback Generator
# =============================================================================


class CoachingFeedbackGenerator:
    """
    Generates personalized coaching feedback based on analysis results.

    Uses detected mistakes, performance metrics, and ML-based pattern
    recognition to provide actionable improvement suggestions.
    """

    def __init__(self):
        """Initialize coaching feedback generator."""
        self.tips_generated: list[CoachingTip] = []

    def generate(
        self,
        metrics: PerformanceMetrics,
        mistakes: list[DetectedMistake],
        crosshair_positions: list[CrosshairPosition],
        actions: list[DetectedAction],
    ) -> list[CoachingTip]:
        """
        Generate personalized coaching tips.

        Args:
            metrics: Performance metrics from analysis
            mistakes: List of detected mistakes
            crosshair_positions: All detected crosshair positions
            actions: All detected actions

        Returns:
            List of prioritized coaching tips
        """
        tips = []

        # Analyze mistake patterns
        tips.extend(self._analyze_mistake_patterns(mistakes))

        # Generate tips based on metrics
        tips.extend(self._generate_metric_tips(metrics))

        # Generate crosshair-specific tips
        tips.extend(self._generate_crosshair_tips(crosshair_positions, metrics))

        # Sort by priority
        tips.sort(key=lambda t: t.priority)

        # Limit to top 5 most important tips
        self.tips_generated = tips[:5]
        return self.tips_generated

    def _analyze_mistake_patterns(self, mistakes: list[DetectedMistake]) -> list[CoachingTip]:
        """Analyze mistake patterns and generate tips."""
        tips = []

        # Count mistakes by type
        mistake_counts: dict[MistakeType, int] = {}
        for mistake in mistakes:
            mistake_counts[mistake.mistake_type] = mistake_counts.get(mistake.mistake_type, 0) + 1

        # Generate tips for most common mistakes
        for mistake_type, count in sorted(mistake_counts.items(), key=lambda x: -x[1])[:3]:
            tip = self._get_tip_for_mistake(mistake_type, count)
            if tip:
                tips.append(tip)

        return tips

    def _get_tip_for_mistake(self, mistake_type: MistakeType, count: int) -> CoachingTip | None:
        """Get coaching tip for a specific mistake type."""
        tips_map = {
            MistakeType.BAD_CROSSHAIR_PLACEMENT: CoachingTip(
                category="aim",
                priority=1,
                title="Improve Crosshair Placement",
                description=f"Detected {count} instances of suboptimal crosshair placement. "
                "Keeping your crosshair at head level is crucial for winning duels.",
                drill_suggestion="Play 15 minutes of Yprac Crosshair Placement map daily",
                related_mistakes=[MistakeType.BAD_CROSSHAIR_PLACEMENT],
            ),
            MistakeType.RUNNING_WHILE_SHOOTING: CoachingTip(
                category="movement",
                priority=2,
                title="Master Counter-Strafing",
                description=f"Detected {count} instances of shooting while moving. "
                "Your bullets are inaccurate when moving.",
                drill_suggestion="Practice counter-strafe timing in deathmatch",
                related_mistakes=[MistakeType.RUNNING_WHILE_SHOOTING],
            ),
            MistakeType.POOR_MOVEMENT: CoachingTip(
                category="movement",
                priority=3,
                title="Stabilize Your Aim",
                description=f"Detected {count} instances of shaky crosshair movement. "
                "Smooth, controlled movements lead to better accuracy.",
                drill_suggestion="Lower your sensitivity slightly and practice tracking",
                related_mistakes=[MistakeType.POOR_MOVEMENT],
            ),
            MistakeType.SLOW_REACTION: CoachingTip(
                category="aim",
                priority=2,
                title="Improve Reaction Time",
                description=f"Detected {count} slow reactions to enemy appearances.",
                drill_suggestion="Play aim trainers focusing on reaction time",
                related_mistakes=[MistakeType.SLOW_REACTION],
            ),
        }

        return tips_map.get(mistake_type)

    def _generate_metric_tips(self, metrics: PerformanceMetrics) -> list[CoachingTip]:
        """Generate tips based on performance metrics."""
        tips = []

        if metrics.crosshair_placement_score < 60:
            tips.append(
                CoachingTip(
                    category="aim",
                    priority=1,
                    title="Focus on Crosshair Placement",
                    description=f"Your crosshair placement score is {metrics.crosshair_placement_score:.0f}/100. "
                    "This is the #1 skill that separates good players from great ones.",
                    drill_suggestion="Watch pro player POVs and study their crosshair placement",
                    related_mistakes=[MistakeType.BAD_CROSSHAIR_PLACEMENT],
                )
            )

        if metrics.crosshair_stability < 50:
            tips.append(
                CoachingTip(
                    category="aim",
                    priority=2,
                    title="Improve Crosshair Stability",
                    description=f"Your crosshair stability is {metrics.crosshair_stability:.0f}/100. "
                    "A stable crosshair leads to more consistent aim.",
                    drill_suggestion="Consider lowering your mouse sensitivity",
                    related_mistakes=[MistakeType.POOR_MOVEMENT],
                )
            )

        if metrics.movement_score < 50:
            tips.append(
                CoachingTip(
                    category="movement",
                    priority=3,
                    title="Work on Movement Mechanics",
                    description=f"Your movement score is {metrics.movement_score:.0f}/100. "
                    "Good movement makes you harder to hit and improves positioning.",
                    drill_suggestion="Practice strafing and bunny hopping in KZ maps",
                    related_mistakes=[],
                )
            )

        return tips

    def _generate_crosshair_tips(
        self, positions: list[CrosshairPosition], metrics: PerformanceMetrics
    ) -> list[CoachingTip]:
        """Generate tips based on crosshair position analysis."""
        tips = []

        if not positions:
            return tips

        # Analyze average crosshair height
        if metrics.average_crosshair_height > 0.55:
            tips.append(
                CoachingTip(
                    category="aim",
                    priority=1,
                    title="Raise Your Crosshair",
                    description="Your crosshair tends to be placed too low on average. "
                    "This means you need to flick up for headshots.",
                    drill_suggestion="Consciously aim at head level while clearing angles",
                    related_mistakes=[MistakeType.BAD_CROSSHAIR_PLACEMENT],
                )
            )

        return tips


# =============================================================================
# Video Analyzer (Main Class)
# =============================================================================


class VideoAnalyzer:
    """
    Main video analysis class.

    Orchestrates crosshair tracking, action detection, mistake analysis,
    and coaching feedback generation for CS2 gameplay videos.

    Usage:
        analyzer = VideoAnalyzer("gameplay.mp4")
        results = analyzer.analyze()

        print(f"Overall Score: {results.performance_metrics.overall_score}")
        for tip in results.coaching_tips:
            print(f"- {tip.title}: {tip.description}")
    """

    # Supported video formats
    SUPPORTED_FORMATS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv"}

    def __init__(self, video_path: str | Path):
        """
        Initialize video analyzer.

        Args:
            video_path: Path to the gameplay video file
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if self.video_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported video format: {self.video_path.suffix}. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )

        self._video_info: dict[str, Any] = {}
        self._initialized = False

    def _initialize(self) -> bool:
        """Initialize video capture and get video info."""
        if self._initialized:
            return True

        if not _check_cv2_available():
            raise ImportError(
                "OpenCV is required for video analysis. "
                "Install with: pip install opencv-python-headless"
            )

        import cv2

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")

        self._video_info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration_seconds": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
        }

        cap.release()
        self._initialized = True
        return True

    def analyze(
        self,
        detect_crosshair: bool = True,
        detect_actions: bool = True,
        detect_mistakes: bool = True,
        generate_coaching: bool = True,
        sample_rate: int = 2,  # Analyze every Nth frame
        progress_callback: Any | None = None,
    ) -> VideoAnalysisResult:
        """
        Analyze the gameplay video.

        Args:
            detect_crosshair: Enable crosshair tracking
            detect_actions: Enable action detection
            detect_mistakes: Enable mistake analysis
            generate_coaching: Generate coaching feedback
            sample_rate: Analyze every Nth frame (higher = faster, less accurate)
            progress_callback: Optional callback(current_frame, total_frames)

        Returns:
            VideoAnalysisResult with all analysis data
        """
        self._initialize()

        import cv2

        logger.info(f"Starting video analysis: {self.video_path}")
        start_time = time.time()

        # Initialize components
        cap = cv2.VideoCapture(str(self.video_path))
        width = self._video_info["width"]
        height = self._video_info["height"]
        fps = self._video_info["fps"]
        total_frames = self._video_info["total_frames"]
        ms_per_frame = 1000.0 / fps

        crosshair_tracker = CrosshairTracker(width, height) if detect_crosshair else None
        action_detector = ActionDetector(fps) if detect_actions else None
        mistake_analyzer = MistakeAnalyzer(width, height, fps) if detect_mistakes else None

        # Results storage
        crosshair_positions: list[CrosshairPosition] = []
        detected_actions: list[DetectedAction] = []
        detected_mistakes: list[DetectedMistake] = []

        frame_num = 0
        frames_analyzed = 0
        prev_action = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Sample frames for efficiency
            if frame_num % sample_rate != 0:
                frame_num += 1
                continue

            timestamp_ms = frame_num * ms_per_frame
            frames_analyzed += 1

            # Crosshair tracking
            crosshair_pos = None
            if crosshair_tracker:
                crosshair_pos = crosshair_tracker.detect(frame, frame_num, timestamp_ms)
                if crosshair_pos:
                    crosshair_positions.append(crosshair_pos)

            # Action detection
            current_action = ActionType.IDLE
            if action_detector:
                action = action_detector.detect(frame, frame_num)
                if action:
                    detected_actions.append(action)
                current_action = action_detector.current_action

            # Mistake analysis
            if mistake_analyzer:
                mistake = mistake_analyzer.analyze_frame(
                    frame_num, crosshair_pos, current_action, prev_action
                )
                if mistake:
                    detected_mistakes.append(mistake)

            prev_action = current_action

            # Progress callback
            if progress_callback:
                progress_callback(frame_num, total_frames)

            frame_num += 1

        cap.release()

        # Finalize action detection
        if action_detector:
            final_action = action_detector.finalize(frame_num)
            if final_action:
                detected_actions.append(final_action)

        # Calculate performance metrics
        metrics = self._calculate_metrics(
            crosshair_positions, detected_actions, detected_mistakes, width, height, fps
        )

        # Generate coaching feedback
        coaching_tips = []
        if generate_coaching:
            feedback_generator = CoachingFeedbackGenerator()
            coaching_tips = feedback_generator.generate(
                metrics, detected_mistakes, crosshair_positions, detected_actions
            )

        analysis_time = time.time() - start_time
        logger.info(f"Video analysis completed in {analysis_time:.2f}s")

        return VideoAnalysisResult(
            video_path=str(self.video_path),
            duration_seconds=self._video_info["duration_seconds"],
            total_frames=total_frames,
            fps=fps,
            resolution=(width, height),
            crosshair_positions=crosshair_positions,
            detected_actions=detected_actions,
            detected_mistakes=detected_mistakes,
            performance_metrics=metrics,
            coaching_tips=coaching_tips,
            analysis_time_seconds=analysis_time,
            frames_analyzed=frames_analyzed,
            sample_rate=sample_rate,
        )

    def _calculate_metrics(
        self,
        crosshair_positions: list[CrosshairPosition],
        actions: list[DetectedAction],
        mistakes: list[DetectedMistake],
        width: int,
        height: int,
        fps: float,
    ) -> PerformanceMetrics:
        """Calculate performance metrics from analysis data."""
        metrics = PerformanceMetrics()

        if crosshair_positions:
            # Crosshair stability (inverse of position variance)
            xs = [p.x for p in crosshair_positions]
            ys = [p.y for p in crosshair_positions]
            x_std = np.std(xs) if xs else 0
            y_std = np.std(ys) if ys else 0

            # Normalize to 0-100 scale
            max_std = np.sqrt(width**2 + height**2) / 4
            stability = max(0, 100 - (np.sqrt(x_std**2 + y_std**2) / max_std * 100))
            metrics.crosshair_stability = round(stability, 1)

            # Crosshair placement score (how often at head level)
            head_level_count = sum(1 for p in crosshair_positions if 0.3 <= p.y / height <= 0.5)
            metrics.crosshair_placement_score = round(
                head_level_count / len(crosshair_positions) * 100, 1
            )

            # Average crosshair height
            metrics.average_crosshair_height = round(
                np.mean([p.y / height for p in crosshair_positions]), 3
            )

            # Movement smoothness (based on frame-to-frame changes)
            if len(crosshair_positions) > 1:
                movements = []
                for i in range(1, len(crosshair_positions)):
                    dx = crosshair_positions[i].x - crosshair_positions[i - 1].x
                    dy = crosshair_positions[i].y - crosshair_positions[i - 1].y
                    movements.append(np.sqrt(dx**2 + dy**2))

                # Smoothness is inverse of movement variance
                if movements:
                    move_std = np.std(movements)
                    smoothness = max(0, 100 - move_std * 2)
                    metrics.crosshair_movement_smoothness = round(smoothness, 1)

        # Action metrics
        shooting_actions = [a for a in actions if a.action_type == ActionType.SHOOTING]
        metrics.shots_detected = len(shooting_actions)

        moving_actions = [a for a in actions if a.action_type == ActionType.MOVING]
        if actions:
            total_action_time = sum(a.end_time_ms - a.start_time_ms for a in actions)
            moving_time = sum(a.end_time_ms - a.start_time_ms for a in moving_actions)
            metrics.movement_efficiency = round(
                (1 - moving_time / max(total_action_time, 1)) * 100, 1
            )

        # Calculate overall scores
        metrics.aim_score = round(
            (metrics.crosshair_stability + metrics.crosshair_placement_score) / 2, 1
        )
        metrics.movement_score = round(
            (metrics.crosshair_movement_smoothness + metrics.movement_efficiency) / 2, 1
        )

        # Penalize for mistakes
        mistake_penalty = min(len(mistakes) * 2, 30)
        metrics.overall_score = round(
            max(0, (metrics.aim_score + metrics.movement_score) / 2 - mistake_penalty), 1
        )

        # Estimate skill level
        if metrics.overall_score >= 80:
            metrics.estimated_skill_level = SkillLevel.PROFESSIONAL
        elif metrics.overall_score >= 65:
            metrics.estimated_skill_level = SkillLevel.EXPERT
        elif metrics.overall_score >= 50:
            metrics.estimated_skill_level = SkillLevel.ADVANCED
        elif metrics.overall_score >= 35:
            metrics.estimated_skill_level = SkillLevel.INTERMEDIATE
        else:
            metrics.estimated_skill_level = SkillLevel.BEGINNER

        return metrics


# =============================================================================
# Convenience Functions
# =============================================================================


def analyze_gameplay_video(video_path: str | Path, sample_rate: int = 2) -> VideoAnalysisResult:
    """
    Convenience function to analyze a gameplay video.

    Args:
        video_path: Path to the video file
        sample_rate: Analyze every Nth frame (higher = faster)

    Returns:
        VideoAnalysisResult with all analysis data

    Example:
        results = analyze_gameplay_video("gameplay.mp4")
        print(f"Score: {results.performance_metrics.overall_score}")

        for tip in results.coaching_tips:
            print(f"Tip: {tip.title}")
            print(f"  {tip.description}")
    """
    analyzer = VideoAnalyzer(video_path)
    return analyzer.analyze(sample_rate=sample_rate)


def get_video_info(video_path: str | Path) -> dict[str, Any]:
    """
    Get basic information about a video file.

    Args:
        video_path: Path to the video file

    Returns:
        Dictionary with video info (width, height, fps, duration, etc.)
    """
    if not _check_cv2_available():
        raise ImportError("OpenCV required. Install with: pip install opencv-python-headless")

    import cv2

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    info = {
        "path": str(path),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration_seconds": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
        "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
    }

    cap.release()
    return info
