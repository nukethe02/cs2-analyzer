"""
Community Feedback Module for OpenSight

Provides:
- User ratings and reviews for analyses
- Comments and annotations
- Feedback collection for AI improvement
- Analytics on user engagement
- SQLite-based persistent storage

This module enables community-driven improvement of the analyzer
by collecting structured feedback that can be used for:
- Identifying common issues
- Prioritizing feature development
- Training/improving AI coaching suggestions
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Default database location
DEFAULT_DB_PATH = Path.home() / ".opensight" / "feedback.db"


@dataclass
class FeedbackEntry:
    """A feedback entry from a user."""

    id: int | None
    demo_hash: str  # Hash of demo file for reference
    user_id: str  # Anonymous user identifier
    rating: int  # 1-5 stars
    category: str  # "accuracy", "usefulness", "ui", "coaching", "other"
    comment: str
    analysis_version: str
    created_at: datetime
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "demo_hash": self.demo_hash,
            "user_id": self.user_id,
            "rating": self.rating,
            "category": self.category,
            "comment": self.comment,
            "analysis_version": self.analysis_version,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class CoachingFeedback:
    """Feedback on a specific coaching insight."""

    id: int | None
    demo_hash: str
    player_steam_id: str
    insight_category: str  # "aim", "positioning", "utility", etc.
    insight_message: str
    was_helpful: bool
    user_correction: str | None  # User's suggested correction
    created_at: datetime

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "demo_hash": self.demo_hash,
            "player_steam_id": self.player_steam_id,
            "insight_category": self.insight_category,
            "insight_message": self.insight_message,
            "was_helpful": self.was_helpful,
            "user_correction": self.user_correction,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class AnalysisAnnotation:
    """User annotation on an analysis."""

    id: int | None
    demo_hash: str
    user_id: str
    round_num: int | None
    tick: int | None
    annotation_type: str  # "highlight", "mistake", "question", "note"
    content: str
    x: float | None  # Position on radar
    y: float | None
    created_at: datetime

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "demo_hash": self.demo_hash,
            "user_id": self.user_id,
            "round_num": self.round_num,
            "tick": self.tick,
            "annotation_type": self.annotation_type,
            "content": self.content,
            "x": self.x,
            "y": self.y,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class FeedbackStats:
    """Aggregated feedback statistics."""

    total_feedback: int
    average_rating: float
    rating_distribution: dict[int, int]  # rating -> count
    category_counts: dict[str, int]
    coaching_helpful_rate: float
    total_annotations: int

    def to_dict(self) -> dict:
        return {
            "total_feedback": self.total_feedback,
            "average_rating": round(self.average_rating, 2),
            "rating_distribution": self.rating_distribution,
            "category_counts": self.category_counts,
            "coaching_helpful_rate": round(self.coaching_helpful_rate, 2),
            "total_annotations": self.total_annotations,
        }


class FeedbackDatabase:
    """
    SQLite-based feedback storage.
    """

    def __init__(self, db_path: Path | None = None):
        """
        Initialize the feedback database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    demo_hash TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
                    category TEXT NOT NULL,
                    comment TEXT,
                    analysis_version TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                );

                CREATE TABLE IF NOT EXISTS coaching_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    demo_hash TEXT NOT NULL,
                    player_steam_id TEXT,
                    insight_category TEXT NOT NULL,
                    insight_message TEXT NOT NULL,
                    was_helpful INTEGER NOT NULL,
                    user_correction TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS annotations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    demo_hash TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    round_num INTEGER,
                    tick INTEGER,
                    annotation_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    x REAL,
                    y REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_feedback_demo ON feedback(demo_hash);
                CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback(user_id);
                CREATE INDEX IF NOT EXISTS idx_coaching_demo ON coaching_feedback(demo_hash);
                CREATE INDEX IF NOT EXISTS idx_annotations_demo ON annotations(demo_hash);
            """)
            conn.commit()

    @contextmanager
    def _connect(self):
        """Context manager for database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def add_feedback(self, feedback: FeedbackEntry) -> int:
        """
        Add a feedback entry.

        Args:
            feedback: Feedback entry to add

        Returns:
            ID of inserted entry
        """
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO feedback (demo_hash, user_id, rating, category, comment, analysis_version, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback.demo_hash,
                    feedback.user_id,
                    feedback.rating,
                    feedback.category,
                    feedback.comment,
                    feedback.analysis_version,
                    json.dumps(feedback.metadata),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def add_coaching_feedback(self, feedback: CoachingFeedback) -> int:
        """Add feedback on a coaching insight."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO coaching_feedback (demo_hash, player_steam_id, insight_category, insight_message, was_helpful, user_correction)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback.demo_hash,
                    feedback.player_steam_id,
                    feedback.insight_category,
                    feedback.insight_message,
                    1 if feedback.was_helpful else 0,
                    feedback.user_correction,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def add_annotation(self, annotation: AnalysisAnnotation) -> int:
        """Add an annotation."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO annotations (demo_hash, user_id, round_num, tick, annotation_type, content, x, y)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    annotation.demo_hash,
                    annotation.user_id,
                    annotation.round_num,
                    annotation.tick,
                    annotation.annotation_type,
                    annotation.content,
                    annotation.x,
                    annotation.y,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_feedback_for_demo(self, demo_hash: str) -> list[FeedbackEntry]:
        """Get all feedback for a demo."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM feedback WHERE demo_hash = ? ORDER BY created_at DESC",
                (demo_hash,),
            ).fetchall()

        return [
            FeedbackEntry(
                id=row["id"],
                demo_hash=row["demo_hash"],
                user_id=row["user_id"],
                rating=row["rating"],
                category=row["category"],
                comment=row["comment"] or "",
                analysis_version=row["analysis_version"] or "",
                created_at=(
                    datetime.fromisoformat(row["created_at"])
                    if row["created_at"]
                    else datetime.now()
                ),
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )
            for row in rows
        ]

    def get_annotations_for_demo(self, demo_hash: str) -> list[AnalysisAnnotation]:
        """Get all annotations for a demo."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM annotations WHERE demo_hash = ? ORDER BY round_num, tick",
                (demo_hash,),
            ).fetchall()

        return [
            AnalysisAnnotation(
                id=row["id"],
                demo_hash=row["demo_hash"],
                user_id=row["user_id"],
                round_num=row["round_num"],
                tick=row["tick"],
                annotation_type=row["annotation_type"],
                content=row["content"],
                x=row["x"],
                y=row["y"],
                created_at=(
                    datetime.fromisoformat(row["created_at"])
                    if row["created_at"]
                    else datetime.now()
                ),
            )
            for row in rows
        ]

    def get_stats(self) -> FeedbackStats:
        """Get aggregated feedback statistics."""
        with self._connect() as conn:
            # Total feedback and average rating
            row = conn.execute(
                "SELECT COUNT(*) as total, AVG(rating) as avg_rating FROM feedback"
            ).fetchone()
            total_feedback = row["total"]
            avg_rating = row["avg_rating"] or 0.0

            # Rating distribution
            rating_dist = {}
            for i in range(1, 6):
                count = conn.execute(
                    "SELECT COUNT(*) as cnt FROM feedback WHERE rating = ?",
                    (i,),
                ).fetchone()["cnt"]
                rating_dist[i] = count

            # Category counts
            category_counts = {}
            rows = conn.execute(
                "SELECT category, COUNT(*) as cnt FROM feedback GROUP BY category"
            ).fetchall()
            for row in rows:
                category_counts[row["category"]] = row["cnt"]

            # Coaching helpful rate
            coaching_row = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN was_helpful = 1 THEN 1 ELSE 0 END) as helpful
                FROM coaching_feedback
                """).fetchone()
            coaching_total = coaching_row["total"]
            coaching_helpful = coaching_row["helpful"] or 0
            coaching_rate = (coaching_helpful / coaching_total * 100) if coaching_total > 0 else 0.0

            # Total annotations
            annotations_count = conn.execute("SELECT COUNT(*) as cnt FROM annotations").fetchone()[
                "cnt"
            ]

        return FeedbackStats(
            total_feedback=total_feedback,
            average_rating=avg_rating,
            rating_distribution=rating_dist,
            category_counts=category_counts,
            coaching_helpful_rate=coaching_rate,
            total_annotations=annotations_count,
        )

    def get_coaching_insights_feedback(self, category: str | None = None) -> list[dict]:
        """
        Get coaching feedback for analysis/improvement.

        Args:
            category: Filter by insight category

        Returns:
            List of feedback with aggregated stats
        """
        with self._connect() as conn:
            if category:
                rows = conn.execute(
                    """
                    SELECT insight_message,
                           COUNT(*) as total,
                           SUM(CASE WHEN was_helpful = 1 THEN 1 ELSE 0 END) as helpful,
                           GROUP_CONCAT(user_correction, '|||') as corrections
                    FROM coaching_feedback
                    WHERE insight_category = ?
                    GROUP BY insight_message
                    ORDER BY total DESC
                    """,
                    (category,),
                ).fetchall()
            else:
                rows = conn.execute("""
                    SELECT insight_category, insight_message,
                           COUNT(*) as total,
                           SUM(CASE WHEN was_helpful = 1 THEN 1 ELSE 0 END) as helpful,
                           GROUP_CONCAT(user_correction, '|||') as corrections
                    FROM coaching_feedback
                    GROUP BY insight_category, insight_message
                    ORDER BY total DESC
                    """).fetchall()

        results = []
        for row in rows:
            corrections = []
            if row["corrections"]:
                corrections = [c for c in row["corrections"].split("|||") if c]

            results.append(
                {
                    "category": row.get("insight_category", category),
                    "message": row["insight_message"],
                    "total_feedback": row["total"],
                    "helpful_count": row["helpful"],
                    "helpful_rate": (
                        (row["helpful"] / row["total"] * 100) if row["total"] > 0 else 0
                    ),
                    "user_corrections": corrections[:5],  # Limit to 5
                }
            )

        return results

    def export_for_training(self) -> dict:
        """
        Export feedback data for AI model training/improvement.

        Returns:
            Dict with structured feedback data
        """
        with self._connect() as conn:
            # Get all coaching feedback
            coaching = conn.execute(
                "SELECT * FROM coaching_feedback ORDER BY created_at"
            ).fetchall()

            # Get all annotations
            annotations = conn.execute("SELECT * FROM annotations ORDER BY created_at").fetchall()

            # Get highly rated analyses
            good_analyses = conn.execute("""
                SELECT demo_hash, AVG(rating) as avg_rating, COUNT(*) as feedback_count
                FROM feedback
                GROUP BY demo_hash
                HAVING AVG(rating) >= 4
                """).fetchall()

        return {
            "coaching_feedback": [
                {
                    "category": r["insight_category"],
                    "message": r["insight_message"],
                    "was_helpful": bool(r["was_helpful"]),
                    "correction": r["user_correction"],
                }
                for r in coaching
            ],
            "annotations": [
                {
                    "type": r["annotation_type"],
                    "content": r["content"],
                    "round": r["round_num"],
                }
                for r in annotations
            ],
            "high_rated_demos": [
                {
                    "demo_hash": r["demo_hash"],
                    "avg_rating": r["avg_rating"],
                    "feedback_count": r["feedback_count"],
                }
                for r in good_analyses
            ],
            "exported_at": datetime.now().isoformat(),
        }


class FeedbackCollector:
    """
    High-level interface for collecting user feedback.
    """

    # Analysis version for tracking
    ANALYSIS_VERSION = "2.0.0"

    def __init__(self, db: FeedbackDatabase | None = None):
        """
        Initialize the feedback collector.

        Args:
            db: Database instance (creates default if None)
        """
        self.db = db or FeedbackDatabase()

    def generate_user_id(self) -> str:
        """Generate anonymous user ID based on session."""
        import uuid

        return hashlib.md5(str(uuid.uuid4()).encode()).hexdigest()[:12]

    def submit_rating(
        self,
        demo_hash: str,
        user_id: str,
        rating: int,
        category: str = "overall",
        comment: str = "",
        metadata: dict | None = None,
    ) -> int:
        """
        Submit a rating for an analysis.

        Args:
            demo_hash: Hash of demo file
            user_id: User identifier
            rating: 1-5 stars
            category: Feedback category
            comment: Optional comment
            metadata: Optional additional metadata

        Returns:
            Feedback entry ID
        """
        feedback = FeedbackEntry(
            id=None,
            demo_hash=demo_hash,
            user_id=user_id,
            rating=max(1, min(5, rating)),
            category=category,
            comment=comment,
            analysis_version=self.ANALYSIS_VERSION,
            created_at=datetime.now(),
            metadata=metadata or {},
        )
        return self.db.add_feedback(feedback)

    def submit_coaching_feedback(
        self,
        demo_hash: str,
        player_steam_id: str,
        insight_category: str,
        insight_message: str,
        was_helpful: bool,
        user_correction: str | None = None,
    ) -> int:
        """
        Submit feedback on a coaching insight.

        Args:
            demo_hash: Hash of demo file
            player_steam_id: Steam ID of player
            insight_category: Category of insight
            insight_message: The insight message
            was_helpful: Whether it was helpful
            user_correction: User's suggested correction

        Returns:
            Feedback entry ID
        """
        feedback = CoachingFeedback(
            id=None,
            demo_hash=demo_hash,
            player_steam_id=player_steam_id,
            insight_category=insight_category,
            insight_message=insight_message,
            was_helpful=was_helpful,
            user_correction=user_correction,
            created_at=datetime.now(),
        )
        return self.db.add_coaching_feedback(feedback)

    def add_annotation(
        self,
        demo_hash: str,
        user_id: str,
        content: str,
        annotation_type: str = "note",
        round_num: int | None = None,
        tick: int | None = None,
        x: float | None = None,
        y: float | None = None,
    ) -> int:
        """
        Add an annotation to an analysis.

        Args:
            demo_hash: Hash of demo file
            user_id: User identifier
            content: Annotation text
            annotation_type: Type of annotation
            round_num: Round number
            tick: Tick number
            x: X position on radar
            y: Y position on radar

        Returns:
            Annotation ID
        """
        annotation = AnalysisAnnotation(
            id=None,
            demo_hash=demo_hash,
            user_id=user_id,
            round_num=round_num,
            tick=tick,
            annotation_type=annotation_type,
            content=content,
            x=x,
            y=y,
            created_at=datetime.now(),
        )
        return self.db.add_annotation(annotation)

    def get_stats(self) -> dict:
        """Get feedback statistics."""
        return self.db.get_stats().to_dict()


# Convenience functions


def submit_feedback(
    demo_hash: str,
    rating: int,
    category: str = "overall",
    comment: str = "",
) -> int:
    """Submit feedback for an analysis."""
    collector = FeedbackCollector()
    user_id = collector.generate_user_id()
    return collector.submit_rating(demo_hash, user_id, rating, category, comment)


def get_feedback_stats() -> dict:
    """Get feedback statistics."""
    collector = FeedbackCollector()
    return collector.get_stats()


def export_training_data() -> dict:
    """Export feedback data for training."""
    db = FeedbackDatabase()
    return db.export_for_training()
