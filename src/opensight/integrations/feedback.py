"""
Community Feedback Module for OpenSight

Provides:
- User ratings and reviews for analyses
- Comments and annotations
- Feedback collection for AI improvement
- Analytics on user engagement

Uses the main SQLAlchemy database via DatabaseManager.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime

from opensight.infra.database import (
    AnnotationRecord,
    CoachingFeedbackRecord,
    FeedbackRecord,
    get_db,
)

logger = logging.getLogger(__name__)


@dataclass
class FeedbackEntry:
    """A feedback entry from a user."""

    id: int | None
    demo_hash: str
    user_id: str
    rating: int
    category: str
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
    insight_category: str
    insight_message: str
    was_helpful: bool
    user_correction: str | None
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
    annotation_type: str
    content: str
    x: float | None
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
    rating_distribution: dict[int, int]
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
    """SQLAlchemy-based feedback storage using the main database."""

    def __init__(self) -> None:
        self._db = get_db()

    def add_feedback(self, feedback: FeedbackEntry) -> int:
        """Add a feedback entry. Returns ID of inserted entry."""
        session = self._db.get_session()
        try:
            record = FeedbackRecord(
                demo_hash=feedback.demo_hash,
                user_id=feedback.user_id,
                rating=max(1, min(5, feedback.rating)),
                category=feedback.category,
                comment=feedback.comment,
                analysis_version=feedback.analysis_version,
                metadata_json=json.dumps(feedback.metadata) if feedback.metadata else None,
            )
            session.add(record)
            session.commit()
            return record.id
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to add feedback: {e}")
            raise
        finally:
            session.close()

    def add_coaching_feedback(self, feedback: CoachingFeedback) -> int:
        """Add feedback on a coaching insight. Returns ID."""
        session = self._db.get_session()
        try:
            record = CoachingFeedbackRecord(
                demo_hash=feedback.demo_hash,
                player_steam_id=feedback.player_steam_id,
                insight_category=feedback.insight_category,
                insight_message=feedback.insight_message,
                was_helpful=feedback.was_helpful,
                user_correction=feedback.user_correction,
            )
            session.add(record)
            session.commit()
            return record.id
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to add coaching feedback: {e}")
            raise
        finally:
            session.close()

    def add_annotation(self, annotation: AnalysisAnnotation) -> int:
        """Add an annotation. Returns ID."""
        session = self._db.get_session()
        try:
            record = AnnotationRecord(
                demo_hash=annotation.demo_hash,
                user_id=annotation.user_id,
                round_num=annotation.round_num,
                tick=annotation.tick,
                annotation_type=annotation.annotation_type,
                content=annotation.content,
                x=annotation.x,
                y=annotation.y,
            )
            session.add(record)
            session.commit()
            return record.id
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to add annotation: {e}")
            raise
        finally:
            session.close()

    def get_stats(self) -> FeedbackStats:
        """Get aggregated feedback statistics."""
        from sqlalchemy import func

        session = self._db.get_session()
        try:
            # Total feedback and average rating
            row = session.query(
                func.count(FeedbackRecord.id),
                func.avg(FeedbackRecord.rating),
            ).first()
            total_feedback = row[0] or 0
            avg_rating = float(row[1]) if row[1] else 0.0

            # Rating distribution
            rating_dist = {}
            for i in range(1, 6):
                count = (
                    session.query(func.count(FeedbackRecord.id))
                    .filter(FeedbackRecord.rating == i)
                    .scalar()
                    or 0
                )
                rating_dist[i] = count

            # Category counts
            category_counts = {}
            rows = (
                session.query(
                    FeedbackRecord.category,
                    func.count(FeedbackRecord.id),
                )
                .group_by(FeedbackRecord.category)
                .all()
            )
            for cat, cnt in rows:
                category_counts[cat] = cnt

            # Coaching helpful rate
            coaching_row = session.query(
                func.count(CoachingFeedbackRecord.id),
                func.sum(func.cast(CoachingFeedbackRecord.was_helpful, type_=None)),
            ).first()
            coaching_total = coaching_row[0] or 0
            coaching_helpful = coaching_row[1] or 0
            coaching_rate = (coaching_helpful / coaching_total * 100) if coaching_total > 0 else 0.0

            # Total annotations
            annotations_count = session.query(func.count(AnnotationRecord.id)).scalar() or 0

            return FeedbackStats(
                total_feedback=total_feedback,
                average_rating=avg_rating,
                rating_distribution=rating_dist,
                category_counts=category_counts,
                coaching_helpful_rate=coaching_rate,
                total_annotations=annotations_count,
            )
        finally:
            session.close()


# Convenience functions


def submit_feedback(
    demo_hash: str,
    rating: int,
    category: str = "overall",
    comment: str = "",
) -> int:
    """Submit feedback for an analysis."""
    user_id = hashlib.md5(str(datetime.now()).encode(), usedforsecurity=False).hexdigest()[:12]
    db = FeedbackDatabase()
    feedback = FeedbackEntry(
        id=None,
        demo_hash=demo_hash,
        user_id=user_id,
        rating=max(1, min(5, rating)),
        category=category,
        comment=comment,
        analysis_version="0.5.0",
        created_at=datetime.now(),
    )
    return db.add_feedback(feedback)


def get_feedback_stats() -> dict:
    """Get feedback statistics."""
    db = FeedbackDatabase()
    return db.get_stats().to_dict()
