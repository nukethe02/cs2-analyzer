"""
Persistent Job Store - Track async analysis jobs across restarts.

Provides persistent job tracking using the database, allowing jobs to survive
server restarts and be queried by external systems.

STATUS: Not yet wired in. The API currently uses the in-memory JobStore from
api/shared.py. To switch to persistent storage, update _get_job_store() in
api/shared.py to return a PersistentJobStore instance instead.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from opensight.infra.database import DatabaseManager, Job

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(UTC)


class PersistentJobStore:
    """
    Persistent job tracking using database storage.

    Stores job metadata, status, and results in SQLite database for durability
    across server restarts.
    """

    def __init__(self, db_manager: DatabaseManager | None = None):
        """
        Initialize job store.

        Args:
            db_manager: Optional DatabaseManager instance. If None, uses global instance.
        """
        if db_manager is None:
            from opensight.infra.database import get_db

            db_manager = get_db()
        self.db = db_manager

    def create_job(self, filename: str, file_size: int) -> dict[str, Any]:
        """
        Create a new job entry in persistent storage.

        Args:
            filename: Original filename of uploaded demo
            file_size: File size in bytes

        Returns:
            Job dictionary with job_id, status, created_at, etc.
        """
        session = self.db.get_session()
        try:
            job_id = str(uuid.uuid4())
            job = Job(
                id=job_id,
                filename=filename,
                file_size=file_size,
                status="pending",
                created_at=_utc_now(),
            )
            session.add(job)
            session.commit()

            logger.info(f"Created job {job_id} for {filename} ({file_size} bytes)")

            return {
                "job_id": job_id,
                "filename": filename,
                "file_size": file_size,
                "status": "pending",
                "created_at": job.created_at.isoformat() if job.created_at else None,
            }
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to create job: {e}")
            raise
        finally:
            session.close()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """
        Retrieve job status and details.

        Args:
            job_id: Job UUID

        Returns:
            Job dictionary or None if not found
        """
        session = self.db.get_session()
        try:
            job = session.query(Job).filter(Job.id == job_id).first()
            if not job:
                return None

            result = {
                "job_id": job.id,
                "filename": job.filename,
                "file_size": job.file_size,
                "status": job.status,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            }

            if job.error_message:
                result["error"] = job.error_message

            if job.result_json:
                try:
                    result["result"] = json.loads(job.result_json)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in job {job_id} result")

            if job.demo_hash:
                result["demo_hash"] = job.demo_hash

            return result
        finally:
            session.close()

    def update_status(
        self,
        job_id: str,
        status: str | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        demo_hash: str | None = None,
    ) -> None:
        """
        Update job status and optionally store result or error.

        Args:
            job_id: Job UUID
            status: New status ("pending", "processing", "completed", "failed"), or None to skip
            result: Optional result dictionary to serialize to JSON
            error: Optional error message
            demo_hash: Optional demo file hash for deduplication
        """
        session = self.db.get_session()
        try:
            job = session.query(Job).filter(Job.id == job_id).first()
            if not job:
                logger.warning(f"Job {job_id} not found for status update")
                return

            if status is not None:
                job.status = status

                if status == "processing" and not job.started_at:
                    job.started_at = _utc_now()

                if status in ("completed", "failed"):
                    job.completed_at = _utc_now()

            if result is not None:
                job.result_json = json.dumps(result)

            if error is not None:
                job.error_message = error

            if demo_hash is not None:
                job.demo_hash = demo_hash

            session.commit()
            logger.info(f"Updated job {job_id} to status: {status}")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update job {job_id}: {e}")
            raise
        finally:
            session.close()

    def list_jobs(self, limit: int = 20, status_filter: str | None = None) -> list[dict[str, Any]]:
        """
        List recent jobs, optionally filtered by status.

        Args:
            limit: Maximum number of jobs to return
            status_filter: Optional status to filter by ("pending", "processing", "completed", "failed")

        Returns:
            List of job dictionaries, sorted by creation time (newest first)
        """
        session = self.db.get_session()
        try:
            query = session.query(Job)

            if status_filter:
                query = query.filter(Job.status == status_filter)

            jobs = query.order_by(Job.created_at.desc()).limit(limit).all()

            return [
                {
                    "job_id": j.id,
                    "filename": j.filename,
                    "status": j.status,
                    "created_at": j.created_at.isoformat() if j.created_at else None,
                    "completed_at": j.completed_at.isoformat() if j.completed_at else None,
                }
                for j in jobs
            ]
        finally:
            session.close()

    def cleanup_expired(self, retention_days: int = 7) -> int:
        """
        Remove old completed/failed jobs to save space.

        Args:
            retention_days: Keep jobs from last N days

        Returns:
            Number of jobs deleted
        """
        session = self.db.get_session()
        try:
            cutoff = _utc_now() - timedelta(days=retention_days)
            deleted = (
                session.query(Job)
                .filter(
                    Job.status.in_(["completed", "failed"]),
                    Job.completed_at < cutoff,
                )
                .delete(synchronize_session=False)
            )
            session.commit()

            if deleted > 0:
                logger.info(f"Cleaned up {deleted} expired jobs older than {retention_days} days")

            return deleted
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to cleanup expired jobs: {e}")
            raise
        finally:
            session.close()
