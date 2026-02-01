"""
Collaborative Analysis Module for CS2 Demo Analyzer.

Enables multi-user sessions where coaches and players can
annotate demos collaboratively with real-time synchronization.
"""

import hashlib
import json
import logging
import queue
import re
import secrets
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# =============================================================================
# Secure Password Hashing (bcrypt with fallback)
# =============================================================================
try:
    import bcrypt

    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False
    logger.warning(
        "bcrypt not installed - using PBKDF2 fallback for password hashing. "
        "Install bcrypt for better security: pip install bcrypt"
    )


def _hash_password(password: str) -> str:
    """
    Securely hash a password using bcrypt (preferred) or PBKDF2 (fallback).

    Returns a string that includes the algorithm identifier for future-proofing.
    """
    if BCRYPT_AVAILABLE:
        # bcrypt with work factor 12 (recommended minimum)
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
        return f"bcrypt:{hashed.decode('utf-8')}"
    else:
        # PBKDF2-SHA256 fallback with 600k iterations (OWASP 2023 recommendation)
        salt = secrets.token_hex(16)
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt.encode("utf-8"), 600000
        )
        return f"pbkdf2:{salt}:{dk.hex()}"


def _verify_password(password: str, password_hash: str) -> bool:
    """
    Verify a password against its hash.

    Handles both bcrypt and PBKDF2 hashes, plus legacy SHA-256 hashes.
    """
    if not password_hash:
        return False

    # Handle new format with algorithm prefix
    if password_hash.startswith("bcrypt:"):
        if not BCRYPT_AVAILABLE:
            logger.error("bcrypt hash found but bcrypt not installed")
            return False
        stored_hash = password_hash[7:].encode("utf-8")
        return bcrypt.checkpw(password.encode("utf-8"), stored_hash)

    elif password_hash.startswith("pbkdf2:"):
        parts = password_hash.split(":")
        if len(parts) != 3:
            return False
        _, salt, stored_dk = parts
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt.encode("utf-8"), 600000
        )
        return secrets.compare_digest(dk.hex(), stored_dk)

    else:
        # Legacy SHA-256 hash (for backwards compatibility)
        # WARNING: This is insecure - sessions should be re-created
        legacy_hash = hashlib.sha256(password.encode()).hexdigest()
        return secrets.compare_digest(legacy_hash, password_hash)

# ============================================================================
# Collaboration Data Types
# ============================================================================


class UserRole(Enum):
    """Roles in a collaborative session."""

    OWNER = "owner"
    COACH = "coach"
    ANALYST = "analyst"
    PLAYER = "player"
    VIEWER = "viewer"


class AnnotationType(Enum):
    """Types of demo annotations."""

    COMMENT = "comment"
    DRAWING = "drawing"
    HIGHLIGHT = "highlight"
    MARKER = "marker"
    TAG = "tag"
    VOICE_NOTE = "voice_note"
    SCREENSHOT = "screenshot"


class AnnotationCategory(Enum):
    """Categories for annotations."""

    POSITIONING = "positioning"
    TIMING = "timing"
    UTILITY = "utility"
    AIM = "aim"
    COMMUNICATION = "communication"
    ECONOMY = "economy"
    STRATEGY = "strategy"
    MISTAKE = "mistake"
    GOOD_PLAY = "good_play"
    QUESTION = "question"
    OTHER = "other"


class CollaborationEvent(Enum):
    """Types of collaboration events."""

    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"
    ANNOTATION_ADDED = "annotation_added"
    ANNOTATION_UPDATED = "annotation_updated"
    ANNOTATION_DELETED = "annotation_deleted"
    PLAYBACK_SYNC = "playback_sync"
    CURSOR_UPDATE = "cursor_update"
    CHAT_MESSAGE = "chat_message"
    SESSION_UPDATED = "session_updated"


# ============================================================================
# User and Session Data Structures
# ============================================================================


@dataclass
class CollaborationUser:
    """A user in a collaborative session."""

    user_id: str
    username: str
    role: UserRole
    avatar_color: str = "#3498db"
    joined_at: str = ""
    is_online: bool = True

    # Cursor state for real-time sync
    cursor_tick: int = 0
    cursor_x: float = 0.0
    cursor_y: float = 0.0

    # Permissions
    can_annotate: bool = True
    can_control_playback: bool = False
    can_invite: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "role": self.role.value,
            "avatar_color": self.avatar_color,
            "joined_at": self.joined_at,
            "is_online": self.is_online,
            "cursor": {"tick": self.cursor_tick, "x": self.cursor_x, "y": self.cursor_y},
            "permissions": {
                "can_annotate": self.can_annotate,
                "can_control_playback": self.can_control_playback,
                "can_invite": self.can_invite,
            },
        }


@dataclass
class Annotation:
    """A single annotation on the demo."""

    annotation_id: str
    annotation_type: AnnotationType
    category: AnnotationCategory
    author_id: str
    author_name: str
    created_at: str
    updated_at: str

    # Position/timing
    tick: int
    round_num: int
    timestamp_ms: float

    # Content
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    # Target
    target_player: str | None = None  # Steam ID
    position: tuple[float, float, float] | None = None

    # Drawing data (for drawing annotations)
    drawing_points: list[tuple[float, float]] = field(default_factory=list)
    drawing_color: str = "#ff0000"
    drawing_thickness: int = 2

    # Metadata
    is_private: bool = False
    tags: list[str] = field(default_factory=list)
    replies: list[dict[str, Any]] = field(default_factory=list)
    reactions: dict[str, list[str]] = field(default_factory=dict)  # emoji -> user_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "annotation_id": self.annotation_id,
            "type": self.annotation_type.value,
            "category": self.category.value,
            "author": {"id": self.author_id, "name": self.author_name},
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "timing": {
                "tick": self.tick,
                "round": self.round_num,
                "timestamp_ms": self.timestamp_ms,
            },
            "content": {"text": self.text, "data": self.data},
            "target": {"player": self.target_player, "position": self.position},
            "drawing": (
                {
                    "points": self.drawing_points,
                    "color": self.drawing_color,
                    "thickness": self.drawing_thickness,
                }
                if self.annotation_type == AnnotationType.DRAWING
                else None
            ),
            "metadata": {
                "is_private": self.is_private,
                "tags": self.tags,
                "replies": self.replies,
                "reactions": self.reactions,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Annotation":
        timing = data.get("timing", {})
        content = data.get("content", {})
        target = data.get("target", {})
        drawing = data.get("drawing", {})
        metadata = data.get("metadata", {})
        author = data.get("author", {})

        return cls(
            annotation_id=data.get("annotation_id", ""),
            annotation_type=AnnotationType(data.get("type", "comment")),
            category=AnnotationCategory(data.get("category", "other")),
            author_id=author.get("id", ""),
            author_name=author.get("name", "Unknown"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            tick=timing.get("tick", 0),
            round_num=timing.get("round", 0),
            timestamp_ms=timing.get("timestamp_ms", 0),
            text=content.get("text", ""),
            data=content.get("data", {}),
            target_player=target.get("player"),
            position=tuple(target["position"]) if target.get("position") else None,
            drawing_points=drawing.get("points", []) if drawing else [],
            drawing_color=drawing.get("color", "#ff0000") if drawing else "#ff0000",
            drawing_thickness=drawing.get("thickness", 2) if drawing else 2,
            is_private=metadata.get("is_private", False),
            tags=metadata.get("tags", []),
            replies=metadata.get("replies", []),
            reactions=metadata.get("reactions", {}),
        )


@dataclass
class PlaybackState:
    """Synchronized playback state."""

    current_tick: int = 0
    is_playing: bool = False
    playback_speed: float = 1.0
    controlled_by: str | None = None  # user_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_tick": self.current_tick,
            "is_playing": self.is_playing,
            "playback_speed": self.playback_speed,
            "controlled_by": self.controlled_by,
        }


@dataclass
class ChatMessage:
    """A chat message in the session."""

    message_id: str
    user_id: str
    username: str
    text: str
    timestamp: str
    reply_to: str | None = None
    mentions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "user_id": self.user_id,
            "username": self.username,
            "text": self.text,
            "timestamp": self.timestamp,
            "reply_to": self.reply_to,
            "mentions": self.mentions,
        }


@dataclass
class CollaborationSession:
    """A collaborative demo review session."""

    session_id: str
    demo_id: str
    demo_name: str
    map_name: str
    created_at: str
    created_by: str

    # Session settings
    title: str = ""
    description: str = ""
    is_public: bool = False
    password_hash: str | None = None
    max_users: int = 10

    # Participants
    users: dict[str, CollaborationUser] = field(default_factory=dict)
    invited_users: list[str] = field(default_factory=list)

    # Annotations
    annotations: dict[str, Annotation] = field(default_factory=dict)

    # Playback state
    playback: PlaybackState = field(default_factory=PlaybackState)

    # Chat
    chat_history: list[ChatMessage] = field(default_factory=list)

    # Activity tracking
    last_activity: str = ""
    total_annotations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "demo_id": self.demo_id,
            "demo_name": self.demo_name,
            "map_name": self.map_name,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "title": self.title,
            "description": self.description,
            "settings": {
                "is_public": self.is_public,
                "has_password": self.password_hash is not None,
                "max_users": self.max_users,
            },
            "users": {uid: u.to_dict() for uid, u in self.users.items()},
            "online_count": sum(1 for u in self.users.values() if u.is_online),
            "total_annotations": len(self.annotations),
            "playback": self.playback.to_dict(),
            "last_activity": self.last_activity,
        }


# ============================================================================
# Event System
# ============================================================================


@dataclass
class SessionEvent:
    """An event in a collaboration session."""

    event_id: str
    event_type: CollaborationEvent
    session_id: str
    user_id: str
    timestamp: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "type": self.event_type.value,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "data": self.data,
        }


class EventBus:
    """Event bus for real-time collaboration events."""

    def __init__(self):
        self.subscribers: dict[str, list[queue.Queue]] = defaultdict(list)
        self._lock = threading.Lock()
        self._event_counter = 0

    def subscribe(self, session_id: str) -> queue.Queue:
        """Subscribe to events for a session."""
        event_queue = queue.Queue()
        with self._lock:
            self.subscribers[session_id].append(event_queue)
        return event_queue

    def unsubscribe(self, session_id: str, event_queue: queue.Queue) -> None:
        """Unsubscribe from session events."""
        with self._lock:
            if event_queue in self.subscribers[session_id]:
                self.subscribers[session_id].remove(event_queue)

    def publish(self, event: SessionEvent) -> None:
        """Publish an event to all subscribers."""
        with self._lock:
            self._event_counter += 1
            event.event_id = f"evt_{self._event_counter}_{int(time.time() * 1000)}"

            for q in self.subscribers.get(event.session_id, []):
                try:
                    q.put_nowait(event)
                except queue.Full:
                    pass

    def get_event(self, event_queue: queue.Queue, timeout: float = 1.0) -> SessionEvent | None:
        """Get the next event from a subscription."""
        try:
            return event_queue.get(timeout=timeout)
        except queue.Empty:
            return None


# ============================================================================
# Collaboration Manager
# ============================================================================


class CollaborationManager:
    """
    Manages collaborative analysis sessions.
    """

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or Path.home() / ".opensight" / "collaboration"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.sessions: dict[str, CollaborationSession] = {}
        self.event_bus = EventBus()
        self._session_counter = 0

        self._load_sessions()

    def _load_sessions(self) -> None:
        """Load saved sessions from disk."""
        for session_file in self.data_dir.glob("session_*.json"):
            try:
                with open(session_file) as f:
                    data = json.load(f)
                session = self._session_from_dict(data)
                self.sessions[session.session_id] = session
            except (OSError, json.JSONDecodeError):
                continue

    def _save_session(self, session: CollaborationSession) -> None:
        """Save session to disk."""
        session_file = self.data_dir / f"session_{session.session_id}.json"
        try:
            with open(session_file, "w") as f:
                json.dump(self._session_to_dict(session), f, indent=2)
        except OSError:
            pass

    def _session_to_dict(self, session: CollaborationSession) -> dict[str, Any]:
        """Convert session to full dictionary for storage."""
        return {
            **session.to_dict(),
            "annotations": {aid: a.to_dict() for aid, a in session.annotations.items()},
            "chat_history": [m.to_dict() for m in session.chat_history],
            "invited_users": session.invited_users,
            "password_hash": session.password_hash,
        }

    def _session_from_dict(self, data: dict[str, Any]) -> CollaborationSession:
        """Reconstruct session from dictionary."""
        settings = data.get("settings", {})

        session = CollaborationSession(
            session_id=data.get("session_id", ""),
            demo_id=data.get("demo_id", ""),
            demo_name=data.get("demo_name", ""),
            map_name=data.get("map_name", ""),
            created_at=data.get("created_at", ""),
            created_by=data.get("created_by", ""),
            title=data.get("title", ""),
            description=data.get("description", ""),
            is_public=settings.get("is_public", False),
            password_hash=data.get("password_hash"),
            max_users=settings.get("max_users", 10),
            invited_users=data.get("invited_users", []),
            last_activity=data.get("last_activity", ""),
        )

        # Reconstruct users (but mark as offline)
        for uid, u_data in data.get("users", {}).items():
            perms = u_data.get("permissions", {})
            session.users[uid] = CollaborationUser(
                user_id=uid,
                username=u_data.get("username", ""),
                role=UserRole(u_data.get("role", "viewer")),
                avatar_color=u_data.get("avatar_color", "#3498db"),
                joined_at=u_data.get("joined_at", ""),
                is_online=False,  # Reset online status
                can_annotate=perms.get("can_annotate", True),
                can_control_playback=perms.get("can_control_playback", False),
                can_invite=perms.get("can_invite", False),
            )

        # Reconstruct annotations
        for aid, a_data in data.get("annotations", {}).items():
            session.annotations[aid] = Annotation.from_dict(a_data)

        # Reconstruct chat
        for m_data in data.get("chat_history", []):
            session.chat_history.append(
                ChatMessage(
                    message_id=m_data.get("message_id", ""),
                    user_id=m_data.get("user_id", ""),
                    username=m_data.get("username", ""),
                    text=m_data.get("text", ""),
                    timestamp=m_data.get("timestamp", ""),
                    reply_to=m_data.get("reply_to"),
                    mentions=m_data.get("mentions", []),
                )
            )

        return session

    def create_session(
        self,
        demo_id: str,
        demo_name: str,
        map_name: str,
        creator_id: str,
        creator_name: str,
        title: str = "",
        description: str = "",
        is_public: bool = False,
        password: str | None = None,
    ) -> CollaborationSession:
        """
        Create a new collaboration session.

        Args:
            demo_id: Demo identifier
            demo_name: Demo file name
            map_name: Map name
            creator_id: Creator's user ID
            creator_name: Creator's display name
            title: Session title
            description: Session description
            is_public: Whether session is publicly joinable
            password: Optional password for access

        Returns:
            Created session
        """
        self._session_counter += 1
        session_id = hashlib.md5(
            f"{demo_id}_{creator_id}_{datetime.now().isoformat()}".encode(),
            usedforsecurity=False,
        ).hexdigest()[:12]

        password_hash = None
        if password:
            password_hash = _hash_password(password)

        session = CollaborationSession(
            session_id=session_id,
            demo_id=demo_id,
            demo_name=demo_name,
            map_name=map_name,
            created_at=datetime.now().isoformat(),
            created_by=creator_id,
            title=title or f"Review: {demo_name}",
            description=description,
            is_public=is_public,
            password_hash=password_hash,
        )

        # Add creator as owner
        owner = CollaborationUser(
            user_id=creator_id,
            username=creator_name,
            role=UserRole.OWNER,
            avatar_color=self._generate_avatar_color(creator_id),
            joined_at=datetime.now().isoformat(),
            is_online=True,
            can_annotate=True,
            can_control_playback=True,
            can_invite=True,
        )
        session.users[creator_id] = owner
        session.last_activity = datetime.now().isoformat()

        self.sessions[session_id] = session
        self._save_session(session)

        return session

    def join_session(
        self,
        session_id: str,
        user_id: str,
        username: str,
        password: str | None = None,
        role: UserRole = UserRole.VIEWER,
    ) -> tuple[CollaborationSession | None, str | None]:
        """
        Join an existing session.

        Args:
            session_id: Session to join
            user_id: User's ID
            username: User's display name
            password: Password if required
            role: Requested role

        Returns:
            Tuple of (session or None, error message or None)
        """
        session = self.sessions.get(session_id)
        if not session:
            return None, "Session not found"

        # Check password
        if session.password_hash:
            if not password:
                return None, "Password required"
            if not _verify_password(password, session.password_hash):
                return None, "Invalid password"

        # Check user limit
        online_users = sum(1 for u in session.users.values() if u.is_online)
        if online_users >= session.max_users:
            return None, "Session is full"

        # Check if user already in session
        if user_id in session.users:
            # Re-join
            session.users[user_id].is_online = True
        else:
            # New user
            user = CollaborationUser(
                user_id=user_id,
                username=username,
                role=role,
                avatar_color=self._generate_avatar_color(user_id),
                joined_at=datetime.now().isoformat(),
                is_online=True,
                can_annotate=role
                in [UserRole.OWNER, UserRole.COACH, UserRole.ANALYST, UserRole.PLAYER],
                can_control_playback=role in [UserRole.OWNER, UserRole.COACH],
                can_invite=role in [UserRole.OWNER, UserRole.COACH],
            )
            session.users[user_id] = user

        session.last_activity = datetime.now().isoformat()
        self._save_session(session)

        # Publish event
        self.event_bus.publish(
            SessionEvent(
                event_id="",
                event_type=CollaborationEvent.USER_JOINED,
                session_id=session_id,
                user_id=user_id,
                timestamp=datetime.now().isoformat(),
                data={"username": username, "role": role.value},
            )
        )

        return session, None

    def leave_session(self, session_id: str, user_id: str) -> bool:
        """Leave a session."""
        session = self.sessions.get(session_id)
        if not session or user_id not in session.users:
            return False

        session.users[user_id].is_online = False
        session.last_activity = datetime.now().isoformat()
        self._save_session(session)

        # Publish event
        self.event_bus.publish(
            SessionEvent(
                event_id="",
                event_type=CollaborationEvent.USER_LEFT,
                session_id=session_id,
                user_id=user_id,
                timestamp=datetime.now().isoformat(),
                data={"username": session.users[user_id].username},
            )
        )

        return True

    def add_annotation(
        self,
        session_id: str,
        user_id: str,
        annotation_type: AnnotationType,
        category: AnnotationCategory,
        tick: int,
        round_num: int,
        text: str = "",
        target_player: str | None = None,
        position: tuple[float, float, float] | None = None,
        drawing_data: dict[str, Any] | None = None,
        tags: list[str] = None,
        is_private: bool = False,
    ) -> Annotation | None:
        """
        Add an annotation to the session.

        Args:
            session_id: Session ID
            user_id: User adding the annotation
            annotation_type: Type of annotation
            category: Annotation category
            tick: Demo tick
            round_num: Round number
            text: Annotation text
            target_player: Target player Steam ID
            position: Position in map
            drawing_data: Drawing data if type is DRAWING
            tags: Tags for the annotation
            is_private: Whether annotation is private

        Returns:
            Created annotation or None
        """
        session = self.sessions.get(session_id)
        if not session:
            return None

        user = session.users.get(user_id)
        if not user or not user.can_annotate:
            return None

        annotation_id = hashlib.md5(
            f"{session_id}_{user_id}_{tick}_{datetime.now().isoformat()}".encode(),
            usedforsecurity=False,
        ).hexdigest()[:10]

        annotation = Annotation(
            annotation_id=annotation_id,
            annotation_type=annotation_type,
            category=category,
            author_id=user_id,
            author_name=user.username,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            tick=tick,
            round_num=round_num,
            timestamp_ms=tick * 15.625,
            text=text,
            target_player=target_player,
            position=position,
            is_private=is_private,
            tags=tags or [],
        )

        if drawing_data and annotation_type == AnnotationType.DRAWING:
            annotation.drawing_points = drawing_data.get("points", [])
            annotation.drawing_color = drawing_data.get("color", "#ff0000")
            annotation.drawing_thickness = drawing_data.get("thickness", 2)

        session.annotations[annotation_id] = annotation
        session.total_annotations += 1
        session.last_activity = datetime.now().isoformat()
        self._save_session(session)

        # Publish event
        self.event_bus.publish(
            SessionEvent(
                event_id="",
                event_type=CollaborationEvent.ANNOTATION_ADDED,
                session_id=session_id,
                user_id=user_id,
                timestamp=datetime.now().isoformat(),
                data=annotation.to_dict(),
            )
        )

        return annotation

    def update_annotation(
        self, session_id: str, user_id: str, annotation_id: str, updates: dict[str, Any]
    ) -> Annotation | None:
        """Update an existing annotation."""
        session = self.sessions.get(session_id)
        if not session:
            return None

        annotation = session.annotations.get(annotation_id)
        if not annotation:
            return None

        # Only author or owner can edit
        user = session.users.get(user_id)
        if not user:
            return None
        if annotation.author_id != user_id and user.role != UserRole.OWNER:
            return None

        # Apply updates
        for key, value in updates.items():
            if hasattr(annotation, key) and key not in ["annotation_id", "author_id", "created_at"]:
                setattr(annotation, key, value)

        annotation.updated_at = datetime.now().isoformat()
        session.last_activity = datetime.now().isoformat()
        self._save_session(session)

        # Publish event
        self.event_bus.publish(
            SessionEvent(
                event_id="",
                event_type=CollaborationEvent.ANNOTATION_UPDATED,
                session_id=session_id,
                user_id=user_id,
                timestamp=datetime.now().isoformat(),
                data={"annotation_id": annotation_id, "updates": updates},
            )
        )

        return annotation

    def delete_annotation(self, session_id: str, user_id: str, annotation_id: str) -> bool:
        """Delete an annotation."""
        session = self.sessions.get(session_id)
        if not session:
            return False

        annotation = session.annotations.get(annotation_id)
        if not annotation:
            return False

        # Only author or owner can delete
        user = session.users.get(user_id)
        if not user:
            return False
        if annotation.author_id != user_id and user.role != UserRole.OWNER:
            return False

        del session.annotations[annotation_id]
        session.last_activity = datetime.now().isoformat()
        self._save_session(session)

        # Publish event
        self.event_bus.publish(
            SessionEvent(
                event_id="",
                event_type=CollaborationEvent.ANNOTATION_DELETED,
                session_id=session_id,
                user_id=user_id,
                timestamp=datetime.now().isoformat(),
                data={"annotation_id": annotation_id},
            )
        )

        return True

    def add_reply(
        self, session_id: str, user_id: str, annotation_id: str, text: str
    ) -> dict[str, Any] | None:
        """Add a reply to an annotation."""
        session = self.sessions.get(session_id)
        if not session:
            return None

        annotation = session.annotations.get(annotation_id)
        if not annotation:
            return None

        user = session.users.get(user_id)
        if not user:
            return None

        reply = {
            "reply_id": hashlib.md5(
                f"{annotation_id}_{user_id}_{time.time()}".encode(),
                usedforsecurity=False,
            ).hexdigest()[:8],
            "user_id": user_id,
            "username": user.username,
            "text": text,
            "timestamp": datetime.now().isoformat(),
        }

        annotation.replies.append(reply)
        annotation.updated_at = datetime.now().isoformat()
        self._save_session(session)

        return reply

    def add_reaction(self, session_id: str, user_id: str, annotation_id: str, emoji: str) -> bool:
        """Add a reaction to an annotation."""
        session = self.sessions.get(session_id)
        if not session:
            return False

        annotation = session.annotations.get(annotation_id)
        if not annotation:
            return False

        if emoji not in annotation.reactions:
            annotation.reactions[emoji] = []

        if user_id not in annotation.reactions[emoji]:
            annotation.reactions[emoji].append(user_id)
            self._save_session(session)

        return True

    def send_chat_message(
        self, session_id: str, user_id: str, text: str, reply_to: str | None = None
    ) -> ChatMessage | None:
        """Send a chat message in the session."""
        session = self.sessions.get(session_id)
        if not session or user_id not in session.users:
            return None

        user = session.users[user_id]
        if not user.is_online:
            return None

        # Find mentions
        mentions = re.findall(r"@(\w+)", text)

        message = ChatMessage(
            message_id=hashlib.md5(
                f"{session_id}_{user_id}_{time.time()}".encode(), usedforsecurity=False
            ).hexdigest()[:10],
            user_id=user_id,
            username=user.username,
            text=text,
            timestamp=datetime.now().isoformat(),
            reply_to=reply_to,
            mentions=mentions,
        )

        session.chat_history.append(message)
        session.last_activity = datetime.now().isoformat()

        # Keep only last 500 messages
        if len(session.chat_history) > 500:
            session.chat_history = session.chat_history[-500:]

        self._save_session(session)

        # Publish event
        self.event_bus.publish(
            SessionEvent(
                event_id="",
                event_type=CollaborationEvent.CHAT_MESSAGE,
                session_id=session_id,
                user_id=user_id,
                timestamp=datetime.now().isoformat(),
                data=message.to_dict(),
            )
        )

        return message

    def update_playback(
        self,
        session_id: str,
        user_id: str,
        tick: int | None = None,
        is_playing: bool | None = None,
        speed: float | None = None,
    ) -> bool:
        """Update synchronized playback state."""
        session = self.sessions.get(session_id)
        if not session:
            return False

        user = session.users.get(user_id)
        if not user or not user.can_control_playback:
            return False

        if tick is not None:
            session.playback.current_tick = tick
        if is_playing is not None:
            session.playback.is_playing = is_playing
        if speed is not None:
            session.playback.playback_speed = speed

        session.playback.controlled_by = user_id
        session.last_activity = datetime.now().isoformat()

        # Publish event
        self.event_bus.publish(
            SessionEvent(
                event_id="",
                event_type=CollaborationEvent.PLAYBACK_SYNC,
                session_id=session_id,
                user_id=user_id,
                timestamp=datetime.now().isoformat(),
                data=session.playback.to_dict(),
            )
        )

        return True

    def update_cursor(self, session_id: str, user_id: str, tick: int, x: float, y: float) -> bool:
        """Update user's cursor position for collaboration."""
        session = self.sessions.get(session_id)
        if not session or user_id not in session.users:
            return False

        user = session.users[user_id]
        user.cursor_tick = tick
        user.cursor_x = x
        user.cursor_y = y

        # Publish event (but don't save to disk - too frequent)
        self.event_bus.publish(
            SessionEvent(
                event_id="",
                event_type=CollaborationEvent.CURSOR_UPDATE,
                session_id=session_id,
                user_id=user_id,
                timestamp=datetime.now().isoformat(),
                data={"tick": tick, "x": x, "y": y},
            )
        )

        return True

    def get_session(self, session_id: str) -> CollaborationSession | None:
        """Get a session by ID."""
        return self.sessions.get(session_id)

    def get_annotations(
        self,
        session_id: str,
        user_id: str,
        round_num: int | None = None,
        category: AnnotationCategory | None = None,
        include_private: bool = False,
    ) -> list[Annotation]:
        """Get annotations from a session with optional filters."""
        session = self.sessions.get(session_id)
        if not session:
            return []

        annotations = list(session.annotations.values())

        # Filter private (unless author or include_private)
        if not include_private:
            annotations = [a for a in annotations if not a.is_private or a.author_id == user_id]

        # Filter by round
        if round_num is not None:
            annotations = [a for a in annotations if a.round_num == round_num]

        # Filter by category
        if category is not None:
            annotations = [a for a in annotations if a.category == category]

        return sorted(annotations, key=lambda a: a.tick)

    def list_sessions(
        self, user_id: str | None = None, include_public: bool = True
    ) -> list[dict[str, Any]]:
        """List available sessions."""
        result = []

        for session in self.sessions.values():
            # Include if user is participant
            if user_id and user_id in session.users:
                result.append(session.to_dict())
            # Include if public
            elif include_public and session.is_public:
                result.append(session.to_dict())

        return sorted(result, key=lambda s: s["last_activity"], reverse=True)

    def _generate_avatar_color(self, user_id: str) -> str:
        """Generate a consistent avatar color for a user."""
        colors = [
            "#e74c3c",
            "#3498db",
            "#2ecc71",
            "#9b59b6",
            "#f39c12",
            "#1abc9c",
            "#e67e22",
            "#34495e",
        ]
        hash_val = int(hashlib.md5(user_id.encode(), usedforsecurity=False).hexdigest()[:8], 16)
        return colors[hash_val % len(colors)]

    def export_annotations(self, session_id: str, format: str = "json") -> str | None:
        """Export session annotations."""
        session = self.sessions.get(session_id)
        if not session:
            return None

        annotations = [a.to_dict() for a in session.annotations.values()]

        if format == "json":
            return json.dumps(
                {
                    "session_id": session_id,
                    "demo_id": session.demo_id,
                    "demo_name": session.demo_name,
                    "map_name": session.map_name,
                    "exported_at": datetime.now().isoformat(),
                    "annotations": annotations,
                },
                indent=2,
            )

        elif format == "markdown":
            md = [f"# Demo Review: {session.demo_name}"]
            md.append(f"\n**Map:** {session.map_name}")
            md.append(f"\n**Annotations:** {len(annotations)}\n")

            by_round = defaultdict(list)
            for a in annotations:
                by_round[a["timing"]["round"]].append(a)

            for round_num in sorted(by_round.keys()):
                md.append(f"\n## Round {round_num}")
                for a in by_round[round_num]:
                    author = a["author"]["name"]
                    text = a["content"]["text"]
                    category = a["category"]
                    md.append(f"\n- **[{category}]** {text} _{author}_")

            return "\n".join(md)

        return None


# ============================================================================
# Convenience Functions
# ============================================================================

_default_manager: CollaborationManager | None = None


def get_manager() -> CollaborationManager:
    """Get or create the default collaboration manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = CollaborationManager()
    return _default_manager


def create_collaboration_session(
    demo_id: str, demo_name: str, map_name: str, creator_id: str, creator_name: str, **kwargs
) -> dict[str, Any]:
    """Create a new collaboration session."""
    manager = get_manager()
    session = manager.create_session(
        demo_id, demo_name, map_name, creator_id, creator_name, **kwargs
    )
    return session.to_dict()


def join_collaboration_session(
    session_id: str, user_id: str, username: str, password: str | None = None
) -> dict[str, Any]:
    """Join an existing collaboration session."""
    manager = get_manager()
    session, error = manager.join_session(session_id, user_id, username, password)
    if error:
        return {"error": error}
    return session.to_dict()


def add_annotation(session_id: str, user_id: str, **kwargs) -> dict[str, Any]:
    """Add an annotation to a session."""
    manager = get_manager()

    # Convert string enums
    if "annotation_type" in kwargs and isinstance(kwargs["annotation_type"], str):
        kwargs["annotation_type"] = AnnotationType(kwargs["annotation_type"])
    if "category" in kwargs and isinstance(kwargs["category"], str):
        kwargs["category"] = AnnotationCategory(kwargs["category"])

    annotation = manager.add_annotation(session_id, user_id, **kwargs)
    if annotation:
        return annotation.to_dict()
    return {"error": "Failed to add annotation"}


def get_session_annotations(
    session_id: str, user_id: str, round_num: int | None = None
) -> list[dict[str, Any]]:
    """Get annotations from a session."""
    manager = get_manager()
    annotations = manager.get_annotations(session_id, user_id, round_num)
    return [a.to_dict() for a in annotations]


def list_sessions(user_id: str | None = None) -> list[dict[str, Any]]:
    """List available collaboration sessions."""
    return get_manager().list_sessions(user_id)


def export_session(session_id: str, format: str = "json") -> str | None:
    """Export session annotations."""
    return get_manager().export_annotations(session_id, format)
