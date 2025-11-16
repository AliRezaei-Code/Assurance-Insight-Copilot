"""ORM models for chat sessions/messages."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional, TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base

if TYPE_CHECKING:  # pragma: no cover
    from app.auth.models import User


class ChatSession(Base):
    """Top-level conversation container."""

    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False, default="New Session")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    owner: Mapped["User"] = relationship("User", back_populates="chat_sessions")
    messages: Mapped[List["ChatMessage"]] = relationship(
        "ChatMessage", back_populates="session", cascade="all, delete-orphan"
    )


class ChatMessage(Base):
    """Individual message exchanged in a chat session."""

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id"), index=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    raw_citations: Mapped[Optional[list[dict[str, str]]]] = mapped_column(JSON, nullable=True)
    risk_highlights: Mapped[Optional[list[str]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    session: Mapped[ChatSession] = relationship("ChatSession", back_populates="messages")
