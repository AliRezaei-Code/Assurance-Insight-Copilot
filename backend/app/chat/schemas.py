"""Pydantic schemas for chat endpoints."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    """Structured reference to a retrieved document chunk."""
    document_id: Optional[int] = None
    file_name: Optional[str] = None
    page_number: Optional[int] = None
    snippet: Optional[str] = None


class ChatSessionCreate(BaseModel):
    """Payload for starting a new chat session."""
    title: Optional[str] = None


class ChatSessionRead(BaseModel):
    """Lightweight representation of a chat session."""
    id: int
    title: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ChatMessageCreate(BaseModel):
    """Payload containing the user's message text."""
    message: str


class ChatMessageRead(BaseModel):
    """Message record returned to the client."""
    id: int
    role: str
    content: str
    created_at: datetime
    citations: Optional[List[Citation]] = Field(default=None, alias="raw_citations")
    risk_highlights: Optional[List[str]] = Field(default=None)

    model_config = ConfigDict(from_attributes=True)


class ChatSessionWithMessages(ChatSessionRead):
    """Session including its ordered messages."""
    messages: List[ChatMessageRead]


class ChatCompletionResponse(BaseModel):
    """Response returned when the assistant answers a question."""
    assistant_message: ChatMessageRead
    citations: List[Citation]
    risk_highlights: List[str]
