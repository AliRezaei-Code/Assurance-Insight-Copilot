"""Pydantic schemas for chat endpoints."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    document_id: Optional[int] = None
    file_name: Optional[str] = None
    page_number: Optional[int] = None
    snippet: Optional[str] = None


class ChatSessionCreate(BaseModel):
    title: Optional[str] = None


class ChatSessionRead(BaseModel):
    id: int
    title: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ChatMessageCreate(BaseModel):
    message: str


class ChatMessageRead(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime
    citations: Optional[List[Citation]] = Field(default=None, alias="raw_citations")
    risk_highlights: Optional[List[str]] = Field(default=None)

    model_config = ConfigDict(from_attributes=True)


class ChatSessionWithMessages(ChatSessionRead):
    messages: List[ChatMessageRead]


class ChatCompletionResponse(BaseModel):
    assistant_message: ChatMessageRead
    citations: List[Citation]
    risk_highlights: List[str]
