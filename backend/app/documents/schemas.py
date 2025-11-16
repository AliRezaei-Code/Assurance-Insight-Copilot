"""Pydantic schemas for documents."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict


class DocumentStatus(str, Enum):
    """Workflow states for document ingestion."""

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class DocumentBase(BaseModel):
    id: int
    file_name: str
    status: DocumentStatus
    created_at: datetime
    engagement_type: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class DocumentSummary(DocumentBase):
    owner_user_id: int


class DocumentDetail(DocumentSummary):
    blob_url: str


class DocumentStatusUpdate(BaseModel):
    status: DocumentStatus
