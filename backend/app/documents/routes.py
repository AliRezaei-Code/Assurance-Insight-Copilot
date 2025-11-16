"""Document management routes."""
from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import models as auth_models
from app.documents import ingestion, models, schemas
from app.dependencies import get_current_user, get_db

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload", response_model=schemas.DocumentDetail, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    current_user: auth_models.User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> schemas.DocumentDetail:
    """Handle document upload, ingestion, and persistence of metadata."""
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Filename is required")

    temp_path = await ingestion.persist_upload_to_disk(file)
    document = models.Document(
        owner_user_id=current_user.id,
        file_name=file.filename,
        blob_url="",
        status=schemas.DocumentStatus.PROCESSING.value,
    )
    session.add(document)
    await session.flush()

    try:
        document.blob_url = await ingestion.upload_file_to_blob(temp_path, current_user.id)
        await ingestion.ingest_document(session, document, temp_path)
        document.status = schemas.DocumentStatus.INDEXED.value
        await session.commit()
        await session.refresh(document)
    except Exception as exc:  # pragma: no cover - integration heavy
        logger.exception("Document ingestion failed", exc_info=exc)
        await session.rollback()
        document.status = schemas.DocumentStatus.FAILED.value
        session.add(document)
        await session.commit()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Document ingestion failed") from exc
    finally:
        ingestion.cleanup_temp_file(temp_path)

    return schemas.DocumentDetail.model_validate(document)


@router.get("/", response_model=List[schemas.DocumentSummary])
async def list_documents(
    current_user: auth_models.User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> List[schemas.DocumentSummary]:
    """Return the caller's documents ordered by recency."""
    statement = (
        select(models.Document)
        .where(models.Document.owner_user_id == current_user.id)
        .order_by(models.Document.created_at.desc())
    )
    documents = (await session.scalars(statement)).all()
    return [schemas.DocumentSummary.model_validate(doc) for doc in documents]


@router.get("/{document_id}", response_model=schemas.DocumentDetail)
async def get_document(
    document_id: int,
    current_user: auth_models.User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> schemas.DocumentDetail:
    """Fetch a single document by id if it belongs to the caller."""
    document = await _get_document_or_404(session, document_id, current_user.id)
    return schemas.DocumentDetail.model_validate(document)


async def _get_document_or_404(session: AsyncSession, document_id: int, user_id: int) -> models.Document:
    """Return a document owned by the user or raise 404."""
    statement = select(models.Document).where(
        models.Document.id == document_id, models.Document.owner_user_id == user_id
    )
    document = await session.scalar(statement)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return document
