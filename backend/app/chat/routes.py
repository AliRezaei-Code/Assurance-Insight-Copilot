"""Chat session and messaging routes."""
from __future__ import annotations

from datetime import datetime
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth import models as auth_models
from app.chat import models, schemas, rag_pipeline
from app.dependencies import get_current_user, get_db

router = APIRouter()
logger = logging.getLogger(__name__)
pipeline = rag_pipeline.RAGPipeline()


@router.post("/sessions", response_model=schemas.ChatSessionRead, status_code=status.HTTP_201_CREATED)
async def create_session(
    payload: schemas.ChatSessionCreate,
    current_user: auth_models.User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> schemas.ChatSessionRead:
    """Create a new chat session for the authenticated user."""
    title = payload.title or f"Client insights {datetime.utcnow():%Y-%m-%d %H:%M}"
    chat_session = models.ChatSession(owner_user_id=current_user.id, title=title)
    session.add(chat_session)
    await session.commit()
    await session.refresh(chat_session)
    return schemas.ChatSessionRead.model_validate(chat_session)


@router.get("/sessions", response_model=List[schemas.ChatSessionRead])
async def list_sessions(
    current_user: auth_models.User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> List[schemas.ChatSessionRead]:
    """Return chat sessions owned by the current user."""
    statement = (
        select(models.ChatSession)
        .where(models.ChatSession.owner_user_id == current_user.id)
        .order_by(models.ChatSession.created_at.desc())
    )
    sessions = (await session.scalars(statement)).all()
    return [schemas.ChatSessionRead.model_validate(chat) for chat in sessions]


@router.get("/sessions/{session_id}", response_model=schemas.ChatSessionWithMessages)
async def get_session_detail(
    session_id: int,
    current_user: auth_models.User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> schemas.ChatSessionWithMessages:
    """Return a session plus its ordered message history."""
    statement = (
        select(models.ChatSession)
        .options(selectinload(models.ChatSession.messages))
        .where(
            models.ChatSession.id == session_id,
            models.ChatSession.owner_user_id == current_user.id,
        )
    )
    chat_session = await session.scalar(statement)
    if chat_session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found")
    ordered_messages = sorted(chat_session.messages, key=lambda msg: msg.created_at)
    return schemas.ChatSessionWithMessages(
        id=chat_session.id,
        title=chat_session.title,
        created_at=chat_session.created_at,
        messages=[schemas.ChatMessageRead.model_validate(msg) for msg in ordered_messages],
    )


@router.post(
    "/sessions/{session_id}/messages",
    response_model=schemas.ChatCompletionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def send_message(
    session_id: int,
    payload: schemas.ChatMessageCreate,
    current_user: auth_models.User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> schemas.ChatCompletionResponse:
    """Persist a user question, run the RAG pipeline, and store the reply."""
    message_text = payload.message.strip()
    if not message_text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Message cannot be empty")

    chat_session = await _get_session_for_user(session, session_id, current_user.id)

    user_message = models.ChatMessage(session_id=chat_session.id, role="user", content=message_text)
    session.add(user_message)
    await session.flush()

    try:
        rag_result = await pipeline.run(question=message_text, user=current_user, session=session)
    except Exception as exc:  # pragma: no cover - integration heavy
        await session.rollback()
        logger.exception("RAG pipeline execution failed", exc_info=exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Assistant was unable to generate a response",
        ) from exc

    assistant_message = models.ChatMessage(
        session_id=chat_session.id,
        role="assistant",
        content=rag_result.answer,
        raw_citations=[citation.model_dump() for citation in rag_result.citations],
        risk_highlights=rag_result.risk_highlights,
    )
    session.add(assistant_message)
    await session.commit()
    await session.refresh(assistant_message)

    logger.info("Generated assistant response for session %s", chat_session.id)

    return schemas.ChatCompletionResponse(
        assistant_message=schemas.ChatMessageRead.model_validate(assistant_message),
        citations=rag_result.citations,
        risk_highlights=rag_result.risk_highlights,
    )


async def _get_session_for_user(
    session: AsyncSession,
    session_id: int,
    user_id: int,
) -> models.ChatSession:
    """Fetch a chat session owned by the user or raise 404."""
    statement = select(models.ChatSession).where(
        models.ChatSession.id == session_id,
        models.ChatSession.owner_user_id == user_id,
    )
    chat_session = await session.scalar(statement)
    if chat_session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found")
    return chat_session
