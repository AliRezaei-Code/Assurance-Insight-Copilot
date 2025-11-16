"""RAG pipeline implementation for Assurance Insight Copilot."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import models as auth_models
from app.chat import schemas as chat_schemas
from app.config import settings
from app.documents import models as document_models

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from openai import AsyncAzureOpenAI
except ImportError:  # pragma: no cover
    AsyncAzureOpenAI = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
except ImportError:  # pragma: no cover
    AzureKeyCredential = None  # type: ignore
    SearchClient = None  # type: ignore


@dataclass
class RetrievedChunk:
    """Container for retrieved chunk metadata and content."""
    chunk_id: str
    document_id: int
    file_name: str
    content: str
    page_number: Optional[int]


@dataclass
class PipelineResult:
    """Structured output from the RAG pipeline."""
    answer: str
    citations: List[chat_schemas.Citation]
    risk_highlights: List[str]
    retrieved_chunks: List[RetrievedChunk]


class AzureContentSafetyClient:
    """Stubbed client for Azure Content Safety integration."""

    def __init__(self) -> None:
        """Record whether remote safety checks should run."""
        self.enabled = settings.environment != "local"

    def validate_user_input(self, text: str) -> None:
        """Validate the inbound prompt for emptiness or disallowed content."""
        if not text.strip():
            raise ValueError("Question cannot be empty")

    def sanitize_model_output(self, text: str) -> str:
        """Scrub the generated answer before returning it to callers."""
        return text.strip()


class AzureOpenAIChatClient:
    """Wrapper for Azure OpenAI chat completions with structured JSON responses."""

    def __init__(self) -> None:
        """Instantiate the Azure OpenAI client or fallback mock."""
        self._client = None
        can_use_remote = (
            AsyncAzureOpenAI is not None
            and settings.environment != "local"
            and settings.azure_openai_api_key not in {"", "changeme"}
        )
        if can_use_remote:
            try:  # pragma: no cover - network required
                self._client = AsyncAzureOpenAI(
                    api_key=settings.azure_openai_api_key,
                    api_version="2024-05-01-preview",
                    azure_endpoint=settings.azure_openai_endpoint,
                )
            except Exception as exc:
                logger.warning("Falling back to mock chat completions: %s", exc)
                self._client = None

    async def generate_response(
        self,
        question: str,
        context_text: str,
        chunks: Sequence[RetrievedChunk],
    ) -> dict:
        """Call Azure OpenAI (or fallback) to generate a structured answer."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an internal assurance & advisory copilot for Canadian mid-market firms. "
                    "Answer strictly using the supplied context. If unsure, say you don't know. "
                    "Only cover accounting, audit, and advisory concerns. Never fabricate regulations. "
                    "Return valid JSON with keys: answer (string), risk_highlights (string list),"
                    " citations (list of {document_id,file_name,page_number})."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context_text}\n\nQuestion: {question}\n"
                    "Ensure every paragraph cites at least one source by name and page."
                ),
            },
        ]

        if self._client is not None:
            try:  # pragma: no cover - network required
                response = await self._client.chat.completions.create(
                    model=settings.azure_openai_chat_deployment,
                    temperature=0.2,
                    response_format={"type": "json_object"},
                    messages=messages,
                )
                content = response.choices[0].message.content or "{}"
                return json.loads(content)
            except Exception as exc:  # pragma: no cover - network required
                logger.warning("Azure OpenAI call failed, using mock response: %s", exc)
        return self._mock_response(question, chunks)

    @staticmethod
    def _mock_response(question: str, chunks: Sequence[RetrievedChunk]) -> dict:
        """Return deterministic responses when Azure OpenAI is unavailable."""
        snippets = [chunk.content[:200] for chunk in chunks]
        summary = " ".join(snippets)[:1200]
        if not summary:
            summary = "I could not locate relevant context for this query."
        risk_highlights = []
        lowered = f"{question} {summary}".lower()
        if "revenue" in lowered and "decline" in lowered:
            risk_highlights.append("Revenue declines detected—confirm if additional disclosures are required.")
        if "inventory" in lowered:
            risk_highlights.append("Inventory referenced—validate existence testing and obsolescence reserves.")
        if not risk_highlights:
            risk_highlights.append("Confirm supporting evidence for key balances before issuing conclusions.")
        citations = [
            {
                "document_id": chunk.document_id,
                "file_name": chunk.file_name,
                "page_number": chunk.page_number,
            }
            for chunk in chunks
        ]
        return {
            "answer": summary,
            "risk_highlights": risk_highlights,
            "citations": citations,
        }


class HybridRetriever:
    """Combines Azure AI Search with a SQL fallback for hybrid retrieval."""

    def __init__(self) -> None:
        """Configure the Azure Search client if credentials are available."""
        self._client = None
        can_use_search = (
            SearchClient is not None
            and AzureKeyCredential is not None
            and settings.environment != "local"
            and settings.azure_ai_search_api_key not in {"", "changeme"}
        )
        if can_use_search:
            try:  # pragma: no cover - network required
                credential = AzureKeyCredential(settings.azure_ai_search_api_key)
                self._client = SearchClient(
                    endpoint=settings.azure_ai_search_endpoint,
                    index_name=settings.azure_ai_search_index_name,
                    credential=credential,
                )
            except Exception as exc:
                logger.warning("Falling back to SQL retriever: %s", exc)
                self._client = None

    async def retrieve(
        self,
        query: str,
        user_id: int,
        session: AsyncSession,
        limit: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """Fetch relevant chunks from Azure AI Search or the SQL fallback."""
        limit = limit or settings.rag_top_k
        if self._client is not None:
            try:  # pragma: no cover - network required
                return await self._retrieve_via_search(query, limit)
            except Exception as exc:
                logger.warning("Azure AI Search failed, falling back to SQL: %s", exc)
        return await self._retrieve_via_sql(query, user_id, session, limit)

    async def _retrieve_via_search(self, query: str, limit: int) -> List[RetrievedChunk]:  # pragma: no cover - network required
        """Retrieve chunks via Azure AI Search."""
        results = await asyncio.to_thread(
            self._client.search,
            search_text=query,
            top=limit,
        )
        chunks: List[RetrievedChunk] = []
        for item in results:
            chunks.append(
                RetrievedChunk(
                    chunk_id=item["id"],
                    document_id=item.get("document_id", 0),
                    file_name=item.get("file_name", "Unknown document"),
                    content=item.get("content", ""),
                    page_number=item.get("page_number"),
                )
            )
        return chunks

    async def _retrieve_via_sql(
        self,
        query: str,
        user_id: int,
        session: AsyncSession,
        limit: int,
    ) -> List[RetrievedChunk]:
        """Retrieve chunks using SQL LIKE queries as a fallback."""
        keywords = [token for token in query.split() if len(token) > 2]
        statement = (
            select(document_models.DocumentChunk, document_models.Document)
            .join(document_models.Document, document_models.Document.id == document_models.DocumentChunk.document_id)
            .where(document_models.Document.owner_user_id == user_id)
            .order_by(document_models.DocumentChunk.created_at.desc())
            .limit(limit * 3)
        )
        if keywords:
            like_statements = [
                document_models.DocumentChunk.content.ilike(f"%{token}%") for token in keywords
            ]
            statement = statement.where(or_(*like_statements))
        results = await session.execute(statement)
        rows = results.all()
        chunks: List[RetrievedChunk] = []
        for chunk, document in rows[: limit or len(rows)]:
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=document.id,
                    file_name=document.file_name,
                    content=chunk.content,
                    page_number=chunk.page_number,
                )
            )
        return chunks


class RAGPipeline:
    """Coordinates retrieval, answer generation, citations, and risk highlights."""

    def __init__(self) -> None:
        """Initialize retriever, chat client, and safety helpers."""
        self.retriever = HybridRetriever()
        self.chat_client = AzureOpenAIChatClient()
        self.content_safety = AzureContentSafetyClient()

    async def run(
        self,
        *,
        question: str,
        user: auth_models.User,
        session: AsyncSession,
    ) -> PipelineResult:
        """Execute the full pipeline: retrieve, reason, and build citations."""
        self.content_safety.validate_user_input(question)
        chunks = await self.retriever.retrieve(query=question, user_id=user.id, session=session, limit=settings.rag_top_k)
        if not chunks:
            fallback = "I could not find relevant documents to answer that question. Please upload supporting files."
            citations: List[chat_schemas.Citation] = []
            return PipelineResult(answer=fallback, citations=citations, risk_highlights=[], retrieved_chunks=[])

        context_text = self._build_context(chunks)
        response_payload = await self.chat_client.generate_response(question, context_text, chunks)
        answer = self.content_safety.sanitize_model_output(response_payload.get("answer", ""))
        risk_highlights = response_payload.get("risk_highlights") or self._derive_risk_highlights(chunks)
        citations = self._convert_to_citations(chunks)
        return PipelineResult(
            answer=answer,
            citations=citations,
            risk_highlights=risk_highlights,
            retrieved_chunks=chunks,
        )

    @staticmethod
    def _build_context(chunks: Sequence[RetrievedChunk]) -> str:
        """Construct a context string for the LLM prompt."""
        parts = []
        for idx, chunk in enumerate(chunks, start=1):
            parts.append(
                f"Source {idx} ({chunk.file_name}, page {chunk.page_number}):\n{chunk.content}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _convert_to_citations(chunks: Sequence[RetrievedChunk]) -> List[chat_schemas.Citation]:
        """Convert retrieved chunks into citation payloads."""
        citations = []
        for chunk in chunks:
            citations.append(
                chat_schemas.Citation(
                    document_id=chunk.document_id,
                    file_name=chunk.file_name,
                    page_number=chunk.page_number,
                    snippet=chunk.content[:300],
                )
            )
        return citations

    @staticmethod
    def _derive_risk_highlights(chunks: Sequence[RetrievedChunk]) -> List[str]:
        """Fallback heuristic for risk highlights when the LLM omits them."""
        highlights: List[str] = []
        for chunk in chunks:
            lowered = chunk.content.lower()
            if "going concern" in lowered:
                highlights.append("Potential going-concern flag detected—validate management's mitigation plan.")
            if "covenant" in lowered:
                highlights.append("Debt covenant references found—confirm compliance calculations.")
            if "estimation" in lowered or "judgment" in lowered:
                highlights.append("Significant estimates noted—ensure sensitivity analyses are documented.")
        return highlights or ["Request clarifications on key risk areas before finalizing conclusions."]
