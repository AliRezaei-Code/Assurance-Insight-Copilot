"""Document ingestion pipeline."""
from __future__ import annotations

import asyncio
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Optional, Sequence
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.documents import models

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from openai import AsyncAzureOpenAI
except ImportError:  # pragma: no cover
    AsyncAzureOpenAI = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from azure.storage.blob import BlobServiceClient
except ImportError:  # pragma: no cover
    BlobServiceClient = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
except ImportError:  # pragma: no cover
    AzureKeyCredential = None  # type: ignore
    SearchClient = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from docx import Document as DocxDocument
except ImportError:  # pragma: no cover
    DocxDocument = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None  # type: ignore

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv"}
DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 150


@dataclass
class TextSegment:
    """Single logical segment of extracted text."""
    text: str
    page_number: Optional[int] = None


@dataclass
class ChunkPayload:
    """Chunk data that will eventually be embedded and indexed."""
    chunk_id: str
    content: str
    page_number: Optional[int]
    embedding: Optional[list[float]] = None
    search_vector_id: Optional[str] = None


class BlobStorageClient:
    """Uploads files to Azure Blob Storage or a local fallback."""

    def __init__(self) -> None:
        """Initialize Azure or local blob storage clients."""
        self.container_name = settings.azure_storage_container
        self.connection_string = settings.azure_storage_connection_string
        self._client = None
        self._container_client = None
        self._local_storage = Path(".local_blob_storage")

        if settings.environment != "local" and BlobServiceClient and "changeme" not in self.connection_string:
            try:
                self._client = BlobServiceClient.from_connection_string(self.connection_string)
                self._container_client = self._client.get_container_client(self.container_name)
                try:
                    self._container_client.create_container()
                except Exception:  # container may already exist
                    pass
            except Exception as exc:  # pragma: no cover - network required
                logger.warning("Falling back to local blob storage: %s", exc)
                self._client = None
                self._container_client = None

        if self._client is None:
            self._local_storage.mkdir(parents=True, exist_ok=True)

    async def upload_file(self, file_path: Path, owner_id: int) -> str:
        blob_name = f"user-{owner_id}/{uuid4().hex}_{file_path.name}"
        if self._container_client is not None:
            def _upload() -> str:
                with file_path.open("rb") as data:
                    self._container_client.upload_blob(blob_name, data, overwrite=True)
                return f"{self._container_client.url}/{blob_name}"

            return await asyncio.to_thread(_upload)

        destination = self._local_storage / blob_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(file_path.read_bytes())
        return str(destination)


class AzureOpenAIEmbeddingClient:
    """Generates embeddings using Azure OpenAI with a local fallback."""

    def __init__(self) -> None:
        """Instantiate Azure OpenAI embedding client or local fallback."""
        self._client = None
        use_remote = (
            AsyncAzureOpenAI is not None
            and settings.environment != "local"
            and settings.azure_openai_api_key not in {"", "changeme"}
        )
        if use_remote:
            try:  # pragma: no cover - network required
                self._client = AsyncAzureOpenAI(
                    api_key=settings.azure_openai_api_key,
                    api_version="2024-05-01-preview",
                    azure_endpoint=settings.azure_openai_endpoint,
                )
            except Exception as exc:
                logger.warning("Falling back to mock embeddings: %s", exc)
                self._client = None

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for each chunk of text."""
        if not texts:
            return []
        if self._client is not None:
            response = await self._client.embeddings.create(
                model=settings.azure_openai_embedding_deployment,
                input=list(texts),
            )
            return [item.embedding for item in response.data]
        return [self._mock_embedding(text) for text in texts]

    @staticmethod
    def _mock_embedding(text: str) -> list[float]:
        """Produce deterministic pseudo-embeddings for local runs."""
        seed = sum(ord(ch) for ch in text) or 1
        return [((seed * (idx + 1)) % 997) / 997 for idx in range(32)]


class AzureAISearchIndexer:
    """Upserts chunk metadata into Azure AI Search (mockable)."""

    def __init__(self) -> None:
        """Configure an Azure AI Search client or choose in-memory fallback."""
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
                logger.warning("Falling back to in-memory indexing: %s", exc)
                self._client = None

    async def upload_chunks(self, document: models.Document, chunks: Sequence[ChunkPayload]) -> None:
        """Upload chunk metadata to the configured search index."""
        if not chunks:
            return
        documents = [
            {
                "id": chunk.chunk_id,
                "document_id": document.id,
                "file_name": document.file_name,
                "page_number": chunk.page_number,
                "content": chunk.content,
                "vector": chunk.embedding,
            }
            for chunk in chunks
        ]
        if self._client is not None:
            await asyncio.to_thread(self._client.upload_documents, documents)
        for chunk in chunks:
            chunk.search_vector_id = chunk.chunk_id


_blob_client = BlobStorageClient()
_embedding_client = AzureOpenAIEmbeddingClient()
_search_indexer = AzureAISearchIndexer()


async def persist_upload_to_disk(upload: UploadFile) -> Path:
    """Persist the in-memory upload to a temporary file."""
    suffix = Path(upload.filename or "upload").suffix
    temp_file = NamedTemporaryFile(delete=False, suffix=suffix)
    content = await upload.read()
    temp_file.write(content)
    temp_file.flush()
    temp_file.close()
    upload.file.seek(0)
    return Path(temp_file.name)


def cleanup_temp_file(file_path: Path) -> None:
    try:
        file_path.unlink(missing_ok=True)
    except Exception:  # pragma: no cover - best-effort cleanup
        logger.debug("Failed to remove temp file %s", file_path, exc_info=True)


async def upload_file_to_blob(file_path: Path, owner_id: int) -> str:
    """Upload local file to blob storage and return the blob URL."""
    return await _blob_client.upload_file(file_path, owner_id)


async def ingest_document(session: AsyncSession, document: models.Document, file_path: Path) -> None:
    """Extracts text, chunks content, creates embeddings, and indexes to Azure AI Search."""
    logger.info("Starting ingestion for document %s", document.id)
    segments = await asyncio.to_thread(extract_text_segments, file_path)
    if not segments:
        raise ValueError("Unable to extract text from document")

    chunks = chunk_text_segments(segments)
    embeddings = await _embedding_client.embed_texts([chunk.content for chunk in chunks])
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding

    await _search_indexer.upload_chunks(document, chunks)

    for chunk in chunks:
        session.add(
            models.DocumentChunk(
                document_id=document.id,
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                page_number=chunk.page_number,
                search_vector_id=chunk.search_vector_id,
            )
        )
    await session.flush()
    logger.info("Completed ingestion for document %s", document.id)


def extract_text_segments(file_path: Path) -> List[TextSegment]:
    """Dispatch to the appropriate extractor based on file extension."""
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")
    if suffix == ".pdf":
        return _extract_pdf(file_path)
    if suffix == ".docx":
        return _extract_docx(file_path)
    if suffix == ".csv":
        return _extract_csv(file_path)
    return _extract_txt(file_path)


def _extract_pdf(file_path: Path) -> List[TextSegment]:
    """Extract text segments from a PDF file."""
    if PdfReader is None:
        raise RuntimeError("pypdf is not installed")
    reader = PdfReader(str(file_path))
    segments: List[TextSegment] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        cleaned = text.strip()
        if cleaned:
            segments.append(TextSegment(text=cleaned, page_number=idx))
    return segments


def _extract_docx(file_path: Path) -> List[TextSegment]:
    """Extract paragraphs from a DOCX file."""
    if DocxDocument is None:
        raise RuntimeError("python-docx is not installed")
    doc = DocxDocument(str(file_path))
    paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
    return [TextSegment(text=paragraph, page_number=index + 1) for index, paragraph in enumerate(paragraphs)]


def _extract_txt(file_path: Path) -> List[TextSegment]:
    """Extract line-by-line text from plain files."""
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return [TextSegment(text=line, page_number=index + 1) for index, line in enumerate(lines)]


def _extract_csv(file_path: Path) -> List[TextSegment]:
    """Convert CSV rows into discrete segments."""
    segments: List[TextSegment] = []
    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        reader = csv.reader(handle)
        for idx, row in enumerate(reader, start=1):
            line = ", ".join(cell.strip() for cell in row if cell.strip())
            if line:
                segments.append(TextSegment(text=line, page_number=idx))
    return segments


def chunk_text_segments(
    segments: Sequence[TextSegment],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[ChunkPayload]:
    """Chunk text into overlapping windows ready for embedding."""
    chunks: List[ChunkPayload] = []
    buffer = ""
    start_page: Optional[int] = None
    for segment in segments:
        text = " ".join(segment.text.split())
        if not text:
            continue
        if start_page is None:
            start_page = segment.page_number
        buffer = f"{buffer} {text}".strip()
        while len(buffer) >= chunk_size:
            chunk_text = buffer[:chunk_size]
            chunks.append(
                ChunkPayload(
                    chunk_id=str(uuid4()),
                    content=chunk_text,
                    page_number=start_page,
                )
            )
            buffer = buffer[chunk_size - overlap :].lstrip()
            start_page = segment.page_number
    if buffer:
        chunks.append(
            ChunkPayload(
                chunk_id=str(uuid4()),
                content=buffer,
                page_number=start_page,
            )
        )
    return chunks
