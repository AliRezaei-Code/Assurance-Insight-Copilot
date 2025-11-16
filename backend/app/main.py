"""FastAPI entrypoint for Assurance Insight Copilot."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.auth.routes import router as auth_router
from app.documents.routes import router as document_router
from app.chat.routes import router as chat_router

app = FastAPI(title="Assurance Insight Copilot", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api/auth", tags=["auth"])
app.include_router(document_router, prefix="/api/documents", tags=["documents"])
app.include_router(chat_router, prefix="/api/chat", tags=["chat"])


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    """Healthcheck endpoint for monitoring."""
    return {"status": "ok"}
