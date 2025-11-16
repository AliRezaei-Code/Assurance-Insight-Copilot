"""Pydantic schemas for authentication."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr, ConfigDict


class UserCreate(BaseModel):
    """Payload for creating a user account."""
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    """Login payload for JWT issuance."""
    email: EmailStr
    password: str


class UserRead(BaseModel):
    """Public representation of a user profile."""
    id: int
    email: EmailStr
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class Token(BaseModel):
    """JWT access token response model."""
    access_token: str
    token_type: str = "bearer"


class TokenPayload(BaseModel):
    """Decoded JWT payload."""
    sub: int
