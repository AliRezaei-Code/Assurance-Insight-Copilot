"""Security helpers for authentication."""
from datetime import datetime, timedelta
from typing import Optional

from fastapi.security import OAuth2PasswordBearer
from jose import jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.auth import models


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a signed JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.access_token_expire_minutes))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


async def get_user_by_id(session: AsyncSession, user_id: int) -> Optional[models.User]:
    return await session.get(models.User, user_id)


async def get_user_by_email(session: AsyncSession, email: str) -> Optional[models.User]:
    statement = select(models.User).where(models.User.email == email.lower())
    return await session.scalar(statement)


async def authenticate_user(session: AsyncSession, email: str, password: str) -> Optional[models.User]:
    user = await get_user_by_email(session, email)
    if not user or not verify_password(password, user.password_hash):
        return None
    return user
