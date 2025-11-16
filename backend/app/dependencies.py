"""Shared FastAPI dependencies."""
from fastapi import Depends, HTTPException, status
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.session import get_db
from app.auth import security, models as auth_models


async def get_current_user(
    token: str = Depends(security.oauth2_scheme),
    session: AsyncSession = Depends(get_db),
) -> auth_models.User:
    """Decode JWT token and retrieve the current user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        user_id = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError as exc:
        raise credentials_exception from exc

    user = await security.get_user_by_id(session, int(user_id))
    if user is None:
        raise credentials_exception
    return user


__all__ = ["get_db", "get_current_user", "settings"]
