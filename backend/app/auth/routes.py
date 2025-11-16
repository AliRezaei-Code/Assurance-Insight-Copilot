"""Authentication routes."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import models, schemas, security
from app.db.session import get_db
from app.dependencies import get_current_user

router = APIRouter()


@router.post("/register", response_model=schemas.UserRead, status_code=status.HTTP_201_CREATED)
async def register_user(
    payload: schemas.UserCreate,
    session: AsyncSession = Depends(get_db),
) -> schemas.UserRead:
    """Register a new internal user and persist hashed credentials."""
    email = payload.email.lower()
    existing = await session.scalar(select(models.User).where(models.User.email == email))
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    new_user = models.User(email=email, password_hash=security.get_password_hash(payload.password))
    session.add(new_user)
    await session.commit()
    await session.refresh(new_user)
    return schemas.UserRead.model_validate(new_user)


@router.post("/login", response_model=schemas.Token)
async def login_user(
    payload: schemas.UserLogin,
    session: AsyncSession = Depends(get_db),
) -> schemas.Token:
    """Exchange credentials for a signed JWT access token."""
    user = await session.scalar(select(models.User).where(models.User.email == payload.email.lower()))
    invalid_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect email or password",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not user or not security.verify_password(payload.password, user.password_hash):
        raise invalid_exception

    token = security.create_access_token({"sub": str(user.id)})
    return schemas.Token(access_token=token)


@router.get("/me", response_model=schemas.UserRead)
async def read_current_user(current_user: models.User = Depends(get_current_user)) -> schemas.UserRead:
    """Return the profile of the authenticated user."""
    return schemas.UserRead.model_validate(current_user)
