"""SQLAlchemy metadata import hub."""
from app.db.session import Base

# Import all ORM models so Alembic and metadata creation can discover them.
# pylint: disable=unused-import
try:
    from app.auth import models as auth_models  # noqa: F401
    from app.documents import models as document_models  # noqa: F401
    from app.chat import models as chat_models  # noqa: F401
    from app.evaluation import models as evaluation_models  # noqa: F401
except ImportError:
    # During initial scaffolding some modules might not exist yet.
    pass

__all__ = ["Base"]
