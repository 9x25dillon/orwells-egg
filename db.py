import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from sqlalchemy import create_engine, String, Integer, Text, DateTime, Float, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB as PG_JSONB
from sqlalchemy.types import JSON


DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://limps:limps@localhost:5432/limps")

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)


class Base(DeclarativeBase):
    pass


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


# Tables
class AALCQueue(Base):
    __tablename__ = "aalc_queue"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True) if engine.url.get_backend_name().startswith("postgresql") else String,
        primary_key=True,
        default=uuid.uuid4,
    )
    task_spec: Mapped[dict] = mapped_column(
        PG_JSONB if engine.url.get_backend_name().startswith("postgresql") else JSON,
        nullable=False,
    )
    ifv_spec: Mapped[dict | None] = mapped_column(
        PG_JSONB if engine.url.get_backend_name().startswith("postgresql") else JSON,
        nullable=True,
    )
    priority: Mapped[int] = mapped_column(Integer, default=100)
    status: Mapped[str] = mapped_column(String, default="queued")
    leased_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)


class RepoRML(Base):
    __tablename__ = "repo_rml"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True) if engine.url.get_backend_name().startswith("postgresql") else String,
        primary_key=True,
        default=uuid.uuid4,
    )
    name: Mapped[str | None] = mapped_column(Text)
    arch: Mapped[dict | None] = mapped_column(
        PG_JSONB if engine.url.get_backend_name().startswith("postgresql") else JSON,
        nullable=True,
    )
    weights: Mapped[bytes | None] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)


class RepoRFV(Base):
    __tablename__ = "repo_rfv"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True) if engine.url.get_backend_name().startswith("postgresql") else String,
        primary_key=True,
        default=uuid.uuid4,
    )
    rml_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True) if engine.url.get_backend_name().startswith("postgresql") else String,
        ForeignKey("repo_rml.id"),
        nullable=True,
    )
    rdata_uri: Mapped[str | None] = mapped_column(Text)
    labels: Mapped[dict | None] = mapped_column(
        PG_JSONB if engine.url.get_backend_name().startswith("postgresql") else JSON,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)


class AALCSnapshot(Base):
    __tablename__ = "aalc_snapshot"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True) if engine.url.get_backend_name().startswith("postgresql") else String,
        primary_key=True,
        default=uuid.uuid4,
    )
    aalc_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True) if engine.url.get_backend_name().startswith("postgresql") else String,
        nullable=True,
    )
    meta: Mapped[dict | None] = mapped_column(
        PG_JSONB if engine.url.get_backend_name().startswith("postgresql") else JSON,
        nullable=True,
    )
    weights_uri: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)


class DSQueryLog(Base):
    __tablename__ = "ds_query_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    aalc_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True) if engine.url.get_backend_name().startswith("postgresql") else String,
        nullable=True,
    )
    query: Mapped[str | None] = mapped_column(Text)
    sql_generated: Mapped[str | None] = mapped_column(Text)
    result_uri: Mapped[str | None] = mapped_column(Text)
    entropy: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)


@contextmanager
def get_session() -> Session:
    session = Session(bind=engine, future=True)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()