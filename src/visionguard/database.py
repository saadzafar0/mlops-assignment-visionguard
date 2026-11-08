"""Persistent prediction history and an audit entry for every local HTTP request."""

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    func,
    select,
)
from sqlalchemy.pool import StaticPool

metadata = MetaData()
predictions = Table(
    "predictions",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("filename", String(200), nullable=False),
    Column("image_sha256", String(64), nullable=False),
    Column("image_format", String(8), nullable=False),
    Column("width", Integer, nullable=False),
    Column("height", Integer, nullable=False),
    Column("label", String(4), nullable=False),
    Column("probability_ai", Float, nullable=False),
    Column("confidence", Float, nullable=False),
    Column("threshold", Float, nullable=False),
    Column("model_name", String(128), nullable=False),
    Column("model_version", String(32), nullable=False),
    Column("model_sha256", String(64), nullable=False),
    Column("backend", String(32), nullable=False),
    Column("latency_ms", Float, nullable=False),
)
request_log = Table(
    "request_log",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("request_id", String(36), nullable=False, unique=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("method", String(10), nullable=False),
    Column("path", String(200), nullable=False),
    Column("status", Integer, nullable=False),
    Column("duration_ms", Float, nullable=False),
)


class Database:
    def __init__(self, url):
        options = {"pool_pre_ping": True, "hide_parameters": True}
        if url.startswith("sqlite:"):
            options["connect_args"] = {"check_same_thread": False}
            if url.endswith(":memory:"):
                options["poolclass"] = StaticPool
            elif url.startswith("sqlite:///"):
                Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(url, **options)
        metadata.create_all(self.engine)

    def save_prediction(self, request_id, image, result, backend):
        row = {
            key: result[key]
            for key in (
                "label",
                "probability_ai",
                "confidence",
                "threshold",
                "model_name",
                "model_version",
                "model_sha256",
                "latency_ms",
            )
        }
        row.update(
            id=request_id,
            created_at=datetime.now(UTC),
            filename=image.filename,
            image_sha256=image.sha256,
            image_format=image.format,
            width=image.width,
            height=image.height,
            backend=backend,
        )
        with self.engine.begin() as connection:
            connection.execute(predictions.insert().values(**row))
        return request_id

    def get_prediction(self, identifier):
        with self.engine.connect() as connection:
            row = (
                connection.execute(select(predictions).where(predictions.c.id == identifier))
                .mappings()
                .first()
            )
            return dict(row) if row else None

    def history(self, page=1, page_size=20):
        with self.engine.connect() as connection:
            total = connection.scalar(select(func.count()).select_from(predictions))
            rows = (
                connection.execute(
                    select(predictions)
                    .order_by(predictions.c.created_at.desc(), predictions.c.id.desc())
                    .limit(page_size)
                    .offset((page - 1) * page_size)
                )
                .mappings()
                .all()
            )
            return [dict(row) for row in rows], total

    def record_request(self, request_id, method, path, status, duration_ms):
        with self.engine.begin() as connection:
            connection.execute(
                request_log.insert().values(
                    request_id=request_id,
                    created_at=datetime.now(UTC),
                    method=method,
                    path=path[:200],
                    status=status,
                    duration_ms=duration_ms,
                )
            )

    def healthy(self):
        with self.engine.connect() as connection:
            connection.execute(select(1))
        return True
