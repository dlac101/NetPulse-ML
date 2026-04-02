"""pgvector-based vector store on TimescaleDB for RAG document retrieval."""

import uuid
from datetime import datetime, timezone

import structlog
from sqlalchemy import text

from netpulse_ml.db.engine import async_session_factory

log = structlog.get_logger()


class VectorStore:
    """Read/write document embeddings using pgvector on PostgreSQL/TimescaleDB."""

    async def initialize(self) -> None:
        """Create pgvector extension and table if they don't exist."""
        async with async_session_factory() as session:
            await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS document_embeddings (
                    id          TEXT PRIMARY KEY,
                    content     TEXT NOT NULL,
                    embedding   vector(384) NOT NULL,
                    metadata    JSONB DEFAULT '{}',
                    created_at  TIMESTAMPTZ DEFAULT NOW()
                )
            """))
            # IVFFlat index for fast similarity search
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_doc_embedding
                ON document_embeddings USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """))
            await session.commit()
        log.info("Vector store initialized")

    async def insert(
        self,
        content: str,
        embedding: list[float],
        metadata: dict | None = None,
        doc_id: str | None = None,
    ) -> str:
        """Insert a document with its embedding vector."""
        doc_id = doc_id or str(uuid.uuid4())
        meta = metadata or {}

        async with async_session_factory() as session:
            await session.execute(
                text("""
                    INSERT INTO document_embeddings (id, content, embedding, metadata, created_at)
                    VALUES (:id, :content, :embedding, :metadata, :created_at)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        created_at = EXCLUDED.created_at
                """),
                {
                    "id": doc_id,
                    "content": content,
                    "embedding": str(embedding),
                    "metadata": meta,
                    "created_at": datetime.now(timezone.utc),
                },
            )
            await session.commit()

        return doc_id

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        """Find the top-k most similar documents by cosine distance."""
        filter_clause = ""
        params: dict = {
            "embedding": str(query_embedding),
            "top_k": top_k,
        }

        if metadata_filter:
            conditions = []
            for i, (key, value) in enumerate(metadata_filter.items()):
                param_name = f"meta_{i}"
                conditions.append(f"metadata->>'{key}' = :{param_name}")
                params[param_name] = str(value)
            filter_clause = "WHERE " + " AND ".join(conditions)

        async with async_session_factory() as session:
            result = await session.execute(
                text(f"""
                    SELECT id, content, metadata,
                           1 - (embedding <=> :embedding::vector) AS similarity
                    FROM document_embeddings
                    {filter_clause}
                    ORDER BY embedding <=> :embedding::vector
                    LIMIT :top_k
                """),
                params,
            )
            rows = result.fetchall()

        return [
            {
                "id": row[0],
                "content": row[1],
                "metadata": row[2],
                "similarity": float(row[3]),
            }
            for row in rows
        ]

    async def count(self) -> int:
        """Count total documents in the store."""
        async with async_session_factory() as session:
            result = await session.execute(
                text("SELECT COUNT(*) FROM document_embeddings")
            )
            return result.scalar() or 0

    async def delete_by_metadata(self, key: str, value: str) -> int:
        """Delete documents matching a metadata key-value pair."""
        async with async_session_factory() as session:
            result = await session.execute(
                text("DELETE FROM document_embeddings WHERE metadata->>:key = :value"),
                {"key": key, "value": value},
            )
            await session.commit()
            return result.rowcount or 0
