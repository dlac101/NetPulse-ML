"""Document indexer: chunks and indexes telemetry summaries + docs into vector store."""

import hashlib
from pathlib import Path

import structlog

from netpulse_ml.config import settings
from netpulse_ml.llm.embedder import Embedder
from netpulse_ml.llm.vector_store import VectorStore

log = structlog.get_logger()


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks by character count.

    Splits on paragraph boundaries first, then sentence boundaries.
    """
    if not text:
        return []

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # If paragraph itself is too long, split by sentences
            if len(para) > chunk_size:
                sentences = para.replace(". ", ".\n").split("\n")
                current_chunk = ""
                for sent in sentences:
                    if len(current_chunk) + len(sent) + 1 <= chunk_size:
                        current_chunk = f"{current_chunk} {sent}" if current_chunk else sent
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sent
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk.strip())

    return [c for c in chunks if c]


class Indexer:
    """Indexes documents into the vector store for RAG retrieval."""

    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    async def index_markdown_file(self, path: Path, source_type: str = "docs") -> int:
        """Index a markdown file by chunking and embedding."""
        if not path.exists():
            log.warning("File not found for indexing", path=str(path))
            return 0

        content = path.read_text(encoding="utf-8")
        chunks = chunk_text(content, settings.rag_chunk_size, settings.rag_chunk_overlap)

        count = 0
        for i, chunk in enumerate(chunks):
            doc_id = hashlib.sha256(f"{path.name}:{i}:{chunk[:50]}".encode()).hexdigest()[:16]
            embedding = self._embedder.embed_text(chunk)
            await self._vector_store.insert(
                content=chunk,
                embedding=embedding,
                metadata={"source": path.name, "source_type": source_type, "chunk_index": i},
                doc_id=doc_id,
            )
            count += 1

        log.info("Indexed file", path=path.name, chunks=count)
        return count

    async def index_fleet_summary(self, summary_text: str) -> int:
        """Index a fleet health summary text."""
        chunks = chunk_text(summary_text, settings.rag_chunk_size, settings.rag_chunk_overlap)

        count = 0
        for i, chunk in enumerate(chunks):
            doc_id = hashlib.sha256(f"fleet_summary:{i}:{chunk[:50]}".encode()).hexdigest()[:16]
            embedding = self._embedder.embed_text(chunk)
            await self._vector_store.insert(
                content=chunk,
                embedding=embedding,
                metadata={"source": "fleet_summary", "source_type": "telemetry", "chunk_index": i},
                doc_id=doc_id,
            )
            count += 1

        return count

    async def index_device_summary(self, device_id: str, summary_text: str) -> int:
        """Index a per-device telemetry summary."""
        doc_id = hashlib.sha256(f"device:{device_id}:{summary_text[:50]}".encode()).hexdigest()[:16]
        embedding = self._embedder.embed_text(summary_text)
        await self._vector_store.insert(
            content=summary_text,
            embedding=embedding,
            metadata={"source": f"device_{device_id}", "source_type": "telemetry", "device_id": device_id},
            doc_id=doc_id,
        )
        return 1

    async def index_project_docs(self) -> int:
        """Index SmartOS documentation from the project root."""
        project_root = Path(__file__).parent.parent.parent.parent.parent
        doc_files = [
            project_root / "classifi.md",
            project_root / "flowstatd.md",
            project_root / "flowstatd-modules.md",
        ]

        total = 0
        for doc_path in doc_files:
            total += await self.index_markdown_file(doc_path, source_type="smartos_docs")

        return total
