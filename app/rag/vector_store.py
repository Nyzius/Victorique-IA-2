import json
import logging
import re
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

from app.config import DOCS_DIR, settings
from app.memory.database import get_connection
from app.rag.document_loader import document_loader
from app.rag.text_splitter import text_splitter
from app.models.ollama_client import ollama_client

logger = logging.getLogger("vector_store")


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _lexical_score(query: str, text: str) -> float:
    """Calcula correspondência léxica/BM25 simplificada entre a busca e o texto."""
    q_lower = query.lower().strip()
    t_lower = text.lower()
    if not q_lower or not t_lower:
        return 0.0

    # Bônus forte se a frase inteira da busca aparece exatamente no texto
    exact_bonus = 0.5 if q_lower in t_lower else 0.0

    # Quebra em termos significativos (palavras com 3 ou mais caracteres)
    terms = [w for w in re.findall(r"\w+", q_lower) if len(w) >= 3]
    if not terms:
        return exact_bonus

    matches = sum(1 for term in terms if term in t_lower)
    term_ratio = matches / len(terms)
    return min(1.0, 0.5 * term_ratio + exact_bonus)


class VectorStore:
    async def index_document(
        self,
        file_path: str,
        collection: str = "Geral",
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Parses, splits, embeds and indexes a document into SQLite.
        Embeddings are computed before opening the write transaction to avoid
        holding a database lock during slow model calls.
        """
        path = Path(file_path).resolve()
        pages = document_loader.load(str(path))
        chunks = text_splitter.split_pages(pages)

        if not chunks:
            raise ValueError(
                "Nenhum texto extraível encontrado no documento. "
                "PDFs digitalizados (imagem) precisam de OCR, que não está incluído nesta versão."
            )

        doc_id = str(uuid.uuid4())
        title = path.name
        ext = path.suffix.lower().replace(".", "")

        logger.info(f"Indexing '{title}': {len(pages)} pages, {len(chunks)} chunks.")

        embedded_chunks = []
        for ch in chunks:
            vector = await ollama_client.generate_embeddings(ch["text"])
            if not vector:
                raise RuntimeError(
                    f"Falha ao gerar embedding para o fragmento {ch['chunk_index']} de '{title}'."
                )
            embedded_chunks.append((ch, vector))

        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO documents (id, title, file_path, file_type, total_pages, total_chunks, collection, project_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (doc_id, title, str(path), ext, len(pages), len(chunks), collection, project_id)
            )

            for ch, vector in embedded_chunks:
                chunk_id = str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO document_chunks (id, document_id, chunk_index, page_number, content, embedding_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (chunk_id, doc_id, ch["chunk_index"], ch["page"], ch["text"], json.dumps(vector))
                )

        return {
            "id": doc_id,
            "title": title,
            "total_pages": len(pages),
            "total_chunks": len(chunks),
            "collection": collection
        }

    async def search(
        self,
        query: str,
        project_id: Optional[str] = None,
        collection: Optional[str] = None,
        top_k: int = 4,
        min_score: float = 0.35
    ) -> List[Dict[str, Any]]:
        """
        Semantic search across indexed document chunks using cosine similarity.
        """
        query_vec = await ollama_client.generate_embeddings(query)
        if not query_vec:
            return []

        sql = """
            SELECT c.id, c.document_id, c.page_number, c.content, c.embedding_json,
                   d.title as doc_title, d.file_path, d.collection
            FROM document_chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE 1=1
        """
        params = []
        if project_id:
            sql += " AND d.project_id = ?"
            params.append(project_id)
        else:
            # No project selected: only global documents. Never leak another
            # project's documents into a global conversation.
            sql += " AND d.project_id IS NULL"
        if collection:
            sql += " AND d.collection = ?"
            params.append(collection)

        scored_results = []
        with get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            for r in rows:
                try:
                    chunk_vec = json.loads(r["embedding_json"])
                    sim = cosine_similarity(query_vec, chunk_vec)
                    
                    if getattr(settings, "rag_hybrid_search", True):
                        lex = _lexical_score(query, r["content"])
                        # Fusão híbrida ponderada: 70% semântica vetorial, 30% correspondência léxica exata
                        final_score = 0.7 * sim + 0.3 * lex
                    else:
                        final_score = sim

                    if final_score >= min_score or (getattr(settings, "rag_hybrid_search", True) and sim >= min_score):
                        scored_results.append({
                            "chunk_id": r["id"],
                            "document_id": r["document_id"],
                            "document_title": r["doc_title"],
                            "file_path": r["file_path"],
                            "page": r["page_number"],
                            "content": r["content"],
                            "score": round(final_score, 4),
                            "similarity": round(sim, 4),
                        })
                except Exception as e:
                    logger.debug(f"Error calculating similarity: {e}")

        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:top_k]

    def list_documents(self, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM documents"
        params = []
        if project_id:
            sql += " WHERE project_id = ?"
            params.append(project_id)
        sql += " ORDER BY created_at DESC"

        with get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def delete_document(self, document_id: str) -> bool:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT file_path FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
            if not row:
                return False
            file_path = row["file_path"]
            cur = conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            if cur.rowcount <= 0:
                return False
            remaining = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE file_path = ?", (file_path,)
            ).fetchone()[0]

        # Remove the stored copy only when no other document row uses it and it
        # lives inside the managed documents directory.
        if remaining == 0 and file_path:
            try:
                p = Path(file_path).resolve()
                docs_root = DOCS_DIR.resolve()
                if p.is_relative_to(docs_root) and p.is_file():
                    p.unlink()
            except Exception as e:
                logger.warning(f"Could not remove document file '{file_path}': {e}")
        return True


vector_store = VectorStore()
