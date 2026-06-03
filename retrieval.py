"""
Schema retrieval engine.

Downloads the beaver-table dataset from HuggingFace, builds semantic embeddings
for every table description, and exposes a retrieval function that returns the
top-K most relevant tables for a given natural-language question.
"""

import json
import logging
from typing import List, Dict, Any, Optional

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import settings

logger = logging.getLogger(__name__)


class SchemaRetriever:
    """Loads table metadata, embeds descriptions, and retrieves relevant tables."""

    def __init__(self):
        self._tables: Dict[str, List[Dict[str, Any]]] = {}
        self._table_texts: Dict[str, List[str]] = {}
        self._table_embeddings: Dict[str, np.ndarray] = {}
        self._model: Optional[SentenceTransformer] = None
        self._loaded_splits: set = set()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _ensure_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model: %s", settings.embedding_model)
            self._model = SentenceTransformer(settings.embedding_model)
        return self._model

    def load_split(self, split: str) -> None:
        """Download one split of beaver-table and pre-compute embeddings."""
        if split in self._loaded_splits:
            return

        logger.info("Downloading beaver-table split=%s from HuggingFace...", split)
        ds = load_dataset("beaverbench/beaver-table", split=split)

        tables: List[Dict[str, Any]] = []
        texts: List[str] = []

        for row in ds:
            table_info = {
                "db": row["db"],
                "table_name": row["table_name"],
                "column_names": row["column_names"],
                "column_types": row["column_types"],
                "example_rows": row.get("example_rows", ""),
                "example_columns": row.get("example_columns", ""),
            }
            tables.append(table_info)

            # Build a rich text representation for embedding
            col_names = row["column_names"]
            if isinstance(col_names, str):
                try:
                    col_names = json.loads(col_names)
                except (json.JSONDecodeError, TypeError):
                    pass
            if isinstance(col_names, list):
                col_str = ", ".join(str(c) for c in col_names)
            else:
                col_str = str(col_names)

            text = (
                f"Database: {row['db']}. "
                f"Table: {row['table_name']}. "
                f"Columns: {col_str}."
            )
            texts.append(text)

        model = self._ensure_model()
        logger.info("Computing embeddings for %d tables...", len(texts))
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        self._tables[split] = tables
        self._table_texts[split] = texts
        self._table_embeddings[split] = embeddings
        self._loaded_splits.add(split)
        logger.info("Split '%s' loaded: %d tables indexed.", split, len(tables))

    # ------------------------------------------------------------------
    # Public retrieval API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        question: str,
        split: str,
        top_k: Optional[int] = None,
        db_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return the top-K tables most relevant to *question*.

        Parameters
        ----------
        question : str
            The natural-language query from the user.
        split : str
            Which beaver-table split to search (dw / nova / neutron).
        top_k : int, optional
            Number of tables to return. Defaults to settings.top_k_tables.
        db_filter : str, optional
            If provided, restrict results to tables belonging to this database.

        Returns
        -------
        list of dict
            Each dict contains table metadata plus a ``score`` field.
        """
        self.load_split(split)
        top_k = top_k or settings.top_k_tables

        model = self._ensure_model()
        q_emb = model.encode([question], convert_to_numpy=True)

        all_tables = self._tables[split]
        embeddings = self._table_embeddings[split]

        # Optional database-level filter
        if db_filter:
            indices = [
                i for i, t in enumerate(all_tables) if t["db"] == db_filter
            ]
            if not indices:
                return []
            embeddings = embeddings[indices]
            filtered_tables = [all_tables[i] for i in indices]
        else:
            filtered_tables = all_tables

        scores = cosine_similarity(q_emb, embeddings)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            entry = dict(filtered_tables[idx])
            entry["score"] = float(scores[idx])
            results.append(entry)

        return results

    def get_schema_context(
        self,
        question: str,
        split: str,
        top_k: Optional[int] = None,
        db_filter: Optional[str] = None,
    ) -> str:
        """Return a formatted schema string for LLM prompting."""
        tables = self.retrieve(question, split, top_k, db_filter)
        parts: List[str] = []
        for t in tables:
            col_names = t["column_names"]
            col_types = t["column_types"]
            if isinstance(col_names, str):
                try:
                    col_names = json.loads(col_names)
                except (json.JSONDecodeError, TypeError):
                    pass
            if isinstance(col_types, str):
                try:
                    col_types = json.loads(col_types)
                except (json.JSONDecodeError, TypeError):
                    pass

            if isinstance(col_names, list) and isinstance(col_types, list):
                cols_desc = ", ".join(
                    f"{n} ({ty})" for n, ty in zip(col_names, col_types)
                )
            else:
                cols_desc = str(col_names)

            parts.append(
                f"Table: {t['table_name']} (database: {t['db']})\n"
                f"  Columns: {cols_desc}\n"
            )
        return "\n".join(parts)

    def get_all_tables_for_db(self, split: str, db: str) -> List[Dict[str, Any]]:
        """Return every table belonging to a given database."""
        self.load_split(split)
        return [t for t in self._tables[split] if t["db"] == db]

    def get_table_names_for_db(self, split: str, db: str) -> List[str]:
        """Return table names for a database."""
        return [t["table_name"] for t in self.get_all_tables_for_db(split, db)]


# Module-level singleton
retriever = SchemaRetriever()
