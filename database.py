"""
SQLite database manager.

Creates in-memory SQLite databases from beaver-table schema definitions,
provides SQL validation and execution helpers, and manages database
lifecycle for the benchmark runner.
"""

import json
import sqlite3
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from retrieval import retriever

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite connections and schema construction from beaver-table data."""

    def __init__(self):
        self._connections: Dict[str, sqlite3.Connection] = {}

    def _get_or_create_db(self, split: str, db_name: str) -> sqlite3.Connection:
        """
        Return an in-memory SQLite connection for the given database name.
        If the connection does not exist yet, create the tables from
        beaver-table metadata.
        """
        key = f"{split}:{db_name}"
        if key in self._connections:
            return self._connections[key]

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row

        tables = retriever.get_all_tables_for_db(split, db_name)
        for table in tables:
            self._create_table(conn, table)

        self._connections[key] = conn
        logger.info("Created in-memory database for %s (split=%s) with %d tables.",
                     db_name, split, len(tables))
        return conn

    @staticmethod
    def _create_table(conn: sqlite3.Connection, table_info: Dict[str, Any]) -> None:
        """Create a table and optionally insert example rows."""
        table_name = table_info["table_name"]

        # Parse column metadata
        col_names = table_info["column_names"]
        col_types = table_info["column_types"]
        if isinstance(col_names, str):
            try:
                col_names = json.loads(col_names)
            except (json.JSONDecodeError, TypeError):
                col_names = [c.strip() for c in col_names.split(",")]
        if isinstance(col_types, str):
            try:
                col_types = json.loads(col_types)
            except (json.JSONDecodeError, TypeError):
                col_types = [t.strip() for t in col_types.split(",")]

        # Map types to SQLite-friendly types
        sqlite_types = []
        for t in col_types:
            t_upper = str(t).upper()
            if any(kw in t_upper for kw in ("INT", "SERIAL", "BIGINT")):
                sqlite_types.append("INTEGER")
            elif any(kw in t_upper for kw in ("FLOAT", "DOUBLE", "REAL", "NUMERIC", "DECIMAL")):
                sqlite_types.append("REAL")
            elif any(kw in t_upper for kw in ("BOOL",)):
                sqlite_types.append("INTEGER")
            elif any(kw in t_upper for kw in ("DATE", "TIME", "TIMESTAMP")):
                sqlite_types.append("TEXT")
            else:
                sqlite_types.append("TEXT")

        # Sanitise column names (replace spaces / special chars)
        safe_cols = []
        for c in col_names:
            safe = re.sub(r"[^a-zA-Z0-9_]", "_", str(c).strip())
            safe_cols.append(safe)

        col_defs = ", ".join(
            f'"{sc}" {st}' for sc, st in zip(safe_cols, sqlite_types)
        )
        create_sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({col_defs});'

        try:
            conn.execute(create_sql)
        except sqlite3.OperationalError as exc:
            logger.warning("Failed to create table %s: %s", table_name, exc)
            return

        # Insert example rows if available
        example_rows = table_info.get("example_rows", "")
        if example_rows:
            if isinstance(example_rows, str):
                try:
                    example_rows = json.loads(example_rows)
                except (json.JSONDecodeError, TypeError):
                    example_rows = []

            if isinstance(example_rows, list):
                placeholders = ", ".join(["?"] * len(safe_cols))
                insert_sql = f'INSERT OR IGNORE INTO "{table_name}" VALUES ({placeholders})'
                for row in example_rows:
                    if isinstance(row, dict):
                        values = [row.get(c, None) for c in col_names]
                    elif isinstance(row, (list, tuple)):
                        values = list(row)
                    else:
                        continue
                    # Pad or truncate to match column count
                    values = values[:len(safe_cols)]
                    while len(values) < len(safe_cols):
                        values.append(None)
                    # Convert non-primitive types to strings
                    values = [
                        json.dumps(v) if isinstance(v, (dict, list)) else v
                        for v in values
                    ]
                    try:
                        conn.execute(insert_sql, values)
                    except sqlite3.Error:
                        pass
                conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_sql(self, sql: str, split: str, db_name: str) -> Dict[str, Any]:
        """
        Validate SQL syntax by preparing it against the database schema.

        Returns a dict with ``valid`` (bool) and optionally ``error`` (str).
        """
        conn = self._get_or_create_db(split, db_name)
        try:
            conn.execute(f"EXPLAIN {sql}")
            return {"valid": True}
        except sqlite3.Error as exc:
            return {"valid": False, "error": str(exc)}

    def execute_sql(
        self, sql: str, split: str, db_name: str
    ) -> Dict[str, Any]:
        """
        Execute a SQL query and return the results.

        Returns
        -------
        dict
            ``success``: bool
            ``columns``: list of column names (if success)
            ``rows``: list of row tuples (if success)
            ``error``: str (if failure)
        """
        conn = self._get_or_create_db(split, db_name)
        try:
            cursor = conn.execute(sql)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            # Convert Row objects to plain tuples
            rows = [tuple(r) for r in rows]
            return {"success": True, "columns": columns, "rows": rows}
        except sqlite3.Error as exc:
            return {"success": False, "error": str(exc)}

    def close_all(self) -> None:
        """Close all open connections."""
        for conn in self._connections.values():
            conn.close()
        self._connections.clear()


# Module-level singleton
db_manager = DatabaseManager()
