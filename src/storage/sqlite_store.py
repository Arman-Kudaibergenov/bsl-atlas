"""SQLite structural index for BSL symbols and 1C metadata objects.

Provides fast exact/FTS5 lookup as the primary layer of the dual-layer architecture.
ChromaDB is used for semantic search; this store handles structural queries.
"""

import hashlib
import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any

from ..parsers.code import CodeParser
from .models import (
    Attribute,
    BSLFunction,
    FunctionContext,
    FunctionInfo,
    IndexStats,
    MetadataInfo,
    MetadataObject,
    ObjectDetails,
    ReferenceInfo,
    TabPart,
)

logger = logging.getLogger(__name__)

_SCHEMA_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    file_hash TEXT NOT NULL,
    module_type TEXT,
    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS symbols (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    params TEXT,
    is_export INTEGER DEFAULT 0,
    line_start INTEGER,
    line_end INTEGER,
    UNIQUE(file_id, name, line_start)
);

CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
    name,
    content='symbols',
    content_rowid='id'
);

CREATE TABLE IF NOT EXISTS calls (
    caller_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    callee_name TEXT NOT NULL,
    PRIMARY KEY(caller_id, callee_name)
);

CREATE TABLE IF NOT EXISTS objects (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    object_type TEXT NOT NULL,
    synonym TEXT,
    full_name TEXT UNIQUE NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS objects_fts USING fts5(
    name, synonym, full_name,
    content='objects',
    content_rowid='id'
);

CREATE TABLE IF NOT EXISTS attributes (
    id INTEGER PRIMARY KEY,
    object_id INTEGER NOT NULL REFERENCES objects(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    type_ref TEXT,
    is_required INTEGER DEFAULT 0,
    UNIQUE(object_id, name)
);

CREATE TABLE IF NOT EXISTS tab_parts (
    id INTEGER PRIMARY KEY,
    object_id INTEGER NOT NULL REFERENCES objects(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    UNIQUE(object_id, name)
);

CREATE TABLE IF NOT EXISTS tab_part_attributes (
    id INTEGER PRIMARY KEY,
    tab_part_id INTEGER NOT NULL REFERENCES tab_parts(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    type_ref TEXT,
    UNIQUE(tab_part_id, name)
);

CREATE TABLE IF NOT EXISTS register_movements (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES objects(id) ON DELETE CASCADE,
    register_name TEXT NOT NULL,
    UNIQUE(document_id, register_name)
);

CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
CREATE INDEX IF NOT EXISTS idx_symbols_file_id ON symbols(file_id);
CREATE INDEX IF NOT EXISTS idx_calls_caller ON calls(caller_id);
CREATE INDEX IF NOT EXISTS idx_calls_callee ON calls(callee_name);
CREATE INDEX IF NOT EXISTS idx_objects_name ON objects(name);
CREATE INDEX IF NOT EXISTS idx_attributes_object ON attributes(object_id);
CREATE INDEX IF NOT EXISTS idx_attributes_type_ref ON attributes(type_ref);
"""

_FTS5_AVAILABLE: bool | None = None


def _check_fts5(conn: sqlite3.Connection) -> bool:
    """Return True if the linked SQLite was compiled with FTS5 support."""
    global _FTS5_AVAILABLE
    if _FTS5_AVAILABLE is not None:
        return _FTS5_AVAILABLE
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _fts5_probe USING fts5(x)")
        conn.execute("DROP TABLE IF EXISTS _fts5_probe")
        _FTS5_AVAILABLE = True
    except sqlite3.OperationalError:
        _FTS5_AVAILABLE = False
    return _FTS5_AVAILABLE


def _sanitize_fts_query(query: str) -> str:
    """Escape FTS5 special characters and optionally append * for prefix search."""
    # Remove characters that have special meaning in FTS5 queries
    sanitized = re.sub(r'["\'\(\)\*\:\^]', " ", query).strip()
    sanitized = re.sub(r"\s+", " ", sanitized)
    if not sanitized:
        return '""'
    words = sanitized.split()
    if len(words) == 1:
        return f"{words[0]}*"
    return sanitized


def _row_to_function_info(row: sqlite3.Row) -> FunctionInfo:
    params_raw = row["params"]
    try:
        params = json.loads(params_raw) if params_raw else []
    except (json.JSONDecodeError, TypeError):
        params = []
    return FunctionInfo(
        name=row["name"],
        type=row["type"],
        params=params,
        is_export=bool(row["is_export"]),
        line_start=row["line_start"] or 0,
        line_end=row["line_end"] or 0,
        module_path=row["path"],
        module_type=row["module_type"] or "",
        file_id=row["file_id"],
        symbol_id=row["id"],
    )


class SQLiteStore:
    """Structural SQLite index for BSL symbols and 1C metadata objects."""

    def __init__(self, db_path: str | Path = "data/bsl_index.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.code_parser = CodeParser()
        self._init_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self) -> None:
        with self._get_conn() as conn:
            conn.executescript(_SCHEMA_DDL)

    def _drop_tables(self) -> None:
        with self._get_conn() as conn:
            # Virtual tables must be dropped before their content tables
            conn.executescript("""
                DROP TABLE IF EXISTS symbols_fts;
                DROP TABLE IF EXISTS objects_fts;
                DROP TABLE IF EXISTS calls;
                DROP TABLE IF EXISTS tab_part_attributes;
                DROP TABLE IF EXISTS tab_parts;
                DROP TABLE IF EXISTS register_movements;
                DROP TABLE IF EXISTS attributes;
                DROP TABLE IF EXISTS symbols;
                DROP TABLE IF EXISTS objects;
                DROP TABLE IF EXISTS files;
            """)

    @staticmethod
    def _hash_file(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def rebuild(
        self,
        bsl_files: list[Path],
        metadata_objects: list[MetadataObject],
    ) -> IndexStats:
        """Drop everything and reindex from scratch."""
        self._drop_tables()
        self._init_schema()

        with self._get_conn() as conn:
            for file_path in bsl_files:
                try:
                    self._index_bsl_file(conn, file_path)
                except Exception as exc:
                    logger.warning(f"Skipping {file_path}: {exc}")

            self._index_metadata_objects(conn, metadata_objects)

        return self.stats()

    def update(self, changed_files: list[Path]) -> IndexStats:
        """Incrementally re-index changed files."""
        with self._get_conn() as conn:
            for file_path in changed_files:
                try:
                    path_str = str(file_path)
                    conn.execute("DELETE FROM files WHERE path = ?", (path_str,))
                    self._index_bsl_file(conn, file_path)
                except Exception as exc:
                    logger.warning(f"Skipping {file_path}: {exc}")

        return self.stats()

    def _index_bsl_file(self, conn: sqlite3.Connection, file_path: Path) -> None:
        file_hash = self._hash_file(file_path)

        # Determine stable path key (relative to nearest src/ parent or absolute)
        try:
            parts = file_path.parts
            src_idx = None
            for i in range(len(parts) - 1, -1, -1):
                if parts[i] == "src":
                    src_idx = i
                    break
            path_str = (
                "/".join(parts[src_idx + 1:]) if src_idx is not None else str(file_path)
            )
        except Exception:
            path_str = str(file_path)

        # Skip if already up-to-date
        row = conn.execute(
            "SELECT file_hash FROM files WHERE path = ?", (path_str,)
        ).fetchone()
        if row and row["file_hash"] == file_hash:
            logger.debug(f"Unchanged, skipping: {path_str}")
            return

        # Parse functions
        try:
            functions: list[BSLFunction] = self.code_parser.parse_file_functions(file_path)
        except AttributeError:
            logger.warning("CodeParser.parse_file_functions not available, skipping BSL indexing")
            return
        except Exception as exc:
            logger.warning(f"Parse error for {file_path}: {exc}")
            functions = []

        module_type = functions[0].module_type if functions else "Module"

        conn.execute(
            "INSERT OR REPLACE INTO files (path, file_hash, module_type) VALUES (?, ?, ?)",
            (path_str, file_hash, module_type),
        )
        file_id: int = conn.execute(
            "SELECT id FROM files WHERE path = ?", (path_str,)
        ).fetchone()["id"]

        fts5_ok = _check_fts5(conn)

        for func in functions:
            params_json = json.dumps(func.params)
            conn.execute(
                """INSERT OR IGNORE INTO symbols
                   (file_id, name, type, params, is_export, line_start, line_end)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    file_id,
                    func.name,
                    func.type,
                    params_json,
                    1 if func.is_export else 0,
                    func.line_start,
                    func.line_end,
                ),
            )
            sym_row = conn.execute(
                "SELECT id FROM symbols WHERE file_id = ? AND name = ? AND line_start = ?",
                (file_id, func.name, func.line_start),
            ).fetchone()
            if sym_row is None:
                continue
            symbol_id: int = sym_row["id"]

            if fts5_ok:
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO symbols_fts(rowid, name) VALUES (?, ?)",
                        (symbol_id, func.name),
                    )
                except sqlite3.OperationalError as exc:
                    logger.debug(f"FTS5 insert failed: {exc}")

            for callee in func.calls:
                conn.execute(
                    "INSERT OR IGNORE INTO calls (caller_id, callee_name) VALUES (?, ?)",
                    (symbol_id, callee),
                )

        logger.debug(f"Indexed {len(functions)} symbols from {path_str}")

    def _index_metadata_objects(
        self,
        conn: sqlite3.Connection,
        metadata_objects: list[MetadataObject],
    ) -> None:
        fts5_ok = _check_fts5(conn)

        for obj in metadata_objects:
            full_name = f"{obj.object_type}.{obj.name}"
            conn.execute(
                """INSERT OR REPLACE INTO objects (name, object_type, synonym, full_name)
                   VALUES (?, ?, ?, ?)""",
                (obj.name, obj.object_type, obj.synonym, full_name),
            )
            object_id: int = conn.execute(
                "SELECT id FROM objects WHERE full_name = ?", (full_name,)
            ).fetchone()["id"]

            if fts5_ok:
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO objects_fts(rowid, name, synonym, full_name) VALUES (?, ?, ?, ?)",
                        (object_id, obj.name, obj.synonym or "", full_name),
                    )
                except sqlite3.OperationalError as exc:
                    logger.debug(f"FTS5 insert failed: {exc}")

            for attr in obj.attributes:
                conn.execute(
                    """INSERT OR IGNORE INTO attributes
                       (object_id, name, type_ref, is_required)
                       VALUES (?, ?, ?, ?)""",
                    (
                        object_id,
                        attr.name,
                        attr.type_ref,
                        1 if attr.is_required else 0,
                    ),
                )

            for tp in obj.tab_parts:
                conn.execute(
                    "INSERT OR IGNORE INTO tab_parts (object_id, name) VALUES (?, ?)",
                    (object_id, tp.name),
                )
                tp_row = conn.execute(
                    "SELECT id FROM tab_parts WHERE object_id = ? AND name = ?",
                    (object_id, tp.name),
                ).fetchone()
                if tp_row is None:
                    continue
                tab_part_id: int = tp_row["id"]

                for tp_attr in tp.attributes:
                    conn.execute(
                        """INSERT OR IGNORE INTO tab_part_attributes
                           (tab_part_id, name, type_ref)
                           VALUES (?, ?, ?)""",
                        (tab_part_id, tp_attr.name, tp_attr.type_ref),
                    )

            for reg_name in obj.registers:
                conn.execute(
                    """INSERT OR IGNORE INTO register_movements
                       (document_id, register_name)
                       VALUES (?, ?)""",
                    (object_id, reg_name),
                )

    # ------------------------------------------------------------------
    # Symbol search
    # ------------------------------------------------------------------

    def find_function(self, name: str, exact: bool = True) -> list[FunctionInfo]:
        """Find BSL functions/procedures by name."""
        with self._get_conn() as conn:
            if exact:
                rows = conn.execute(
                    """SELECT s.*, f.path, f.module_type
                       FROM symbols s
                       JOIN files f ON s.file_id = f.id
                       WHERE s.name = ? COLLATE NOCASE""",
                    (name,),
                ).fetchall()
                return [_row_to_function_info(r) for r in rows]

            # Non-exact: try FTS5 first, fall back to LIKE
            fts5_ok = _check_fts5(conn)
            if fts5_ok:
                try:
                    fts_query = _sanitize_fts_query(name)
                    rows = conn.execute(
                        """SELECT s.*, f.path, f.module_type
                           FROM symbols s
                           JOIN files f ON s.file_id = f.id
                           WHERE s.id IN (
                               SELECT rowid FROM symbols_fts WHERE name MATCH ?
                           )""",
                        (fts_query,),
                    ).fetchall()
                    return [_row_to_function_info(r) for r in rows]
                except sqlite3.OperationalError as exc:
                    logger.debug(f"FTS5 search failed, falling back to LIKE: {exc}")

            rows = conn.execute(
                """SELECT s.*, f.path, f.module_type
                   FROM symbols s
                   JOIN files f ON s.file_id = f.id
                   WHERE s.name LIKE ? COLLATE NOCASE""",
                (f"%{name}%",),
            ).fetchall()
            return [_row_to_function_info(r) for r in rows]

    def get_module_functions(self, module_path: str) -> list[FunctionInfo]:
        """Return all functions from a given module, ordered by line number."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT s.*, f.path, f.module_type
                   FROM symbols s
                   JOIN files f ON s.file_id = f.id
                   WHERE f.path = ?
                   ORDER BY s.line_start""",
                (module_path,),
            ).fetchall()

            if not rows:
                rows = conn.execute(
                    """SELECT s.*, f.path, f.module_type
                       FROM symbols s
                       JOIN files f ON s.file_id = f.id
                       WHERE f.path LIKE ?
                       ORDER BY s.line_start""",
                    (f"%{module_path}%",),
                ).fetchall()

            return [_row_to_function_info(r) for r in rows]

    def get_function_context(self, function_name: str) -> FunctionContext | None:
        """Return call-graph context for a function."""
        with self._get_conn() as conn:
            sym_row = conn.execute(
                """SELECT s.*, f.path, f.module_type
                   FROM symbols s
                   JOIN files f ON s.file_id = f.id
                   WHERE s.name = ? COLLATE NOCASE
                   LIMIT 1""",
                (function_name,),
            ).fetchone()

            if sym_row is None:
                return None

            func_info = _row_to_function_info(sym_row)
            symbol_id = sym_row["id"]

            calls_rows = conn.execute(
                "SELECT callee_name FROM calls WHERE caller_id = ?",
                (symbol_id,),
            ).fetchall()
            calls = [r["callee_name"] for r in calls_rows]

            called_by_rows = conn.execute(
                """SELECT DISTINCT s.name
                   FROM calls c
                   JOIN symbols s ON c.caller_id = s.id
                   WHERE c.callee_name = ? COLLATE NOCASE""",
                (function_name,),
            ).fetchall()
            called_by = [r["name"] for r in called_by_rows]

            return FunctionContext(function=func_info, calls=calls, called_by=called_by)

    # ------------------------------------------------------------------
    # Metadata search
    # ------------------------------------------------------------------

    def search_metadata(self, query: str, limit: int = 10) -> list[MetadataInfo]:
        """Search metadata objects by name, synonym, or full_name."""
        with self._get_conn() as conn:
            fts5_ok = _check_fts5(conn)
            rows = None

            if fts5_ok:
                try:
                    fts_query = _sanitize_fts_query(query)
                    rows = conn.execute(
                        """SELECT o.*
                           FROM objects o
                           JOIN objects_fts fts ON o.id = fts.rowid
                           WHERE objects_fts MATCH ?
                           LIMIT ?""",
                        (fts_query, limit),
                    ).fetchall()
                except sqlite3.OperationalError as exc:
                    logger.debug(f"FTS5 metadata search failed, falling back to LIKE: {exc}")
                    rows = None

            if rows is None:
                like_pattern = f"%{query}%"
                rows = conn.execute(
                    """SELECT * FROM objects
                       WHERE name LIKE ?
                          OR synonym LIKE ?
                          OR full_name LIKE ?
                       LIMIT ?""",
                    (like_pattern, like_pattern, like_pattern, limit),
                ).fetchall()

            return [
                MetadataInfo(
                    name=r["name"],
                    object_type=r["object_type"],
                    synonym=r["synonym"] or "",
                    full_name=r["full_name"],
                    object_id=r["id"],
                )
                for r in rows
            ]

    def get_object_attributes(self, full_name: str) -> ObjectDetails | None:
        """Return full attribute/tab-part details for a metadata object."""
        with self._get_conn() as conn:
            obj_row = conn.execute(
                "SELECT * FROM objects WHERE full_name = ?", (full_name,)
            ).fetchone()

            if obj_row is None:
                # Try partial match on name alone
                name_part = full_name.split(".")[-1] if "." in full_name else full_name
                obj_row = conn.execute(
                    "SELECT * FROM objects WHERE name = ? COLLATE NOCASE LIMIT 1",
                    (name_part,),
                ).fetchone()

            if obj_row is None:
                return None

            object_id: int = obj_row["id"]

            attr_rows = conn.execute(
                "SELECT * FROM attributes WHERE object_id = ?", (object_id,)
            ).fetchall()
            attributes = [
                Attribute(
                    name=r["name"],
                    type_ref=r["type_ref"] or "",
                    is_required=bool(r["is_required"]),
                )
                for r in attr_rows
            ]

            tp_rows = conn.execute(
                "SELECT * FROM tab_parts WHERE object_id = ?", (object_id,)
            ).fetchall()
            tab_parts: list[TabPart] = []
            for tp_row in tp_rows:
                tp_attr_rows = conn.execute(
                    "SELECT * FROM tab_part_attributes WHERE tab_part_id = ?",
                    (tp_row["id"],),
                ).fetchall()
                tp_attrs = [
                    Attribute(name=r["name"], type_ref=r["type_ref"] or "")
                    for r in tp_attr_rows
                ]
                tab_parts.append(TabPart(name=tp_row["name"], attributes=tp_attrs))

            reg_rows = conn.execute(
                "SELECT register_name FROM register_movements WHERE document_id = ?",
                (object_id,),
            ).fetchall()
            registers = [r["register_name"] for r in reg_rows]

            return ObjectDetails(
                name=obj_row["name"],
                object_type=obj_row["object_type"],
                synonym=obj_row["synonym"] or "",
                full_name=obj_row["full_name"],
                attributes=attributes,
                tab_parts=tab_parts,
                registers=registers,
            )

    def find_references_to(self, object_full_name: str) -> list[ReferenceInfo]:
        """Find all attributes whose type_ref references the given metadata object."""
        # Build search patterns: direct full_name + common 1C reference suffixes
        name_part = object_full_name.split(".")[-1] if "." in object_full_name else object_full_name
        obj_type = object_full_name.split(".")[0] if "." in object_full_name else ""

        patterns: list[str] = [f"%{object_full_name}%", f"%{name_part}%"]
        # Add СправочникСсылка.X style patterns
        suffix_map = {
            "Справочник": "СправочникСсылка",
            "Документ": "ДокументСсылка",
            "Перечисление": "ПеречислениеСсылка",
            "ПланВидовХарактеристик": "ПланВидовХарактеристикСсылка",
            "ПланСчетов": "ПланСчетовСсылка",
        }
        if obj_type in suffix_map:
            patterns.append(f"%{suffix_map[obj_type]}.{name_part}%")

        results: list[ReferenceInfo] = []
        seen: set[tuple[str, str]] = set()

        with self._get_conn() as conn:
            for pattern in patterns:
                attr_rows = conn.execute(
                    """SELECT a.name AS attr_name, a.type_ref, o.full_name
                       FROM attributes a
                       JOIN objects o ON a.object_id = o.id
                       WHERE a.type_ref LIKE ?""",
                    (pattern,),
                ).fetchall()
                for r in attr_rows:
                    key = (r["full_name"], r["attr_name"])
                    if key not in seen:
                        seen.add(key)
                        results.append(
                            ReferenceInfo(
                                referencing_object=r["full_name"],
                                attribute_name=r["attr_name"],
                                attribute_type=r["type_ref"] or "",
                            )
                        )

                tpa_rows = conn.execute(
                    """SELECT tpa.name AS attr_name, tpa.type_ref,
                              o.full_name || '.' || tp.name AS ref_obj
                       FROM tab_part_attributes tpa
                       JOIN tab_parts tp ON tpa.tab_part_id = tp.id
                       JOIN objects o ON tp.object_id = o.id
                       WHERE tpa.type_ref LIKE ?""",
                    (pattern,),
                ).fetchall()
                for r in tpa_rows:
                    key = (r["ref_obj"], r["attr_name"])
                    if key not in seen:
                        seen.add(key)
                        results.append(
                            ReferenceInfo(
                                referencing_object=r["ref_obj"],
                                attribute_name=r["attr_name"],
                                attribute_type=r["type_ref"] or "",
                            )
                        )

        return results

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> IndexStats:
        """Return current index counts."""
        with self._get_conn() as conn:
            files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            symbols = conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
            objects = conn.execute("SELECT COUNT(*) FROM objects").fetchone()[0]
            attributes = conn.execute("SELECT COUNT(*) FROM attributes").fetchone()[0]
        return IndexStats(files=files, symbols=symbols, objects=objects, attributes=attributes)

    def has_data(self) -> bool:
        """Return True if the index contains any indexed content."""
        s = self.stats()
        return s.symbols > 0 or s.objects > 0
