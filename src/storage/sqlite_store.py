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

CREATE VIRTUAL TABLE IF NOT EXISTS code_fts USING fts5(
    file_path UNINDEXED,
    function_name UNINDEXED,
    module_type UNINDEXED,
    body,
    line_start UNINDEXED,
    tokenize="unicode61"
);

CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    file_hash TEXT NOT NULL,
    module_type TEXT,
    object_name TEXT,
    mtime REAL,
    size INTEGER,
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
    centrality REAL DEFAULT 0,
    UNIQUE(file_id, name, line_start)
);

CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
    name,
    content='symbols',
    content_rowid='id'
);

CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY,
    src_symbol_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    dst_symbol_id INTEGER,                 -- NULL until/unless resolved
    callee_name TEXT NOT NULL,
    qualifier TEXT,                        -- NULL for unqualified/local calls
    edge_type TEXT NOT NULL DEFAULT 'call_unresolved',
    resolved INTEGER NOT NULL DEFAULT 0,
    meta TEXT
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

-- Indirect edges (the high-value 1C layer): handlers fired by events, not calls.
CREATE TABLE IF NOT EXISTS subscriptions (
    id INTEGER PRIMARY KEY,
    name TEXT,
    source TEXT,                  -- raw source type list, e.g. "ДокументСсылка.РеализацияТоваров"
    event TEXT,                   -- e.g. "ПередЗаписью"
    handler_module TEXT,          -- common-module object name
    handler_method TEXT,
    handler_symbol_id INTEGER     -- resolved handler symbol (NULL if not found)
);

-- Graph roots: scheduled jobs, web/HTTP services, etc. invoked by the platform.
CREATE TABLE IF NOT EXISTS entry_points (
    id INTEGER PRIMARY KEY,
    kind TEXT,                    -- scheduled_job | web_service | http_service
    name TEXT,
    handler_module TEXT,
    handler_method TEXT,
    handler_symbol_id INTEGER
);

-- code -> data edges: object/table refs mined from query text inside a function.
CREATE TABLE IF NOT EXISTS query_refs (
    id INTEGER PRIMARY KEY,
    src_symbol_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    object_full_name TEXT NOT NULL,
    UNIQUE(src_symbol_id, object_full_name)
);
"""

# Secondary indices kept OUT of the table DDL so rebuild() can defer them until
# after the bulk insert (5-50x faster first build — no per-row index maintenance).
_SCHEMA_INDICES = """
CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
CREATE INDEX IF NOT EXISTS idx_symbols_centrality ON symbols(centrality);
CREATE INDEX IF NOT EXISTS idx_subscriptions_event ON subscriptions(event);
CREATE INDEX IF NOT EXISTS idx_subscriptions_handler ON subscriptions(handler_symbol_id);
CREATE INDEX IF NOT EXISTS idx_entry_points_handler ON entry_points(handler_symbol_id);
CREATE INDEX IF NOT EXISTS idx_query_refs_src ON query_refs(src_symbol_id);
CREATE INDEX IF NOT EXISTS idx_query_refs_obj ON query_refs(object_full_name);
CREATE INDEX IF NOT EXISTS idx_symbols_file_id ON symbols(file_id);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_symbol_id);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_symbol_id);
CREATE INDEX IF NOT EXISTS idx_edges_callee ON edges(callee_name);
CREATE INDEX IF NOT EXISTS idx_edges_qualifier ON edges(qualifier);
CREATE INDEX IF NOT EXISTS idx_files_object_name ON files(object_name);
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


# --- Discovery-card enrichment (lab winner "v_best_callees", 2026-06-08) -----------
# The card embedded for semantic discovery. Measured on 3 independent eval sets
# (held-out doc-comment intents, paraphrases, hand-written): adding the function's
# leading // doc-comment lifts honest semantic hit@5 ~0.37->0.60 / sibling-aware
# 0.53->0.77, MRR x2. For the doc-less generic handlers a humanization phrase fills
# the same role. Keep the bare signature/name (NEVER reorder it to the back —
# that regressed badly) and a light callees/touches tail.
_HANDLER_INTENT = {
    "обработкапроведения": "проведение документа, формирование движений по регистрам",
    "обработкапроведениянасервере": "проведение документа на сервере, формирование движений",
    "обработказаполнения": "заполнение документа при создании на основании",
    "обработкапроверкизаполнения": "проверка заполнения реквизитов документа",
    "обработкапроверкизаполнениянасервере": "проверка заполнения реквизитов на сервере",
    "обработкаудаленияпроведения": "отмена проведения документа, очистка движений",
    "присозданиинасервере": "открытие и инициализация формы на сервере",
    "приоткрытии": "открытие формы",
    "призакрытии": "закрытие формы",
    "передзаписью": "перед записью объекта",
    "передзаписьюнасервере": "перед записью объекта на сервере",
    "послезаписи": "после записи объекта",
    "послезаписинасервере": "после записи объекта на сервере",
    "призаписи": "при записи объекта",
    "обработкаудаления": "обработка удаления объекта",
    "обработкаоповещения": "обработка оповещения формы",
    "обработкакоманды": "обработка команды",
    "причтениинасервере": "чтение данных формы на сервере",
}
_KIND_RU = {
    "Documents": "Документ", "Catalogs": "Справочник", "InformationRegisters": "РегистрСведений",
    "AccumulationRegisters": "РегистрНакопления", "CommonModules": "ОбщийМодуль",
    "DataProcessors": "Обработка", "Reports": "Отчет", "Enums": "Перечисление",
    "BusinessProcesses": "БизнесПроцесс", "Tasks": "Задача",
    "ChartsOfCharacteristicTypes": "ПланВидовХарактеристик",
}
_DOC_STOP = ("Параметры", "Возвращаемое значение", "Возвращаемое значения", "Пример", "Возвращает")


def _kind_from_path(path: str) -> str:
    parts = (path or "").replace("\\", "/").split("/")
    for key in _KIND_RU:
        if key in parts:
            return _KIND_RU[key]
    return ""


def _owner_phrase(kind: str, owner: str, synonym: str) -> str:
    if synonym and kind:
        return f'{kind} "{synonym}"'
    if synonym:
        return f'"{synonym}"'
    if owner and kind:
        return f"{kind} {owner}"
    return owner


def _extract_doc_comment(lines: list[str], line_start: int) -> str:
    """Leading // doc-comment summary (first paragraph, before Параметры/Возвр.).
    lines: full file (0-based list); line_start: 1-based declaration line."""
    if not lines or not line_start:
        return ""
    block = []
    j = line_start - 2  # line above the declaration (0-based)
    while j >= 0 and lines[j].strip().startswith("//"):
        block.append(lines[j])
        j -= 1
    block.reverse()
    out: list[str] = []
    for ln in block:
        s = ln.strip().lstrip("/").strip()
        if not s:
            if out:
                break
            continue
        if set(s) <= set("/=-* "):
            if out:
                break
            continue
        if any(s.startswith(k) for k in _DOC_STOP):
            break
        out.append(s)
        if len(" ".join(out)) > 200:
            break
    return re.sub(r"\s+", " ", " ".join(out)).strip()[:200]


def _compose_card(*, name, type_, params, is_export, module_type, path, owner,
                  synonym, callees, touches, doc) -> str:
    """The v_best_callees discovery card. Single source of truth for card()/cards_for_module()."""
    kind = _kind_from_path(path)
    op = _owner_phrase(kind, owner or "", synonym or "")
    sig = f"{type_} {name}({', '.join(params)})" + (" Экспорт" if is_export else "")
    base = f"{sig} — {module_type}" + (f" {op}" if op else "")
    if callees:
        base += f". Вызывает: {', '.join(callees[:8])}"
    if touches:
        base += f". Работает с: {', '.join(touches[:8])}"
    if doc:
        return f"{doc}. {base}"
    intent = _HANDLER_INTENT.get(name.lower(), "")
    if intent:
        return f"{intent}. {base}"
    return base


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
            conn.executescript(_SCHEMA_INDICES)
            self._migrate(conn)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Idempotent migration for DBs created before edges/Fix A landed."""
        cols = {row[1] for row in conn.execute("PRAGMA table_info(files)")}
        for col, decl in (("object_name", "TEXT"), ("mtime", "REAL"), ("size", "INTEGER")):
            if col not in cols:
                conn.execute(f"ALTER TABLE files ADD COLUMN {col} {decl}")
                logger.info(f"Migrated files: added column {col}")

        sym_cols = {row[1] for row in conn.execute("PRAGMA table_info(symbols)")}
        if "centrality" not in sym_cols:
            conn.execute("ALTER TABLE symbols ADD COLUMN centrality REAL DEFAULT 0")
            logger.info("Migrated symbols: added column centrality")

    def _drop_tables(self) -> None:
        with self._get_conn() as conn:
            # Virtual tables must be dropped before their content tables
            conn.executescript("""
                DROP TABLE IF EXISTS code_fts;
                DROP TABLE IF EXISTS symbols_fts;
                DROP TABLE IF EXISTS objects_fts;
                DROP TABLE IF EXISTS edges;
                DROP TABLE IF EXISTS calls;
                DROP TABLE IF EXISTS subscriptions;
                DROP TABLE IF EXISTS entry_points;
                DROP TABLE IF EXISTS query_refs;
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

    # 1C export type folders -> the segment after the folder is the object name.
    _TYPE_FOLDERS = {
        "Catalogs", "Documents", "CommonModules", "Reports", "DataProcessors",
        "InformationRegisters", "AccumulationRegisters", "AccountingRegisters",
        "CalculationRegisters", "Enums", "ChartsOfCharacteristicTypes",
        "ChartsOfAccounts", "ChartsOfCalculationTypes", "BusinessProcesses",
        "Tasks", "ExchangePlans", "Constants", "CommonForms", "CommonCommands",
        "DocumentJournals", "FilterCriteria", "SettingsStorages",
        "WebServices", "HTTPServices", "ScheduledJobs",
    }

    @staticmethod
    def _module_path_key(file_path: Path) -> str:
        """Stable path key: portion after the nearest `src/` parent, else absolute.

        Must match the key used everywhere (CodeParser.parse_file_functions,
        detector, purge) so hash/edge lookups line up.
        """
        try:
            parts = file_path.parts
            for i in range(len(parts) - 1, -1, -1):
                if parts[i] == "src":
                    return "/".join(parts[i + 1:])
        except Exception:
            pass
        return str(file_path)

    @classmethod
    def _object_name_from_key(cls, path_key: str) -> str | None:
        """Owning 1C object name from a path key, e.g.
        `CommonModules/ОбщегоНазначения/Ext/Module.bsl` -> `ОбщегоНазначения`.
        Enables qualifier->common-module resolution (Fix A)."""
        segs = path_key.replace("\\", "/").split("/")
        for i, seg in enumerate(segs[:-1]):
            if seg in cls._TYPE_FOLDERS and i + 1 < len(segs):
                return segs[i + 1]
        return None

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def rebuild(
        self,
        bsl_files: list[Path],
        metadata_objects: list[MetadataObject],
        subscriptions: list[dict] | None = None,
        entry_points: list[dict] | None = None,
    ) -> IndexStats:
        """Drop everything and reindex from scratch.

        subscriptions/entry_points (parsed from metadata XML by the caller) become
        indirect edges; after all symbols exist their handlers are resolved and
        PageRank centrality is computed over the resolved call graph.
        """
        self._drop_tables()

        with self._get_conn() as conn:
            # Bulk-load: tables only (no secondary indices yet), relaxed durability.
            conn.executescript(_SCHEMA_DDL)
            self._migrate(conn)
            conn.execute("PRAGMA synchronous=OFF")
            conn.execute("PRAGMA temp_store=MEMORY")

            for file_path in bsl_files:
                try:
                    self._index_bsl_file(conn, file_path)
                except Exception as exc:
                    logger.warning(f"Skipping {file_path}: {exc}")

            self._index_metadata_objects(conn, metadata_objects)
            self._index_subscriptions(conn, subscriptions or [])
            self._index_entry_points(conn, entry_points or [])

            # Build indices once over the full data, THEN run index-dependent passes.
            conn.executescript(_SCHEMA_INDICES)
            self._resolve_edges(conn, only_unresolved=False)
            self._resolve_indirect_handlers(conn)
            self._compute_centrality(conn)

        return self.stats()

    def update(
        self,
        changed_files: list[Path],
        deleted_keys: list[str] | None = None,
    ) -> IndexStats:
        """Incrementally re-index changed files and drop deleted ones.

        Per-file purge cleans symbols, edges (src cascade), FTS rows, and nulls
        inbound resolved edges; then the file is reparsed. Finally unresolved
        edges are (re)resolved so new common-module methods become reachable.
        """
        with self._get_conn() as conn:
            for key in (deleted_keys or []):
                self._purge_file(conn, key)
            for file_path in changed_files:
                try:
                    key = self._module_path_key(file_path)
                    self._purge_file(conn, key)
                    self._index_bsl_file(conn, file_path)
                except Exception as exc:
                    logger.warning(f"Skipping {file_path}: {exc}")

            self._resolve_edges(conn, only_unresolved=True)
            self._resolve_indirect_handlers(conn)
            # Centrality is left stale on incremental (global PageRank is heavy and
            # drifts slowly); recomputed on full rebuild / recompute_centrality().

        return self.stats()

    def _purge_file(self, conn: sqlite3.Connection, path_key: str) -> None:
        """Remove a file and everything derived from it, keeping FTS consistent."""
        row = conn.execute("SELECT id FROM files WHERE path = ?", (path_key,)).fetchone()
        if row is None:
            return
        file_id = row["id"]
        fts5_ok = _check_fts5(conn)

        sym_rows = conn.execute(
            "SELECT id, name FROM symbols WHERE file_id = ?", (file_id,)
        ).fetchall()
        for s in sym_rows:
            # Inbound edges that resolved to this symbol must drop back to unresolved
            conn.execute(
                "UPDATE edges SET dst_symbol_id = NULL, resolved = 0, "
                "edge_type = 'call_unresolved' WHERE dst_symbol_id = ?",
                (s["id"],),
            )
            # Indirect-edge handlers point here without an FK — unlink them too
            conn.execute(
                "UPDATE subscriptions SET handler_symbol_id = NULL WHERE handler_symbol_id = ?",
                (s["id"],),
            )
            conn.execute(
                "UPDATE entry_points SET handler_symbol_id = NULL WHERE handler_symbol_id = ?",
                (s["id"],),
            )
            if fts5_ok:
                try:
                    conn.execute(
                        "INSERT INTO symbols_fts(symbols_fts, rowid, name) VALUES('delete', ?, ?)",
                        (s["id"], s["name"]),
                    )
                except sqlite3.OperationalError as exc:
                    logger.debug(f"symbols_fts delete failed: {exc}")

        if fts5_ok:
            try:
                conn.execute("DELETE FROM code_fts WHERE file_path = ?", (path_key,))
            except sqlite3.OperationalError as exc:
                logger.debug(f"code_fts delete failed: {exc}")

        # Cascade removes symbols + edges(src_symbol_id)
        conn.execute("DELETE FROM files WHERE id = ?", (file_id,))

    def _index_bsl_file(self, conn: sqlite3.Connection, file_path: Path) -> None:
        file_hash = self._hash_file(file_path)

        # Stable path key (relative to nearest src/ parent or absolute)
        path_str = self._module_path_key(file_path)
        object_name = self._object_name_from_key(path_str)
        try:
            st = file_path.stat()
            mtime, size = st.st_mtime, st.st_size
        except OSError:
            mtime, size = None, None

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
            "INSERT OR REPLACE INTO files (path, file_hash, module_type, object_name, mtime, size) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (path_str, file_hash, module_type, object_name, mtime, size),
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

            for call in func.calls:
                conn.execute(
                    "INSERT INTO edges (src_symbol_id, callee_name, qualifier, edge_type, resolved) "
                    "VALUES (?, ?, ?, 'call_unresolved', 0)",
                    (symbol_id, call.name, call.qualifier),
                )

            for obj_full in getattr(func, "query_refs", []):
                conn.execute(
                    "INSERT OR IGNORE INTO query_refs (src_symbol_id, object_full_name) VALUES (?, ?)",
                    (symbol_id, obj_full),
                )

            # Index function body into FTS5 for fast grep
            if func.body and fts5_ok:
                try:
                    conn.execute(
                        "INSERT INTO code_fts(file_path, function_name, module_type, body, line_start) VALUES (?, ?, ?, ?, ?)",
                        (path_str, func.name, module_type, func.body, str(func.line_start)),
                    )
                except sqlite3.OperationalError as exc:
                    logger.debug(f"code_fts insert failed: {exc}")

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
    # Indirect edges (subscriptions / entry points)
    # ------------------------------------------------------------------

    def _index_subscriptions(self, conn: sqlite3.Connection, subscriptions: list[dict]) -> None:
        for sub in subscriptions:
            conn.execute(
                """INSERT INTO subscriptions
                   (name, source, event, handler_module, handler_method)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    sub.get("name", ""),
                    sub.get("source", ""),
                    sub.get("event", ""),
                    sub.get("handler_module", ""),
                    sub.get("handler_method", ""),
                ),
            )

    def _index_entry_points(self, conn: sqlite3.Connection, entry_points: list[dict]) -> None:
        for ep in entry_points:
            conn.execute(
                """INSERT INTO entry_points
                   (kind, name, handler_module, handler_method)
                   VALUES (?, ?, ?, ?)""",
                (
                    ep.get("kind", ""),
                    ep.get("name", ""),
                    ep.get("handler_module", ""),
                    ep.get("handler_method", ""),
                ),
            )

    def _resolve_indirect_handlers(self, conn: sqlite3.Connection) -> None:
        """Link subscription / entry-point handlers to their symbol id.

        Handler = (common-module object_name, method). Resolution mirrors Fix A:
        find the method in the named common module.
        """
        for table in ("subscriptions", "entry_points"):
            conn.execute(
                f"""
                UPDATE {table}
                SET handler_symbol_id = (
                    SELECT s.id FROM symbols s
                    JOIN files f ON s.file_id = f.id
                    WHERE f.object_name = {table}.handler_module
                      AND s.name = {table}.handler_method COLLATE NOCASE
                    LIMIT 1
                )
                WHERE handler_method IS NOT NULL AND handler_method != ''
                """
            )

    # ------------------------------------------------------------------
    # Centrality (Aider-style repomap ranking)
    # ------------------------------------------------------------------

    def _compute_centrality(
        self,
        conn: sqlite3.Connection,
        damping: float = 0.85,
        iterations: int = 20,
    ) -> None:
        """PageRank over the resolved call graph -> symbols.centrality.

        Pure power iteration (no numpy). Edges are resolved call_* with a known
        dst; hubs (ОбщегоНазначения-style) float to the top. One-time on rebuild.
        """
        sym_ids = [r[0] for r in conn.execute("SELECT id FROM symbols")]
        n = len(sym_ids)
        if n == 0:
            return

        out_links: dict[int, list[int]] = {}
        for src, dst in conn.execute(
            "SELECT src_symbol_id, dst_symbol_id FROM edges "
            "WHERE resolved = 1 AND dst_symbol_id IS NOT NULL"
        ):
            out_links.setdefault(src, []).append(dst)

        rank = {sid: 1.0 / n for sid in sym_ids}
        base = (1.0 - damping) / n
        for _ in range(iterations):
            nxt = {sid: base for sid in sym_ids}
            dangling = 0.0
            for sid in sym_ids:
                outs = out_links.get(sid)
                if not outs:
                    dangling += rank[sid]
                    continue
                share = damping * rank[sid] / len(outs)
                for dst in outs:
                    if dst in nxt:
                        nxt[dst] += share
            # Redistribute dangling mass uniformly
            if dangling:
                add = damping * dangling / n
                for sid in sym_ids:
                    nxt[sid] += add
            rank = nxt

        conn.executemany(
            "UPDATE symbols SET centrality = ? WHERE id = ?",
            [(score, sid) for sid, score in rank.items()],
        )

    def recompute_centrality(self) -> None:
        """Public re-run of PageRank (e.g. after a batch of incremental updates)."""
        with self._get_conn() as conn:
            self._compute_centrality(conn)

    # ------------------------------------------------------------------
    # Edge resolution (Fix A)
    # ------------------------------------------------------------------

    def _resolve_edges(self, conn: sqlite3.Connection, only_unresolved: bool = True) -> None:
        """Resolve call edges to dst_symbol_id where possible.

        1. Common-module calls: qualifier matches a CommonModule object_name and
           the method exists there  -> call_commonmodule, resolved.
        2. Local calls (no qualifier): a symbol of that name in the SAME file
           -> call_local, resolved.
        Anything else stays resolved=0 but is reclassified (Pass 3) into an
        HONEST taxonomy instead of one undifferentiated bucket:
          - call_dynamic : qualified call on a runtime object (var.Method) — needs
            type inference, unknowable statically.
          - call_platform: bare call whose name is defined NOWHERE in the index —
            a 1C platform global (Сообщить/НачатьТранзакцию…), external by nature.
          - call_unresolved: bare call whose name IS defined somewhere — a GENUINE
            miss worth investigating (the function exists in code but didn't link).
        None of these are guessed a dst_symbol_id; resolved stays 0.
        """
        guard = "AND edges.resolved = 0" if only_unresolved else ""

        # Pass 1 — common-module resolution
        conn.execute(
            f"""
            UPDATE edges
            SET dst_symbol_id = (
                    SELECT s.id FROM symbols s
                    JOIN files f ON s.file_id = f.id
                    WHERE f.module_type = 'CommonModule'
                      AND f.object_name = edges.qualifier
                      AND s.name = edges.callee_name COLLATE NOCASE
                    LIMIT 1
                ),
                edge_type = 'call_commonmodule',
                resolved = 1
            WHERE edges.qualifier IS NOT NULL
              {guard}
              AND EXISTS (
                    SELECT 1 FROM symbols s
                    JOIN files f ON s.file_id = f.id
                    WHERE f.module_type = 'CommonModule'
                      AND f.object_name = edges.qualifier
                      AND s.name = edges.callee_name COLLATE NOCASE
                )
            """
        )

        # Pass 2 — local (same-file) resolution for unqualified calls
        conn.execute(
            f"""
            UPDATE edges
            SET dst_symbol_id = (
                    SELECT s2.id FROM symbols s2
                    WHERE s2.file_id = (
                            SELECT s1.file_id FROM symbols s1 WHERE s1.id = edges.src_symbol_id
                          )
                      AND s2.name = edges.callee_name COLLATE NOCASE
                    LIMIT 1
                ),
                edge_type = 'call_local',
                resolved = 1
            WHERE edges.qualifier IS NULL
              {guard}
              AND EXISTS (
                    SELECT 1 FROM symbols s2
                    WHERE s2.file_id = (
                            SELECT s1.file_id FROM symbols s1 WHERE s1.id = edges.src_symbol_id
                          )
                      AND s2.name = edges.callee_name COLLATE NOCASE
                )
            """
        )

        # Pass 3 — reclassify the still-unresolved remainder (resolved stays 0).
        # A NOCASE membership test against 138k symbols per edge would be a full
        # scan each time (symbols.name is BINARY-collated), so materialise the
        # distinct names once into a NOCASE-keyed temp table the IN-subquery can
        # index into — turns an O(edges×symbols) scan into O(edges×log).
        conn.execute("DROP TABLE IF EXISTS _symnames")
        conn.execute("CREATE TEMP TABLE _symnames (name TEXT COLLATE NOCASE PRIMARY KEY)")
        conn.execute("INSERT OR IGNORE INTO _symnames(name) SELECT name FROM symbols")

        # 3a: qualified -> dynamic (runtime-object method, needs type inference).
        conn.execute(
            """
            UPDATE edges SET edge_type = 'call_dynamic'
            WHERE resolved = 0
              AND qualifier IS NOT NULL AND qualifier <> ''
              AND edge_type <> 'call_dynamic'
            """
        )
        # 3b: bare call whose name is defined nowhere -> platform global (external).
        conn.execute(
            """
            UPDATE edges SET edge_type = 'call_platform'
            WHERE resolved = 0
              AND (qualifier IS NULL OR qualifier = '')
              AND edge_type <> 'call_platform'
              AND callee_name NOT IN (SELECT name FROM _symnames)
            """
        )
        # 3c: bare call whose name IS defined somewhere -> genuine miss. Keep it
        # call_unresolved (and reset any stale tag from a prior pass).
        conn.execute(
            """
            UPDATE edges SET edge_type = 'call_unresolved'
            WHERE resolved = 0
              AND (qualifier IS NULL OR qualifier = '')
              AND edge_type <> 'call_unresolved'
              AND callee_name IN (SELECT name FROM _symnames)
            """
        )
        conn.execute("DROP TABLE IF EXISTS _symnames")

    def count_edges(self) -> dict[str, int]:
        """Return edge counts overall, resolved, and by edge_type."""
        with self._get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
            resolved = conn.execute("SELECT COUNT(*) FROM edges WHERE resolved = 1").fetchone()[0]
            by_type = {
                r[0]: r[1]
                for r in conn.execute(
                    "SELECT edge_type, COUNT(*) FROM edges GROUP BY edge_type"
                ).fetchall()
            }
        return {"total": total, "resolved": resolved, "by_type": by_type}

    # ------------------------------------------------------------------
    # Change detection (incremental trigger)
    # ------------------------------------------------------------------

    def detect_changes(self, source: str | Path) -> tuple[list[Path], list[str]]:
        """Scan `source` for .bsl files and diff against the stored index.

        Fast path uses (mtime, size); only suspects are hashed (the design's
        hash != files.file_hash test, made cheap on a 40k-file base).

        Returns (changed_files, deleted_keys): files to (re)index and path keys
        present in the index but gone from disk.
        """
        source = Path(source)
        disk_files = list(source.rglob("*.bsl"))

        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT path, file_hash, mtime, size FROM files"
            ).fetchall()
        stored = {r["path"]: r for r in rows}

        changed: list[Path] = []
        seen: set[str] = set()
        refresh: list[tuple[str, float, int]] = []

        for f in disk_files:
            key = self._module_path_key(f)
            seen.add(key)
            rec = stored.get(key)
            try:
                st = f.stat()
            except OSError:
                continue
            if rec is None:
                changed.append(f)
                continue
            if (
                rec["mtime"] is not None
                and rec["size"] is not None
                and rec["size"] == st.st_size
                and abs(rec["mtime"] - st.st_mtime) < 1e-6
            ):
                continue  # unchanged — skip the hash entirely
            # Stat differs: confirm by content hash before reparsing
            if rec["file_hash"] != self._hash_file(f):
                changed.append(f)
            else:
                refresh.append((key, st.st_mtime, st.st_size))

        if refresh:
            with self._get_conn() as conn:
                conn.executemany(
                    "UPDATE files SET mtime = ?, size = ? WHERE path = ?",
                    [(m, s, k) for k, m, s in refresh],
                )

        deleted = [k for k in stored if k not in seen]
        return changed, deleted

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

    def get_function_context(
        self, function_name: str, object: str | None = None
    ) -> FunctionContext | None:
        """Return call-graph context for a function.

        object (optional): owning object/module to disambiguate when the name is
        reused across modules. Without it the highest-export/id def is picked and
        ambiguous_definitions reports the collision count.
        """
        with self._get_conn() as conn:
            sym_rows = conn.execute(
                """SELECT s.*, f.path, f.module_type, f.object_name
                   FROM symbols s
                   JOIN files f ON s.file_id = f.id
                   WHERE s.name = ? COLLATE NOCASE
                   ORDER BY s.is_export DESC, s.id""",
                (function_name,),
            ).fetchall()

            if not sym_rows:
                return None

            # The name may be defined in many modules (1C reuses handler names
            # heavily). We answer for one symbol but report how many exist so the
            # caller knows the pick was ambiguous (honest, not silent LIMIT 1).
            ambiguous = len(sym_rows)
            sym_row = sym_rows[0]
            if object:
                ol = object.lower()
                for cand in sym_rows:
                    if ol == (cand["object_name"] or "").lower() or ol in (cand["path"] or "").lower():
                        sym_row = cand
                        break
            func_info = _row_to_function_info(sym_row)
            symbol_id = sym_row["id"]

            calls_rows = conn.execute(
                "SELECT DISTINCT callee_name FROM edges WHERE src_symbol_id = ?",
                (symbol_id,),
            ).fetchall()
            calls = [r["callee_name"] for r in calls_rows]

            # Reverse calls: ONE entry per distinct caller SYMBOL, qualified by
            # module+line. De-duping by bare name (the old behaviour) silently
            # collapsed hundreds of real callers — every module's ПередЗаписью /
            # Подключаемый_* looked like a single caller. GROUP BY src_symbol_id
            # keeps each call site; the resolved dst_symbol_id is the precise key,
            # with an honest name fallback for still-unresolved same-name edges.
            called_by_rows = conn.execute(
                """SELECT s.name AS name,
                          f.object_name AS module,
                          f.module_type AS module_type,
                          s.line_start AS line
                   FROM edges e
                   JOIN symbols s ON e.src_symbol_id = s.id
                   JOIN files f ON s.file_id = f.id
                   WHERE e.dst_symbol_id = ?
                      OR (e.dst_symbol_id IS NULL AND e.callee_name = ? COLLATE NOCASE)
                   GROUP BY e.src_symbol_id
                   ORDER BY f.object_name, s.name""",
                (symbol_id, function_name),
            ).fetchall()
            called_by = [
                {
                    "name": r["name"],
                    "module": r["module"] or r["module_type"],
                    "line": r["line"],
                }
                for r in called_by_rows
            ]

            return FunctionContext(
                function=func_info,
                calls=calls,
                called_by=called_by,
                ambiguous_definitions=ambiguous,
            )

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
    # Code grep via FTS5
    # ------------------------------------------------------------------

    def has_code_fts(self) -> bool:
        """Return True if code_fts has been populated."""
        with self._get_conn() as conn:
            try:
                count = conn.execute("SELECT COUNT(*) FROM code_fts").fetchone()[0]
                return count > 0
            except sqlite3.OperationalError:
                return False

    def code_grep(
        self,
        pattern: str,
        case_sensitive: bool = False,
        limit: int = 20,
    ) -> list[dict] | None:
        """Search for pattern in indexed BSL function bodies using FTS5.

        Returns list of match dicts (same schema as CodeGrep.search),
        or None if code_fts is not populated (caller should fall back).
        """
        if not pattern:
            return []

        with self._get_conn() as conn:
            # Build FTS5 query from the pattern:
            # extract word tokens, use first word as prefix for narrowing,
            # then do exact substring match on the returned bodies.
            words = re.findall(r"[А-Яа-яёЁA-Za-z0-9_]+", pattern)
            if not words:
                return []

            if len(words) == 1:
                fts_query = words[0] + "*"
            else:
                # Phrase: all words must appear in sequence
                fts_query = '"' + " ".join(words) + '"'

            try:
                rows = conn.execute(
                    "SELECT file_path, function_name, module_type, body, line_start FROM code_fts WHERE body MATCH ?",
                    (fts_query,),
                ).fetchall()
            except sqlite3.OperationalError as exc:
                logger.debug(f"code_fts MATCH failed ({exc}), skipping FTS path")
                return None

            if not rows:
                return []

            search_pat = pattern if case_sensitive else pattern.lower()
            results: list[dict] = []

            for row in rows:
                body: str = row["body"] or ""
                body_lines = body.splitlines()
                try:
                    line_start = int(row["line_start"])
                except (ValueError, TypeError):
                    line_start = 1

                for i, line in enumerate(body_lines):
                    check = line if case_sensitive else line.lower()
                    if search_pat not in check:
                        continue

                    abs_line = line_start + i

                    ctx_start = max(0, i - 2)
                    ctx_end = min(len(body_lines), i + 3)
                    context = "\n".join(body_lines[ctx_start:ctx_end])

                    results.append({
                        "file": row["file_path"],
                        "line": abs_line,
                        "text": line.strip(),
                        "function": row["function_name"],
                        "module_type": row["module_type"],
                        "context": context,
                    })

                    if len(results) >= limit:
                        break

                if len(results) >= limit:
                    break

            results.sort(key=lambda r: (r["file"], r["line"]))
            return results[:limit]

    # ------------------------------------------------------------------
    # Wave 1 graph tools: repomap / context / triggers / verify / card
    # ------------------------------------------------------------------

    # Object-module lifecycle handlers fired on write/posting (no qualifier needed).
    WRITE_HANDLERS = {
        "ПередЗаписью": "ПередЗаписью",
        "ПриЗаписи": "ПриЗаписи",
        "ОбработкаПроведения": "Проведение",
        "ОбработкаУдаленияПроведения": "ОтменаПроведения",
        "ПередУдалением": "ПередУдалением",
        "ОбработкаПроверкиЗаполнения": "ОбработкаПроверкиЗаполнения",
        "ОбработкаЗаполнения": "Заполнение",
        "ПриКопировании": "ПриКопировании",
    }

    def _pick_symbol_row(self, conn, name: str, module_hint: str | None = None):
        """Best symbol row for a name: prefer module hint, exports, then centrality."""
        rows = conn.execute(
            """SELECT s.*, f.path, f.module_type, f.object_name
               FROM symbols s JOIN files f ON s.file_id = f.id
               WHERE s.name = ? COLLATE NOCASE""",
            (name,),
        ).fetchall()
        if not rows:
            return None
        if module_hint:
            for r in rows:
                if module_hint.lower() in (r["path"] or "").lower() or \
                   module_hint.lower() == (r["object_name"] or "").lower():
                    return r
        return sorted(
            rows,
            key=lambda r: (r["is_export"], r["centrality"] or 0.0),
            reverse=True,
        )[0]

    def repomap(
        self,
        scope: str | None = None,
        token_budget: int = 4000,
        limit: int = 60,
    ) -> list[dict]:
        """Signatures-only map of the highest-centrality symbols (Aider repomap).

        scope (optional) filters by owning object name or module path substring.
        token_budget bounds output (~4 chars/token). No bodies, no vectors, no LLM.
        """
        char_budget = token_budget * 4
        where = ""
        params: list = []
        if scope:
            where = "WHERE (f.object_name = ? OR f.path LIKE ?)"
            params = [scope, f"%{scope}%"]
        sql = f"""SELECT s.name, s.type, s.params, s.is_export, s.centrality,
                         s.line_start, f.path, f.module_type, f.object_name
                  FROM symbols s JOIN files f ON s.file_id = f.id
                  {where}
                  ORDER BY s.centrality DESC, s.is_export DESC
                  LIMIT ?"""
        params.append(limit)
        out: list[dict] = []
        used = 0
        with self._get_conn() as conn:
            for r in conn.execute(sql, params).fetchall():
                try:
                    plist = json.loads(r["params"]) if r["params"] else []
                except (json.JSONDecodeError, TypeError):
                    plist = []
                sig = f"{r['type']} {r['name']}({', '.join(plist)})"
                if r["is_export"]:
                    sig += " Экспорт"
                line = {
                    "signature": sig,
                    "module": r["path"],
                    "module_type": r["module_type"],
                    "owner": r["object_name"],
                    "line_start": r["line_start"],
                    "centrality": round(r["centrality"] or 0.0, 6),
                }
                used += len(sig) + len(r["path"] or "")
                if used > char_budget and out:
                    break
                out.append(line)
        return out

    def context_for(
        self, symbol_name: str, budget_chars: int = 6000, object: str | None = None
    ) -> dict | None:
        """Assemble a one-shot context bundle for a symbol (highest-ROI lever).

        Returns signature + body + resolved callees + callers + touched
        objects/registers + query refs, budget-bounded. Replaces ~10 grep/read
        round-trips with one deterministic call.

        object (optional): owning object/module to disambiguate when the name is
        reused across many modules (1C reuses ОбработкаПроведения/ПередЗаписью in
        hundreds of objects). Without it, the highest-export/centrality def is
        picked and `ambiguous_definitions` reports how many share the name.
        """
        with self._get_conn() as conn:
            r = self._pick_symbol_row(conn, symbol_name, module_hint=object)
            if r is None:
                return None
            ambiguous = conn.execute(
                "SELECT COUNT(*) FROM symbols WHERE name = ? COLLATE NOCASE",
                (symbol_name,),
            ).fetchone()[0]
            matched = object is None or (
                object.lower() in (r["path"] or "").lower()
                or object.lower() == (r["object_name"] or "").lower()
            )
            sym_id = r["id"]
            try:
                plist = json.loads(r["params"]) if r["params"] else []
            except (json.JSONDecodeError, TypeError):
                plist = []

            callees = [
                {
                    "name": e["callee_name"],
                    "qualifier": e["qualifier"],
                    "edge_type": e["edge_type"],
                    "resolved": bool(e["resolved"]),
                }
                for e in conn.execute(
                    "SELECT callee_name, qualifier, edge_type, resolved FROM edges "
                    "WHERE src_symbol_id = ? ORDER BY resolved DESC",
                    (sym_id,),
                ).fetchall()
            ]

            callers = [
                {"name": c["name"], "module": c["path"], "line_start": c["line_start"]}
                for c in conn.execute(
                    """SELECT DISTINCT s.name, f.path, s.line_start
                       FROM edges e
                       JOIN symbols s ON e.src_symbol_id = s.id
                       JOIN files f ON s.file_id = f.id
                       WHERE e.dst_symbol_id = ?
                          OR (e.dst_symbol_id IS NULL AND e.callee_name = ? COLLATE NOCASE)""",
                    (sym_id, symbol_name),
                ).fetchall()
            ]

            query_refs = [
                qr["object_full_name"]
                for qr in conn.execute(
                    "SELECT object_full_name FROM query_refs WHERE src_symbol_id = ?",
                    (sym_id,),
                ).fetchall()
            ]

            # If this lives in an object module, surface the owner's register movements
            registers: list[str] = []
            if r["object_name"]:
                for rg in conn.execute(
                    """SELECT rm.register_name FROM register_movements rm
                       JOIN objects o ON rm.document_id = o.id
                       WHERE o.name = ?""",
                    (r["object_name"],),
                ).fetchall():
                    registers.append(rg["register_name"])

            body = self._read_symbol_body(conn, sym_id) or ""
            if len(body) > budget_chars:
                body = body[:budget_chars] + "\n... [truncated]"

            return {
                "name": r["name"],
                "type": r["type"],
                "params": plist,
                "is_export": bool(r["is_export"]),
                "module_path": r["path"],
                "module_type": r["module_type"],
                "owner": r["object_name"],
                "line_start": r["line_start"],
                "line_end": r["line_end"],
                "body": body,
                "callees": callees,
                "callers": callers,
                "touched_objects": query_refs,
                "registers": sorted(set(registers)),
                "ambiguous_definitions": ambiguous,
                "requested_object": object,
                "object_matched": matched,
            }

    def _read_lines(self, path: str) -> list[str]:
        """Read a source file's lines (utf-8-sig), small LRU-ish cache for doc-comment
        extraction during card building. Returns [] if unreadable."""
        if not path:
            return []
        cache = getattr(self, "_lines_cache", None)
        if cache is None:
            cache = self._lines_cache = {}
        if path in cache:
            return cache[path]
        try:
            with open(path, encoding="utf-8-sig") as f:
                lines = f.readlines()
        except OSError:
            lines = []
        if len(cache) > 64:  # bound memory; card building is module-local anyway
            cache.clear()
        cache[path] = lines
        return lines

    def _read_symbol_body(self, conn, symbol_id: int) -> str | None:
        """Fetch a symbol's body from code_fts (function_name + file_path + line)."""
        row = conn.execute(
            """SELECT s.name, s.line_start, f.path
               FROM symbols s JOIN files f ON s.file_id = f.id
               WHERE s.id = ?""",
            (symbol_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            fts = conn.execute(
                "SELECT body FROM code_fts WHERE file_path = ? AND function_name = ? "
                "AND line_start = ? LIMIT 1",
                (row["path"], row["name"], str(row["line_start"])),
            ).fetchone()
        except sqlite3.OperationalError:
            return None
        return fts["body"] if fts else None

    def triggers_on_write(self, object_full_name: str, event: str | None = None) -> dict:
        """"What fires when Документ.X is written/posted" — the killer 1C query.

        Combines: object-module lifecycle handlers, matching event subscriptions,
        and register movements. Each is an indirect edge the call graph alone misses.
        """
        name_part = object_full_name.split(".")[-1] if "." in object_full_name else object_full_name
        type_part = object_full_name.split(".")[0] if "." in object_full_name else ""

        result: dict = {
            "object": object_full_name,
            "object_handlers": [],
            "subscriptions": [],
            "register_movements": [],
        }

        with self._get_conn() as conn:
            handler_names = list(self.WRITE_HANDLERS.keys())
            if event:
                handler_names = [h for h, ev in self.WRITE_HANDLERS.items() if event in (h, ev)]
            if handler_names:
                placeholders = ",".join("?" * len(handler_names))
                rows = conn.execute(
                    f"""SELECT s.name, s.line_start, f.path, f.module_type
                        FROM symbols s JOIN files f ON s.file_id = f.id
                        WHERE f.object_name = ? AND f.module_type = 'ObjectModule'
                          AND s.name IN ({placeholders})
                        ORDER BY s.line_start""",
                    [name_part, *handler_names],
                ).fetchall()
                result["object_handlers"] = [
                    {"handler": r["name"], "module": r["path"], "line_start": r["line_start"]}
                    for r in rows
                ]

            sub_sql = "SELECT name, source, event, handler_module, handler_method, handler_symbol_id FROM subscriptions WHERE source LIKE ?"
            sub_params: list = [f"%{name_part}%"]
            if event:
                sub_sql += " AND event = ?"
                sub_params.append(event)
            for r in conn.execute(sub_sql, sub_params).fetchall():
                result["subscriptions"].append({
                    "name": r["name"],
                    "source": r["source"],
                    "event": r["event"],
                    "handler": f"{r['handler_module']}.{r['handler_method']}",
                    "resolved": r["handler_symbol_id"] is not None,
                })

            for r in conn.execute(
                """SELECT rm.register_name FROM register_movements rm
                   JOIN objects o ON rm.document_id = o.id
                   WHERE o.name = ? OR o.full_name = ?""",
                (name_part, object_full_name),
            ).fetchall():
                result["register_movements"].append(r["register_name"])

        return result

    def verify_call(self, caller: str, callee: str) -> dict:
        """Oracle: does `caller` call `callee`? Verdict + evidence (never guess)."""
        with self._get_conn() as conn:
            row = conn.execute(
                """SELECT e.edge_type, e.resolved, e.qualifier
                   FROM edges e JOIN symbols s ON e.src_symbol_id = s.id
                   WHERE s.name = ? COLLATE NOCASE AND e.callee_name = ? COLLATE NOCASE
                   LIMIT 1""",
                (caller, callee),
            ).fetchone()
        if row is None:
            return {"holds": False, "reason": f"No edge {caller} -> {callee} in graph"}
        return {
            "holds": True,
            "edge_type": row["edge_type"],
            "resolved": bool(row["resolved"]),
            "qualifier": row["qualifier"],
        }

    def verify_field(self, object_full_name: str, field: str) -> dict:
        """Oracle: does `field` exist on the object (attribute or tab-part attr)?"""
        details = self.get_object_attributes(object_full_name)
        if details is None:
            return {"holds": False, "reason": f"Object {object_full_name} not found"}
        if any(a.name.lower() == field.lower() for a in details.attributes):
            return {"holds": True, "where": "attribute"}
        for tp in details.tab_parts:
            if any(a.name.lower() == field.lower() for a in tp.attributes):
                return {"holds": True, "where": f"tab_part:{tp.name}"}
        return {
            "holds": False,
            "reason": f"No attribute '{field}' on {object_full_name}",
            "available": [a.name for a in details.attributes][:30],
        }

    def card(self, symbol_name: str) -> dict | None:
        """Deterministic zero-LLM skeleton card from the graph (discovery layer)."""
        with self._get_conn() as conn:
            r = self._pick_symbol_row(conn, symbol_name)
            if r is None:
                return None
            sym_id = r["id"]
            try:
                plist = json.loads(r["params"]) if r["params"] else []
            except (json.JSONDecodeError, TypeError):
                plist = []
            callees = [
                e["callee_name"]
                for e in conn.execute(
                    "SELECT DISTINCT callee_name FROM edges WHERE src_symbol_id = ? "
                    "AND resolved = 1 LIMIT 15",
                    (sym_id,),
                ).fetchall()
            ]
            touches = [
                q["object_full_name"]
                for q in conn.execute(
                    "SELECT object_full_name FROM query_refs WHERE src_symbol_id = ? LIMIT 15",
                    (sym_id,),
                ).fetchall()
            ]
            synonym = ""
            if r["object_name"]:
                syn = conn.execute(
                    "SELECT synonym FROM objects WHERE name = ? LIMIT 1", (r["object_name"],)
                ).fetchone()
                synonym = syn["synonym"] if syn else ""

            owner = r["object_name"] or ""
            sig = f"{r['type']} {r['name']}({', '.join(plist)})"
            if r["is_export"]:
                sig += " Экспорт"
            doc = _extract_doc_comment(self._read_lines(r["path"]), r["line_start"] or 0)
            summary = _compose_card(
                name=r["name"], type_=r["type"], params=plist, is_export=bool(r["is_export"]),
                module_type=r["module_type"], path=r["path"], owner=owner, synonym=synonym,
                callees=callees, touches=touches, doc=doc,
            )
            return {
                "symbol": r["name"],
                "signature": sig,
                "module_type": r["module_type"],
                "owner": owner,
                "synonym": synonym,
                "calls": callees,
                "touches": touches,
                "doc": doc,
                "summary": summary,
            }

    def cards_for_module(self, module_path: str) -> dict[str, str]:
        """Bulk skeleton-card summaries for every symbol in one module.

        Mirrors card()['summary'] exactly but resolves all symbols of a module
        in 3 queries instead of N lookups — built for the vector indexer, which
        embeds these cards (not function bodies) for discovery. Keyed by the
        lowercased symbol name (BSL identifiers are case-insensitive).
        """
        out: dict[str, str] = {}
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT s.id, s.name, s.type, s.params, s.is_export, s.line_start,
                          f.module_type, f.object_name, f.path
                   FROM symbols s JOIN files f ON s.file_id = f.id
                   WHERE f.path = ?""",
                (module_path,),
            ).fetchall()
            if not rows:
                return out
            src_lines = self._read_lines(module_path)

            sym_ids = [r["id"] for r in rows]
            placeholders = ",".join("?" * len(sym_ids))

            callees_by: dict[int, list[str]] = {}
            for e in conn.execute(
                f"SELECT src_symbol_id, callee_name FROM edges "
                f"WHERE src_symbol_id IN ({placeholders}) AND resolved = 1",
                sym_ids,
            ).fetchall():
                lst = callees_by.setdefault(e["src_symbol_id"], [])
                if len(lst) < 15 and e["callee_name"] not in lst:
                    lst.append(e["callee_name"])

            touches_by: dict[int, list[str]] = {}
            for q in conn.execute(
                f"SELECT src_symbol_id, object_full_name FROM query_refs "
                f"WHERE src_symbol_id IN ({placeholders})",
                sym_ids,
            ).fetchall():
                lst = touches_by.setdefault(q["src_symbol_id"], [])
                if len(lst) < 15:
                    lst.append(q["object_full_name"])

            synonym_cache: dict[str, str] = {}
            for r in rows:
                owner = r["object_name"] or ""
                if owner and owner not in synonym_cache:
                    syn = conn.execute(
                        "SELECT synonym FROM objects WHERE name = ? LIMIT 1", (owner,)
                    ).fetchone()
                    synonym_cache[owner] = syn["synonym"] if syn else ""
                synonym = synonym_cache.get(owner, "")

                try:
                    plist = json.loads(r["params"]) if r["params"] else []
                except (json.JSONDecodeError, TypeError):
                    plist = []

                callees = callees_by.get(r["id"], [])
                touches = touches_by.get(r["id"], [])
                doc = _extract_doc_comment(src_lines, r["line_start"] or 0)
                out[r["name"].lower()] = _compose_card(
                    name=r["name"], type_=r["type"], params=plist,
                    is_export=bool(r["is_export"]), module_type=r["module_type"],
                    path=module_path, owner=owner, synonym=synonym,
                    callees=callees, touches=touches, doc=doc,
                )
        return out

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
