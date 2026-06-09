"""FastMCP server for 1C codebase search.

Dual-layer architecture:
- Structural layer (SQLite + FTS5): search_function, get_module_functions,
  get_function_context, metadatasearch, get_object_details — instant, no embeddings
- Semantic layer (ChromaDB): codesearch, helpsearch, search_code_filtered — vector search
"""

import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastmcp import FastMCP

from .config import config
from .indexer import VectorIndexer
from .indexer.embeddings import create_embedding_provider
from .parsers.metadata_xml import MetadataXMLParser
from .search import HybridSearch
from .search.code_grep import CodeGrep
from .storage.sqlite_store import SQLiteStore

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global instances
indexer: VectorIndexer | None = None
search: HybridSearch | None = None
sqlite_store: SQLiteStore | None = None
_code_grep = CodeGrep()


def _rebuild_sqlite():
    """Scan source_path and rebuild SQLite structural index."""
    if not sqlite_store:
        return

    source = config.source_path
    if not source.exists():
        logger.warning(f"SOURCE_PATH does not exist: {source}, skipping SQLite rebuild")
        return

    logger.info(f"Rebuilding SQLite index from {source}")

    # Collect BSL files
    bsl_files = list(source.rglob("*.bsl"))
    logger.info(f"Found {len(bsl_files)} BSL files")

    # Collect metadata objects + indirect-edge sources from XML dump
    xml_parser = MetadataXMLParser()
    metadata_objects = xml_parser.parse_directory(source)
    subscriptions = xml_parser.parse_event_subscriptions(source)
    entry_points = xml_parser.parse_entry_points(source)
    logger.info(
        f"Found {len(metadata_objects)} metadata objects, "
        f"{len(subscriptions)} subscriptions, {len(entry_points)} entry points"
    )

    stats = sqlite_store.rebuild(bsl_files, metadata_objects, subscriptions, entry_points)
    logger.info(
        f"SQLite rebuild complete: {stats.files} files, {stats.symbols} symbols, "
        f"{stats.objects} objects, {stats.attributes} attributes"
    )


def _reindex_changed() -> dict:
    """Incremental SQLite refresh: detect changed/deleted .bsl, update only those.

    This is the daily-loop fix — a doraborotka reaches the index in seconds
    instead of a full rebuild. Vector layer (if active) gets a best-effort delta.
    """
    if not sqlite_store:
        return {"error": "SQLite store not initialized"}

    source = config.source_path
    if not source.exists():
        logger.warning(f"SOURCE_PATH does not exist: {source}, skipping incremental")
        return {"error": f"SOURCE_PATH does not exist: {source}"}

    changed, deleted = sqlite_store.detect_changes(source)
    logger.info(f"Incremental: {len(changed)} changed, {len(deleted)} deleted")

    if changed or deleted:
        sqlite_store.update(changed, deleted_keys=deleted)

    result: dict = {
        "changed_files": len(changed),
        "deleted_files": len(deleted),
    }
    s = sqlite_store.stats()
    result["sqlite"] = {
        "files": s.files,
        "symbols": s.symbols,
        "objects": s.objects,
        "attributes": s.attributes,
    }
    result["edges"] = sqlite_store.count_edges()

    # Best-effort vector delta (only in full mode with an active indexer)
    if indexer and (changed or deleted):
        try:
            _swap_to_reindex_provider()
            vstats = indexer.reindex_files_vector(changed, deleted)
            result["chromadb"] = vstats
        except Exception as exc:
            logger.error(f"Vector incremental failed: {exc}", exc_info=True)
            result["chromadb_error"] = str(exc)

    return result


def _swap_to_reindex_provider():
    """Swap VectorIndexer embedding provider to local model for reindexing.

    Cloud provider (INDEXING_PROVIDER) is only for initial bulk indexing.
    All subsequent reindex calls use REINDEX_PROVIDER (default: ollama).
    """
    if not indexer:
        return
    if config.reindex_provider == config.indexing_provider:
        logger.info(f"Reindex provider same as indexing provider ({config.reindex_provider}), no swap needed")
        return

    logger.info(
        f"Swapping embedding provider for reindex: "
        f"{config.indexing_provider} → {config.reindex_provider}"
    )
    reindex_base_url = config.ollama_base_url if config.reindex_provider == "ollama" else config.openai_api_base
    reindex_model = config.ollama_model if config.reindex_provider == "ollama" else config.embedding_model

    reindex_embedding = create_embedding_provider(
        provider=config.reindex_provider,
        api_key=config.get_api_key(config.reindex_provider),
        model=reindex_model,
        base_url=reindex_base_url,
    )
    # Swap the provider inside the ChromaDB embedding function wrapper
    indexer.embedding_function._provider = reindex_embedding
    indexer.embedding_provider = reindex_embedding
    logger.info(f"Embedding provider swapped to {config.reindex_provider} for reindex")


def init_services():
    """Initialize all services."""
    global indexer, search, sqlite_store

    # Validate config
    errors = config.validate()
    if errors:
        for error in errors:
            logger.error(error)
        raise ValueError("Configuration errors: " + "; ".join(errors))

    # --- SQLite structural layer ---
    sqlite_store = SQLiteStore(db_path=config.sqlite_db_path)

    if config.sqlite_auto_rebuild:
        if sqlite_store.has_data():
            # Index already exists → only pick up what changed since last run.
            # Runs in the background so startup isn't blocked on the scan.
            logger.info("SQLITE_AUTO_REBUILD=true, index present → incremental refresh in background")

            def _incremental_background():
                try:
                    res = _reindex_changed()
                    logger.info(f"Startup incremental refresh complete: {res}")
                except Exception as e:
                    logger.error(f"Startup incremental refresh failed: {e}", exc_info=True)

            threading.Thread(target=_incremental_background, daemon=True).start()
        else:
            logger.info("SQLITE_AUTO_REBUILD=true, empty index → full structural build...")
            _rebuild_sqlite()
    else:
        existing = sqlite_store.stats()
        logger.info(
            f"SQLite: existing index has {existing.symbols} symbols, {existing.objects} objects"
        )

    # --- ChromaDB vector layer (full mode only) ---
    if config.indexing_mode == "full":
        # Pass the structural graph so code functions embed as skeleton cards
        # (discovery layer), not raw bodies.
        indexer = VectorIndexer(config, graph_store=sqlite_store)

        # Create search embedding provider (if different from indexing)
        search_embedding_provider = None
        logger.info(f"Indexing provider: {config.indexing_provider}, Search provider: {config.search_provider}")
        if config.search_provider != config.indexing_provider:
            logger.info(f"Creating separate search provider: {config.search_provider}")
            search_base_url = config.ollama_base_url if config.search_provider == "ollama" else config.openai_api_base
            search_model = config.ollama_model if config.search_provider == "ollama" else config.embedding_model

            search_embedding_provider = create_embedding_provider(
                provider=config.search_provider,
                api_key=config.get_api_key(config.search_provider),
                model=search_model,
                base_url=search_base_url,
            )

        search = HybridSearch(
            metadata_collection=indexer.metadata_collection,
            code_collection=indexer.code_collection,
            help_collection=indexer.help_collection,
            search_embedding_provider=search_embedding_provider,
            reranker_url=config.reranker_url,
        )

        if config.chromadb_auto_index:
            sqlite_has_data = sqlite_store.has_data()
            logger.info(f"CHROMADB_AUTO_INDEX=true, starting ChromaDB indexing in background (sqlite_enabled={sqlite_has_data})...")

            def _chromadb_background():
                try:
                    indexer.index_directory(sqlite_enabled=sqlite_has_data)
                    logger.info("ChromaDB background indexing complete")
                except Exception as e:
                    logger.error(f"ChromaDB background indexing failed: {e}", exc_info=True)

            thread = threading.Thread(target=_chromadb_background, daemon=True)
            thread.start()
        else:
            logger.info("CHROMADB_AUTO_INDEX=false, skipping ChromaDB indexing at startup")

        logger.info("Services initialized (SQLite ready, ChromaDB active)")
    else:
        logger.info("INDEXING_MODE=fast: ChromaDB disabled. Semantic tools unavailable. SQLite ready.")


@asynccontextmanager
async def lifespan(app: FastMCP):
    """Lifespan context for FastMCP."""
    init_services()
    yield
    logger.info("Shutting down...")


# Create FastMCP app
mcp = FastMCP(
    name="1c-cloud-mcp",
    instructions="""
    MCP server for searching 1C codebase.

    Structural tools (SQLite, instant, no embeddings needed):
    - search_function: Find function/procedure by name across all modules
    - get_module_functions: List all functions in a specific module file
    - get_function_context: Get call graph for a function (calls + called by)
    - metadatasearch: Search 1C metadata objects (catalogs, documents, registers)
    - get_object_details: Get attributes, tabular parts, and register movements

    Semantic tools (ChromaDB vector search):
    - codesearch: Semantic search over BSL code
    - helpsearch: Search 1C documentation
    - search_code_filtered: Semantic code search with structural filters

    Utility:
    - reindex: Rebuild indexes (SQLite always, ChromaDB optional)
    - stats: Indexer statistics
    """,
)


# ---------------------------------------------------------------------------
# Structural tools — SQLite layer
# ---------------------------------------------------------------------------


@mcp.tool()
def search_function(name: str, exact: bool = True) -> list[dict]:
    """Find a function or procedure by name across all indexed modules.

    Args:
        name: Function name to search (e.g. "ПровестиДокумент")
        exact: If True, exact name match; if False, FTS5 fuzzy search

    Returns:
        List of matching functions with module path and signature
    """
    if not sqlite_store:
        return [{"error": "SQLite store not initialized"}]

    results = sqlite_store.find_function(name, exact=exact)
    if not results and exact:
        # Auto-fallback to fuzzy
        results = sqlite_store.find_function(name, exact=False)

    return [
        {
            "name": r.name,
            "type": r.type,
            "params": r.params,
            "is_export": r.is_export,
            "line_start": r.line_start,
            "line_end": r.line_end,
            "module_path": r.module_path,
            "module_type": r.module_type,
        }
        for r in results
    ]


@mcp.tool()
def get_module_functions(module_path: str) -> list[dict]:
    """List all functions and procedures in a BSL module.

    Args:
        module_path: Path or partial path to the module file
                     (e.g. "CommonModules/МодульОбщий" or just "МодульОбщий")

    Returns:
        List of functions with signatures, ordered by line number
    """
    if not sqlite_store:
        return [{"error": "SQLite store not initialized"}]

    results = sqlite_store.get_module_functions(module_path)
    return [
        {
            "name": r.name,
            "type": r.type,
            "params": r.params,
            "is_export": r.is_export,
            "line_start": r.line_start,
            "line_end": r.line_end,
            "module_type": r.module_type,
        }
        for r in results
    ]


# Folded into read_function(include_body=False) — unregistered to remove the confusing
# third "function" tool (search_function = locate, read_function = read+graph).
def get_function_context(function_name: str, object: str | None = None) -> dict:
    """Get call graph context for a function: what it calls and who calls it.

    Args:
        function_name: Name of the function (e.g. "ПровестиДокумент")
        object: optional owning object/module to disambiguate a name reused across
                modules (e.g. "ПоступлениеТоваров"). 1C reuses handler names
                (ОбработкаПроведения/ПередЗаписью) in hundreds of objects — pass this
                to target the right def instead of the arbitrary highest-export pick.

    Returns:
        Dict with function info, list of called functions, and list of callers
    """
    if not sqlite_store:
        return {"error": "SQLite store not initialized"}

    ctx = sqlite_store.get_function_context(function_name, object=object)
    if not ctx:
        return {"error": f"Function '{function_name}' not found in index"}

    return {
        "function": {
            "name": ctx.function.name,
            "type": ctx.function.type,
            "params": ctx.function.params,
            "is_export": ctx.function.is_export,
            "module_path": ctx.function.module_path,
            "module_type": ctx.function.module_type,
            "line_start": ctx.function.line_start,
        },
        "calls": ctx.calls,
        "called_by": ctx.called_by,
        "called_by_count": len(ctx.called_by),
        "ambiguous_definitions": ctx.ambiguous_definitions,
    }


@mcp.tool()
def get_object_details(full_name: str) -> dict:
    """Get attributes, tabular parts, and register movements for a metadata object.

    Args:
        full_name: Full object name (e.g. "Документ.ЛизинговыйДоговор")
                   or just the name (e.g. "ЛизинговыйДоговор")

    Returns:
        Object details with all attributes, tabular sections, and register movements
    """
    if not sqlite_store:
        return {"error": "SQLite store not initialized"}

    details = sqlite_store.get_object_attributes(full_name)
    if not details:
        return {"error": f"Object '{full_name}' not found in index"}

    return {
        "name": details.name,
        "object_type": details.object_type,
        "synonym": details.synonym,
        "full_name": details.full_name,
        "attributes": [
            {"name": a.name, "type": a.type_ref, "required": a.is_required}
            for a in details.attributes
        ],
        "tab_parts": [
            {
                "name": tp.name,
                "attributes": [
                    {"name": a.name, "type": a.type_ref}
                    for a in tp.attributes
                ],
            }
            for tp in details.tab_parts
        ],
        "registers": details.registers,
    }


# ---------------------------------------------------------------------------
# Structural + fallback: metadatasearch
# ---------------------------------------------------------------------------


@mcp.tool()
def metadatasearch(query: str, limit: int = 10) -> list[dict]:
    """Search 1C metadata objects.

    Uses SQLite FTS5 index (instant) if available, otherwise falls back to ChromaDB.

    Args:
        query: Search query (e.g. "Справочник Контрагенты", "ЛизинговыйДоговор")
        limit: Maximum number of results (default: 10)

    Returns:
        List of matching metadata objects
    """
    # SQLite path (fast)
    if sqlite_store and sqlite_store.has_data():
        results = sqlite_store.search_metadata(query, limit)
        return [
            {
                "full_name": r.full_name,
                "object_type": r.object_type,
                "name": r.name,
                "synonym": r.synonym,
                "source": "sqlite",
            }
            for r in results
        ]

    # ChromaDB fallback
    if not search:
        return [{"error": "Search service not initialized"}]

    results = search.search_metadata(query, limit)
    return [
        {
            "full_path": r["full_path"],
            "object_type": r["object_type"],
            "name": r["name"],
            "content": r["content"][:500] + "..." if len(r["content"]) > 500 else r["content"],
            "score": round(r["score"], 3),
            "source": "chromadb",
        }
        for r in results
    ]


# ---------------------------------------------------------------------------
# Graph tools — Wave 1 (repomap / context / triggers / verify / card)
# All SQLite, instant, no embeddings. Named as the questions an AI actually asks.
# ---------------------------------------------------------------------------


@mcp.tool()
def repomap(scope: str | None = None, token_budget: int = 4000) -> list[dict]:
    """Lay-of-the-land map: signatures only of the most central symbols (no bodies).

    PageRank over the call graph surfaces the hubs (ОбщегоНазначения-style modules,
    core procedures). Cheap orientation before diving in. Optionally scope to an
    object name or module-path substring.

    Args:
        scope: object name (e.g. "ОбщегоНазначения") or module-path substring; None = whole base
        token_budget: approximate output budget in tokens (default 4000)

    Returns:
        Signatures with module, owner, and centrality — ranked, bounded.
    """
    if not sqlite_store:
        return [{"error": "SQLite store not initialized"}]
    return sqlite_store.repomap(scope=scope, token_budget=token_budget)


@mcp.tool()
def read_function(
    symbol_name: str,
    budget_chars: int = 6000,
    object: str | None = None,
    include_body: bool = True,
) -> dict:
    """Read a function/procedure with its full neighbourhood in ONE call — the
    "give me the whole procedure and everything around it" tool.

    Returns signature + full body + resolved callees + callers (file:line) + touched
    objects/registers + query refs, budget-bounded. Use this once you know the function
    name (e.g. after search_function or code_grep) instead of grep+read round-trips.
    Set include_body=False for a cheap call-graph-only view (folds the old
    get_function_context).

    Args:
        symbol_name: function/procedure name (e.g. "ПровестиДокумент")
        budget_chars: max body characters to include (default 6000)
        object: optional owning object/module to disambiguate a name reused across
                modules (e.g. "ПоступлениеТоваров"). Without it the highest-export
                def is picked; the result always reports ambiguous_definitions and
                object_matched so the caller knows whether the pick was targeted.
        include_body: include the procedure body text (default True). False = graph only.
    """
    if not sqlite_store:
        return {"error": "SQLite store not initialized"}
    bundle = sqlite_store.context_for(symbol_name, budget_chars=budget_chars, object=object)
    if bundle is None:
        return {"error": f"Symbol '{symbol_name}' not found in index"}
    if not include_body and isinstance(bundle, dict):
        bundle = {k: v for k, v in bundle.items() if k != "body"}
    return bundle


@mcp.tool()
def triggers_on_write(object_full_name: str, event: str | None = None) -> dict:
    """What fires when an object is written/posted — object handlers + subscriptions + register movements.

    The high-value 1C query the call graph alone can't answer: combines object-module
    lifecycle handlers (ПередЗаписью/ОбработкаПроведения/…), matching event
    subscriptions, and register movements.

    Args:
        object_full_name: e.g. "Документ.РеализацияТоваров"
        event: optional filter, e.g. "ПередЗаписью" / "Проведение"
    """
    if not sqlite_store:
        return {"error": "SQLite store not initialized"}
    return sqlite_store.triggers_on_write(object_full_name, event=event)


@mcp.tool()
def verify_call(caller: str, callee: str) -> dict:
    """Oracle: does `caller` actually call `callee`? Check before asserting it.

    Returns holds + edge_type + resolved, or holds=false with a reason. Use to
    fact-check a claim against the graph instead of guessing.
    """
    if not sqlite_store:
        return {"error": "SQLite store not initialized"}
    return sqlite_store.verify_call(caller, callee)


@mcp.tool()
def verify_field(object_full_name: str, field: str) -> dict:
    """Oracle: does `field` exist on the object (attribute or tabular-part attribute)?

    Returns holds + where, or holds=false with available attribute names. Use before
    referencing a 1C requisite you're not certain exists.
    """
    if not sqlite_store:
        return {"error": "SQLite store not initialized"}
    return sqlite_store.verify_field(object_full_name, field)


# NOTE: `card` is an INTERNAL primitive (used to build vector-embedding cards),
# not exposed as an agent tool — context_for covers the agent-facing need and
# avoids menu clutter / tool-choice guessing.
def card(symbol_name: str) -> dict:
    """Deterministic one-line skeleton card for a symbol (zero LLM, from the graph).

    Module type + owner(synonym) + what it calls + what data it touches. A compact
    "what is this" without reading the body.
    """
    if not sqlite_store:
        return {"error": "SQLite store not initialized"}
    c = sqlite_store.card(symbol_name)
    if c is None:
        return {"error": f"Symbol '{symbol_name}' not found in index"}
    return c


def _resolve_source_path(path: str) -> Path:
    """Resolve a tool-supplied path: absolute as-is, else relative to SOURCE_PATH."""
    p = Path(path)
    if p.is_absolute():
        return p
    return config.source_path / path


@mcp.tool()
def get_form_info(form_path: str) -> dict:
    """Structure + event handlers of a managed form (Form.xml), read on demand.

    Returns attributes, commands, items, declared handlers, and which handlers
    actually exist as procedures in the form module. Use before editing a form
    module so you know the real handler names.

    Args:
        form_path: path to Form.xml (absolute, or relative to SOURCE_PATH)
    """
    from .parsers.form_xml import FormXMLParser

    p = _resolve_source_path(form_path)
    if not p.exists():
        return {"error": f"Form.xml not found: {p}"}
    info = FormXMLParser().parse_file(p)
    if info is None:
        return {"error": f"Could not parse Form.xml: {p}"}
    return info


@mcp.tool()
def get_skd_info(template_path: str) -> dict:
    """Datasets / query text / fields / parameters of a DataCompositionSchema (СКД).

    Returns each dataset's query, the object refs inside it (what data the report
    reads), available fields, and schema parameters. Read on demand.

    Args:
        template_path: path to the СКД Template XML (absolute or relative to SOURCE_PATH)
    """
    from .parsers.skd_xml import SKDXMLParser

    p = _resolve_source_path(template_path)
    if not p.exists():
        return {"error": f"Template not found: {p}"}
    info = SKDXMLParser().parse_file(p)
    if info is None:
        return {"error": f"Not a DataCompositionSchema: {p}"}
    return info


# ---------------------------------------------------------------------------
# Semantic tools — ChromaDB layer
# ---------------------------------------------------------------------------

_FAST_MODE_MSG = (
    "Semantic search requires INDEXING_MODE=full. "
    "Current mode: fast. SQLite search covers 80% of queries."
)


def _tool_if(enabled: bool):
    """Register as an MCP tool only when `enabled`; otherwise leave the function
    unregistered so it does NOT appear in tools/list. Prevents the agent from
    seeing/calling vector tools that are dead in fast mode (false 'not found')."""
    return mcp.tool() if enabled else (lambda fn: fn)


_SEMANTIC_ENABLED = config.indexing_mode == "full"


@_tool_if(_SEMANTIC_ENABLED)
def codesearch(query: str, limit: int = 10) -> list[dict]:
    """Search 1C BSL code semantically.

    Args:
        query: Search query (e.g. "ПроверитьЗаполнение", "расчет суммы договора")
        limit: Maximum number of results (default: 10)

    Returns:
        List of matching code fragments with module path and content
    """
    if config.indexing_mode != "full":
        return [{"info": _FAST_MODE_MSG}]
    if not search:
        return [{"error": "Search service not initialized"}]

    results = search.search_code(query, limit)
    return [
        {
            "full_path": r["full_path"],
            "name": r["name"],
            "module_type": r["metadata"].get("module_type", ""),
            "is_export": r["metadata"].get("is_export", False),
            "functions": r["metadata"].get("functions", ""),
            "content": r["content"][:800] + "..." if len(r["content"]) > 800 else r["content"],
            "score": round(r["score"], 3),
        }
        for r in results
    ]


@_tool_if(_SEMANTIC_ENABLED)
def helpsearch(query: str, limit: int = 10) -> list[dict]:
    """Search 1C documentation.

    Args:
        query: Search query (e.g. "Справочник создание", "Регистр сведений")
        limit: Maximum number of results (default: 10)

    Returns:
        List of matching documentation sections
    """
    if config.indexing_mode != "full":
        return [{"info": _FAST_MODE_MSG}]
    if not search:
        return [{"error": "Search service not initialized"}]

    results = search.search_help(query, limit)
    return [
        {
            "full_path": r["full_path"],
            "title": r["metadata"].get("title", r["name"]),
            "content": r["content"][:800] + "..." if len(r["content"]) > 800 else r["content"],
            "score": round(r["score"], 3),
        }
        for r in results
    ]


@_tool_if(_SEMANTIC_ENABLED)
def search_code_filtered(
    query: str,
    module_type: str | None = None,
    only_export: bool = False,
    limit: int = 10,
) -> list[dict]:
    """Search BSL code with structural filters.

    Semantic search over code combined with filters for module type and export flag.

    Args:
        query: Semantic search query
        module_type: Filter by module type: "CommonModule", "ObjectModule",
                     "ManagerModule", "FormModule", "RecordSetModule"
        only_export: If True, return only exported functions
        limit: Maximum number of results (default: 10)

    Returns:
        List of matching code fragments
    """
    if config.indexing_mode != "full":
        return [{"info": _FAST_MODE_MSG}]
    if not search:
        return [{"error": "Search service not initialized"}]

    results = search.search_code_filtered(
        query=query,
        module_type=module_type,
        only_export=only_export,
        limit=limit,
    )
    return [
        {
            "full_path": r["full_path"],
            "name": r["name"],
            "module_type": r["metadata"].get("module_type", ""),
            "is_export": r["metadata"].get("is_export", False),
            "content": r["content"][:800] + "..." if len(r["content"]) > 800 else r["content"],
            "score": round(r["score"], 3),
        }
        for r in results
    ]


# ---------------------------------------------------------------------------
# Raw grep tool — works in INDEXING_MODE=fast
# ---------------------------------------------------------------------------


@mcp.tool()
def code_grep(pattern: str, case_sensitive: bool = False, limit: int = 100) -> list[dict]:
    """Search for text pattern in BSL code with AST context.

    Unlike regular grep, returns matches with structural context:
    which function/procedure contains the match, in which module.

    Uses SQLite FTS5 index (instant, <100ms) when available.
    Falls back to file scanning when index is not yet built.

    Args:
        pattern: Text pattern to search (substring match, not regex)
        case_sensitive: Whether search is case-sensitive (default: False)
        limit: Maximum results to return (default: 100). When the result count
            reaches `limit`, output is TRUNCATED and a final {"_truncated": true}
            element is appended — narrow the pattern or raise `limit` to see the
            rest. Do NOT conclude "only N matches exist" when truncated.

    Returns: List of matches with fields:
        - file: relative path to .bsl file
        - line: line number of match
        - text: the matching line (trimmed)
        - function: name of containing function/procedure (or "module level")
        - module_type: type of module (ObjectModule, ManagerModule, etc.)
        - context: 2 lines before + 2 lines after the match
    """
    results = None
    # Fast path: FTS5 index (instant)
    if sqlite_store and sqlite_store.has_code_fts():
        results = sqlite_store.code_grep(pattern, case_sensitive=case_sensitive, limit=limit)

    if results is None:
        # Fallback: file scanning (slow, used when index not yet built)
        source = config.source_path
        if not source.exists():
            return [{"error": f"SOURCE_PATH does not exist: {source}"}]
        results = _code_grep.search(
            pattern=pattern,
            source_path=source,
            case_sensitive=case_sensitive,
            limit=limit,
        )

    # Truncation signal: when full, the agent must NOT assume this is the whole set.
    if len(results) >= limit:
        results = results + [{
            "_truncated": True,
            "shown": len(results),
            "hint": f"Показаны первые {len(results)} совпадений (лимит достигнут). "
                    f"Сузь паттерн/добавь контекст или подними limit — это НЕ все совпадения.",
        }]
    return results


# ---------------------------------------------------------------------------
# Utility tools
# ---------------------------------------------------------------------------


@mcp.tool()
def reindex_changed() -> dict:
    """Pick up edited/added/deleted .bsl files since the last index — fast.

    Scans the source tree, detects files whose content changed (mtime/size
    prefilter + hash confirm), re-indexes ONLY those into the SQLite graph
    (symbols, call edges with qualifier resolution), and drops vanished files.
    Use this after editing 1C code so the index reflects your changes in seconds
    instead of a full rebuild.

    Returns:
        Counts of changed/deleted files plus current SQLite + edge statistics.
    """
    return _reindex_changed()


# Admin-only foot-gun (full rebuild) — unregistered from the agent menu. Use the CLI /
# container restart for a full rebuild; agents use reindex_changed (incremental).
def reindex(rebuild_sqlite: bool = True, force_chromadb: bool = False) -> dict:
    """Re-index the 1C codebase.

    SQLite rebuild is instant. ChromaDB reindex uses LOCAL embedding model
    (REINDEX_PROVIDER, default: ollama/qwen3-embedding:8b) — no cloud API calls.
    Cloud provider (INDEXING_PROVIDER) is only used for initial bulk indexing at startup.

    Args:
        rebuild_sqlite: If True (default), rebuild SQLite structural index
        force_chromadb: If True, clear and rebuild ChromaDB vector index too

    Returns:
        Indexing statistics
    """
    result: dict = {"status": "completed"}

    if rebuild_sqlite and sqlite_store:
        _rebuild_sqlite()
        s = sqlite_store.stats()
        result["sqlite"] = {
            "files": s.files,
            "symbols": s.symbols,
            "objects": s.objects,
            "attributes": s.attributes,
        }

    if force_chromadb and indexer:
        _swap_to_reindex_provider()
        indexer.clear_all()
        sqlite_has_data = sqlite_store.has_data() if sqlite_store else False
        chromadb_stats = indexer.index_directory(sqlite_enabled=sqlite_has_data)
        result["chromadb"] = chromadb_stats
        result["chromadb_provider"] = config.reindex_provider

    return result


@mcp.tool()
def stats() -> dict:
    """Get statistics about indexed data.

    Returns:
        Statistics from both SQLite and ChromaDB layers
    """
    result: dict = {}

    if sqlite_store:
        s = sqlite_store.stats()
        result["sqlite"] = {
            "files": s.files,
            "symbols": s.symbols,
            "objects": s.objects,
            "attributes": s.attributes,
        }
        result["edges"] = sqlite_store.count_edges()

    result["indexing_mode"] = config.indexing_mode

    if indexer:
        chroma = indexer.get_stats()
        result["chromadb"] = chroma["collections"]
        result["tracked_files"] = chroma.get("tracked_files", {})
        result["embedding_provider"] = chroma.get("embedding_provider", "")

    return result


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    """Health check endpoint."""
    from starlette.responses import JSONResponse

    response: dict = {"status": "healthy", "indexing_mode": config.indexing_mode}

    if sqlite_store:
        s = sqlite_store.stats()
        response["sqlite"] = {"symbols": s.symbols, "objects": s.objects}

    if indexer:
        chroma = indexer.get_stats()
        response["chromadb"] = chroma["collections"]
        response["embedding_provider"] = config.embedding_provider

    if not sqlite_store:
        return JSONResponse({"status": "initializing"}, status_code=503)

    return JSONResponse(response)


@mcp.custom_route("/reindex", methods=["POST"])
async def reindex_endpoint(request):
    """Trigger reindex via HTTP (SQLite + ChromaDB).

    Body (JSON, optional):
        force (bool): If true, clear ChromaDB and re-index everything using
                      cloud provider (openrouter). Default: false — incremental
                      reindex using file_tracker + local ollama.
    """
    from starlette.background import BackgroundTask
    from starlette.responses import JSONResponse

    if not sqlite_store:
        return JSONResponse({"error": "Services not initialized"}, status_code=503)

    try:
        body = await request.json()
        force = bool(body.get("force", False))
    except Exception:
        force = False

    def run_indexing(force: bool):
        try:
            if force:
                logger.info("Starting background FULL reindex (force=true, local provider)...")
                if sqlite_store:
                    _rebuild_sqlite()
                if indexer:
                    _swap_to_reindex_provider()
                    indexer.clear_all()
                    sqlite_has_data = sqlite_store.has_data() if sqlite_store else False
                    indexer.index_directory(sqlite_enabled=sqlite_has_data)
                logger.info(f"Background full reindex completed (provider: {config.reindex_provider})")
            else:
                logger.info("Starting background INCREMENTAL reindex (force=false, changed-files only)...")
                # Changed-file detector → update() only those (SQLite graph),
                # plus a best-effort vector delta. No full rebuild.
                res = _reindex_changed()
                logger.info(f"Background incremental reindex completed: {res}")
        except Exception as e:
            logger.error(f"Background reindex failed: {e}", exc_info=True)

    mode = "full (local)" if force else "incremental (local)"
    return JSONResponse(
        {"status": "started", "message": f"Reindex started in background ({mode})", "force": force},
        background=BackgroundTask(run_indexing, force),
    )


def main():
    """Entry point for the MCP server."""
    import uvicorn

    init_services()

    logger.info(f"Starting MCP server on {config.host}:{config.port}")

    transport = config.mcp_transport
    if transport == "sse":
        app = mcp.sse_app()
    else:
        app = mcp.http_app()

    logger.info(f"MCP transport: {transport}")

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
