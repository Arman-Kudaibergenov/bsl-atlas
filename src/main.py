"""FastMCP server for 1C codebase search.

Dual-layer architecture:
- Structural layer (SQLite + FTS5): search_function, get_module_functions,
  get_function_context, metadatasearch, get_object_details — instant, no embeddings
- Semantic layer (ChromaDB): codesearch, helpsearch, search_code_filtered — vector search
"""

import asyncio
import json
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastmcp import FastMCP

from .config import config
from .indexer import VectorIndexer
from .indexer.embeddings import create_embedding_provider
from .parsers.metadata_xml import MetadataXMLParser
from .search import HybridSearch
from .storage.sqlite_store import SQLiteStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global instances
indexer: VectorIndexer | None = None
search: HybridSearch | None = None
sqlite_store: SQLiteStore | None = None


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

    # Collect metadata objects from XML dump
    xml_parser = MetadataXMLParser()
    metadata_objects = xml_parser.parse_directory(source)
    logger.info(f"Found {len(metadata_objects)} XML metadata objects")

    stats = sqlite_store.rebuild(bsl_files, metadata_objects)
    logger.info(
        f"SQLite rebuild complete: {stats.files} files, {stats.symbols} symbols, "
        f"{stats.objects} objects, {stats.attributes} attributes"
    )


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
        logger.info("SQLITE_AUTO_REBUILD=true, building structural index...")
        _rebuild_sqlite()
    else:
        existing = sqlite_store.stats()
        logger.info(
            f"SQLite: existing index has {existing.symbols} symbols, {existing.objects} objects"
        )

    # --- ChromaDB vector layer ---
    indexer = VectorIndexer(config)

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

    logger.info("Services initialized (SQLite ready, ChromaDB indexing in background)")


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


@mcp.tool()
def get_function_context(function_name: str) -> dict:
    """Get call graph context for a function: what it calls and who calls it.

    Args:
        function_name: Name of the function (e.g. "ПровестиДокумент")

    Returns:
        Dict with function info, list of called functions, and list of callers
    """
    if not sqlite_store:
        return {"error": "SQLite store not initialized"}

    ctx = sqlite_store.get_function_context(function_name)
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
# Semantic tools — ChromaDB layer
# ---------------------------------------------------------------------------


@mcp.tool()
def codesearch(query: str, limit: int = 10) -> list[dict]:
    """Search 1C BSL code semantically.

    Args:
        query: Search query (e.g. "ПроверитьЗаполнение", "расчет суммы договора")
        limit: Maximum number of results (default: 10)

    Returns:
        List of matching code fragments with module path and content
    """
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


@mcp.tool()
def helpsearch(query: str, limit: int = 10) -> list[dict]:
    """Search 1C documentation.

    Args:
        query: Search query (e.g. "Справочник создание", "Регистр сведений")
        limit: Maximum number of results (default: 10)

    Returns:
        List of matching documentation sections
    """
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


@mcp.tool()
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
# Utility tools
# ---------------------------------------------------------------------------


@mcp.tool()
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

    response: dict = {"status": "healthy"}

    if sqlite_store:
        s = sqlite_store.stats()
        response["sqlite"] = {"symbols": s.symbols, "objects": s.objects}

    if indexer:
        chroma = indexer.get_stats()
        response["chromadb"] = chroma["collections"]
        response["embedding_provider"] = config.embedding_provider

    if not indexer and not sqlite_store:
        return JSONResponse({"status": "initializing"}, status_code=503)

    return JSONResponse(response)


@mcp.custom_route("/reindex", methods=["POST"])
async def reindex_endpoint(request):
    """Trigger full reindex via HTTP (SQLite + ChromaDB)."""
    from starlette.background import BackgroundTask
    from starlette.responses import JSONResponse

    if not indexer and not sqlite_store:
        return JSONResponse({"error": "Services not initialized"}, status_code=503)

    def run_indexing():
        try:
            logger.info("Starting background full reindex...")
            if sqlite_store:
                _rebuild_sqlite()
            if indexer:
                _swap_to_reindex_provider()
                sqlite_has_data = sqlite_store.has_data() if sqlite_store else False
                indexer.index_directory(sqlite_enabled=sqlite_has_data)
            logger.info(f"Background reindex completed (provider: {config.reindex_provider})")
        except Exception as e:
            logger.error(f"Background reindex failed: {e}", exc_info=True)

    return JSONResponse(
        {"status": "started", "message": "Reindex started in background"},
        background=BackgroundTask(run_indexing),
    )


def main():
    """Entry point for the MCP server."""
    import uvicorn

    init_services()

    logger.info(f"Starting MCP server on {config.host}:{config.port}")

    uvicorn.run(
        mcp.http_app(),
        host=config.host,
        port=config.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
