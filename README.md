# 1C Codemetadata MCP Server

MCP server for structural and semantic search across 1C:Enterprise codebase. Gives AI assistants instant access to your configuration: find functions, trace call graphs, search by metadata object names, and run semantic queries over BSL code — all without reading raw files.

## What it does

- **Structural search** (SQLite + FTS5, instant): find functions by name, list all procedures in a module, trace what calls what, search metadata objects (catalogs, documents, registers, etc.)
- **Semantic search** (ChromaDB, vector): find code by description — "how posting is implemented", "where error logging happens"
- **Dual-layer**: SQLite rebuilds in seconds on startup; ChromaDB indexes once in the background via your embedding provider of choice

## Prerequisites

- Docker + Docker Compose
- 1C:Enterprise 8.3 (Configurator to export the config)
- OpenRouter API key — [get one free](https://openrouter.ai/keys)

## Quick start

### 1. Export your 1C configuration

In 1C Configurator: **Configuration → Dump config to files (Выгрузить конфигурацию в файлы)**

Choose an empty directory, e.g. `C:\my-config\cf\`. After export you'll have hundreds of XML files and `.bsl` modules.

### 2. Clone and configure

```bash
git clone https://github.com/your-username/1c-codemetadata.git
cd 1c-codemetadata
cp .env.example .env
```

Edit `.env`:

```env
SOURCE_PATH=C:\my-config     # path to the directory containing cf/ subdir
OPENROUTER_API_KEY=sk-or-v1-...
```

### 3. Start

```bash
docker compose up -d
```

First start takes 2–5 minutes: SQLite indexes immediately, ChromaDB vectorizes in the background (progress at `http://localhost:8000/health`).

### 4. Connect to Claude

**Claude Desktop** — add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "1c-codemetadata": {
      "type": "http",
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

Config file location:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

**Claude Code** — add to `.mcp.json` in your project root:

```json
{
  "mcpServers": {
    "1c-codemetadata": {
      "type": "http",
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

---

## Available MCP tools

### Structural (SQLite — instant)

| Tool | What it does |
|------|-------------|
| `search_function(name)` | Find function/procedure by name across all modules |
| `get_module_functions(path)` | List all procedures/functions in a module |
| `get_function_context(name)` | Call graph: what this function calls and who calls it |
| `metadatasearch(query)` | Full-text search across metadata objects |
| `get_object_details(full_name)` | Attributes, tabular sections, register dimensions for an object |

### Semantic (ChromaDB — vector)

| Tool | What it does |
|------|-------------|
| `codesearch(query)` | Find code by natural language description |
| `helpsearch(query)` | Search indexed help content |
| `search_code_filtered(query, object_type)` | Filtered vector search (e.g. only Documents) |

### Utility

| Tool | What it does |
|------|-------------|
| `reindex(force_chromadb)` | Rebuild indexes after config changes |
| `stats()` | Index statistics: object count, function count, etc. |

---

## Configuration

All settings via environment variables (set in `.env`):

### Embedding providers

The server uses three separate providers for different operations — you can mix and match:

| Variable | Used for | Default |
|----------|----------|---------|
| `INDEXING_PROVIDER` | Initial ChromaDB bulk fill (runs once) | `openrouter` |
| `SEARCH_PROVIDER` | Every search query | `openrouter` |
| `REINDEX_PROVIDER` | Incremental reindex after code changes | `openrouter` |

Supported values: `openrouter`, `openai`, `ollama`, `cohere`, `jina`

### Hybrid setup (recommended if you have Ollama)

If you run Ollama locally, you can make search and reindex free — only the initial indexing uses cloud API:

```env
INDEXING_PROVIDER=openrouter    # cloud, fast, parallel — used once
SEARCH_PROVIDER=ollama          # free local inference for every query
REINDEX_PROVIDER=ollama         # free local inference for reindex

OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=qwen3-embedding:8b  # best for Russian/BSL
```

`qwen3-embedding:8b` requires ~5 GB RAM. Pull it: `ollama pull qwen3-embedding:8b`

### OpenRouter model

Default is `qwen/qwen3-embedding-8b` — optimized for Russian and Cyrillic code. Override:

```env
EMBEDDING_MODEL=openai/text-embedding-3-small
```

### Indexing settings

```env
AUTO_INDEX=true              # rebuild SQLite index on every startup
CHROMADB_AUTO_INDEX=true     # vectorize on first start; set false afterwards
EMBEDDING_CONCURRENCY=5      # parallel embedding requests (5 = safe, 10 = faster)
EMBEDDING_BATCH_SIZE=10      # texts per API request
```

> **After the first run** set `CHROMADB_AUTO_INDEX=false` — the vector index persists in `chroma_db/` and only needs updating when your config changes.

---

## Updating the index after config changes

When you re-export your 1C config and want to reflect changes:

```bash
# Rebuild SQLite (instant) + optionally re-vectorize
curl -X POST http://localhost:8000/reindex
```

Or via MCP tool: `reindex(force_chromadb=True)` to also update vectors.

---

## Source directory structure

The server expects your config export at `SOURCE_PATH`. It looks for `cf/` subdirectory:

```
SOURCE_PATH/
└── cf/
    ├── Catalogs/
    │   ├── Контрагенты.xml
    │   └── Контрагенты/Ext/ObjectModule.bsl
    ├── Documents/
    ├── CommonModules/
    └── ...
```

This is the standard output of **Configurator → Dump config to files**.

---

## Health check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "sqlite": {"objects": 345, "functions": 1240},
  "chromadb": {"indexed": 1240, "status": "ready"}
}
```

---

## License

MIT
