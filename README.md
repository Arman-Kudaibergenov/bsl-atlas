# BSL Atlas

[![Docker Hub](https://img.shields.io/docker/v/armankudaibergenov/bsl-atlas?label=Docker%20Hub&logo=docker)](https://hub.docker.com/r/armankudaibergenov/bsl-atlas)

Public MCP server for indexed 1C code search. It indexes XML/BSL sources exported from 1C Configurator and exposes structural and semantic search tools for AI assistants.

## Status

- Public/external codemetadata product
- Internal/private counterpart exists in a separate private repository
- This repo is for public onboarding and portable deployment, not private Jefest operator runbooks

## What it does

- Structural search via SQLite/FTS: functions, procedures, metadata objects, attributes, call graph
- Optional semantic search via ChromaDB embeddings
- Fast startup path with `INDEXING_MODE=fast`
- Reindex support after configuration dumps change

## Modes

| Mode | What you get | Requirements |
|------|---------------|--------------|
| `fast` | Structural search only | Docker, exported 1C sources |
| `full` | Structural + semantic search | Docker, exported 1C sources, embedding provider/API key |

`fast` is the default and is the recommended starting point.

## Important: Docker source mount is required

If you run `bsl-atlas` in Docker, the container must see your exported 1C sources through the `SOURCE_PATH -> /data/source` bind mount from `docker-compose.yml`.

- `SOURCE_PATH` is required for indexing real project files
- If the bind mount fails, `/data/source` exists but is empty and Atlas will report that the source directory is empty
- This is separate from RLM. Atlas needs the source mount because it reads XML/BSL files directly

## Quick Start

### 1. Export 1C sources

In 1C Configurator use `Configuration -> Dump configuration to files` and point it to an empty directory.

### 2. Download config files

```bash
curl -O https://raw.githubusercontent.com/Arman-Kudaibergenov/bsl-atlas/master/docker-compose.yml
curl -O https://raw.githubusercontent.com/Arman-Kudaibergenov/bsl-atlas/master/.env.example
cp .env.example .env
```

### 3. Configure `.env`

```env
SOURCE_PATH=C:\bsl-src
INDEXING_MODE=fast
```

For `full` mode also set an embedding provider and API key.

### 4. Start

```bash
docker compose up -d
```

### 5. Connect from Claude

Add to `claude_desktop_config.json` or project `.mcp.json`:

```json
{
  "mcpServers": {
    "bsl-atlas": {
      "type": "http",
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

## Windows notes

Docker Desktop on Windows can fail on paths with spaces or Cyrillic characters. If your real path looks like `C:\1C\Exports\My Config`, create an ASCII alias first and mount that path instead.

```powershell
cmd /c mklink /D C:\bsl-src "C:\1C\Exports\My Config"
```

Then set:

```env
SOURCE_PATH=C:\bsl-src
```

If Atlas reports that `SOURCE_PATH` is empty, the bind mount is wrong even if the folder exists inside the container.

## Supported layouts

The source path can point to any of these layouts:

```text
SOURCE_PATH/
  cf/
    Catalogs/
    Documents/
    CommonModules/
```

```text
SOURCE_PATH/
  Catalogs/
  Documents/
  CommonModules/
```

```text
SOURCE_PATH/
  cfe/
    MyExtension/
      Catalogs/
      CommonModules/
```

## Core tools

- `search_function(name)` - find a function or procedure by name
- `get_module_functions(path)` - list functions in a module
- `get_function_context(name)` - call graph
- `metadatasearch(query)` - search metadata objects
- `get_object_details(full_name)` - inspect object structure
- `codesearch(query)` - semantic search in `full` mode
- `helpsearch(query)` - semantic help search in `full` mode
- `reindex(force_chromadb)` - rebuild indexes after changes
- `stats()` - index statistics

## Reindex after changes

After you re-export the 1C configuration:

```bash
curl -X POST http://localhost:8000/reindex
```

## Embedding defaults

- Recommended family: `qwen3-embedding-4b`
- OpenRouter name: `qwen/qwen3-embedding-4b`
- Ollama name: `qwen3-embedding:4b`

## Project boundary

- `bsl-atlas` is the public product line
- `1c-enhanced-codemetadata` is the private/internal implementation line
- `AuditJefest` remains the private production/operator truth for the Jefest contour
