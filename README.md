# BSL Atlas

[![Docker Hub](https://img.shields.io/docker/v/armankudaibergenov/bsl-atlas?label=Docker%20Hub&logo=docker)](https://hub.docker.com/r/armankudaibergenov/bsl-atlas)

MCP-сервер для 1С:Предприятие — векторный поиск, структурный индекс и граф вызовов в одном инструменте. Даёт AI-ассистентам мгновенный доступ к вашей конфигурации: находит функции, строит граф вызовов, ищет объекты метаданных и выполняет семантические запросы по BSL-коду — без чтения сырых файлов.

## Что умеет

- **Структурный поиск** (SQLite + FTS5, мгновенно): поиск функций по имени, список процедур модуля, граф вызовов (что вызывает что), поиск объектов метаданных (справочники, документы, регистры и др.)
- **Семантический поиск** (ChromaDB, векторный): найти код по описанию — "как реализовано проведение", "где логируются ошибки"
- **Два слоя**: SQLite пересобирается за секунды при старте; ChromaDB индексируется один раз в фоне через провайдер эмбеддингов на ваш выбор

## Что нужно

- Docker + Docker Compose
- 1С:Предприятие 8.3 (Конфигуратор для выгрузки конфигурации)
- API-ключ OpenRouter — [openrouter.ai/keys](https://openrouter.ai/keys)

## Быстрый старт

### 1. Выгрузить конфигурацию

В Конфигураторе: **Конфигурация → Выгрузить конфигурацию в файлы**

Укажите пустую папку, например `C:\my-config\`. После выгрузки появятся сотни XML-файлов и `.bsl`-модулей.

### 2. Клонировать и настроить

```bash
git clone https://github.com/your-username/bsl-atlas.git
cd bsl-atlas
cp .env.example .env
```

Отредактировать `.env`:

```env
SOURCE_PATH=C:\my-config     # папка с выгрузкой (содержит cf/)
OPENROUTER_API_KEY=sk-or-v1-...
```

### 3. Запустить

```bash
docker compose up -d
```

Образ скачается автоматически с Docker Hub (~500 МБ, один раз). SQLite проиндексируется сразу, ChromaDB векторизует в фоне — прогресс: `http://localhost:8000/health`.

### 4. Подключить к Claude

**Claude Desktop** — добавить в `claude_desktop_config.json`:

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

Расположение файла:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

**Claude Code** — добавить в `.mcp.json` в корне проекта:

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

---

## Инструменты MCP

### Структурные (SQLite — мгновенно)

| Инструмент | Что делает |
|-----------|-----------|
| `search_function(name)` | Найти функцию/процедуру по имени во всех модулях |
| `get_module_functions(path)` | Список всех процедур/функций модуля |
| `get_function_context(name)` | Граф вызовов: что вызывает функция и кто вызывает её |
| `metadatasearch(query)` | Полнотекстовый поиск по объектам метаданных |
| `get_object_details(full_name)` | Реквизиты, табличные части, измерения регистра |

### Семантические (ChromaDB — векторный поиск)

| Инструмент | Что делает |
|-----------|-----------|
| `codesearch(query)` | Поиск кода по описанию на естественном языке |
| `helpsearch(query)` | Поиск по проиндексированной справке |
| `search_code_filtered(query, object_type)` | Векторный поиск с фильтром (например, только Документы) |

### Утилиты

| Инструмент | Что делает |
|-----------|-----------|
| `reindex(force_chromadb)` | Перестроить индексы после изменений конфигурации |
| `stats()` | Статистика индекса: количество объектов, функций и др. |

---

## Настройка

Все параметры задаются через переменные окружения в `.env`.

### Провайдеры эмбеддингов

Сервер использует три отдельных провайдера — можно комбинировать:

| Переменная | Используется для | По умолчанию |
|-----------|-----------------|-------------|
| `INDEXING_PROVIDER` | Первоначальное заполнение ChromaDB (один раз) | `openrouter` |
| `SEARCH_PROVIDER` | Каждый поисковый запрос | `openrouter` |
| `REINDEX_PROVIDER` | Переиндексация после изменений кода | `openrouter` |

Поддерживаемые значения: `openrouter`, `openai`, `ollama`, `cohere`, `jina`

### Гибридная схема (рекомендуется если есть Ollama)

Если у вас запущен Ollama локально — поиск и переиндексация становятся бесплатными, облако используется только для первоначальной индексации:

```env
INDEXING_PROVIDER=openrouter    # облако, быстро, параллельно — один раз
SEARCH_PROVIDER=ollama          # бесплатный локальный инференс для каждого запроса
REINDEX_PROVIDER=ollama         # бесплатный локальный инференс для переиндексации

OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=qwen3-embedding:8b  # лучшая модель для русского/BSL
```

`qwen3-embedding:8b` требует ~5 ГБ RAM. Скачать: `ollama pull qwen3-embedding:8b`

### Модель OpenRouter

По умолчанию используется `qwen/qwen3-embedding-8b` — оптимизирована для русского языка и кириллического кода. Переопределить:

```env
EMBEDDING_MODEL=openai/text-embedding-3-small
```

### Параметры индексации

```env
AUTO_INDEX=true              # пересобирать SQLite при каждом старте
CHROMADB_AUTO_INDEX=true     # векторизовать при первом запуске; после — false
EMBEDDING_CONCURRENCY=5      # параллельные запросы к API (5 — безопасно, 10 — быстрее)
EMBEDDING_BATCH_SIZE=10      # текстов в одном запросе к API
```

> **После первого запуска** установите `CHROMADB_AUTO_INDEX=false` — векторный индекс сохраняется в папке `chroma_db/` рядом с `docker-compose.yml` (или по пути `CHROMA_PATH` если задан в `.env`). При повторном запуске индекс загружается из этой папки — повторная векторизация не нужна.

---

## Обновление индекса после изменений конфигурации

После повторной выгрузки конфигурации из 1С:

```bash
curl -X POST http://localhost:8000/reindex
```

Или через MCP-инструмент: `reindex(force_chromadb=True)` для обновления векторов.

---

## Структура директории с исходниками

Сервер ожидает выгрузку конфигурации по пути `SOURCE_PATH`. Ищет подпапку `cf/`:

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

Это стандартный результат **Конфигуратор → Выгрузить конфигурацию в файлы**.

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

## Лицензия

MIT
