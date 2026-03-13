# BSL Atlas

[![Docker Hub](https://img.shields.io/docker/v/armankudaibergenov/bsl-atlas?label=Docker%20Hub&logo=docker)](https://hub.docker.com/r/armankudaibergenov/bsl-atlas)

MCP-сервер для 1С:Предприятие — векторный поиск, структурный индекс и граф вызовов в одном инструменте. Даёт AI-ассистентам мгновенный доступ к вашей конфигурации: находит функции, строит граф вызовов, ищет объекты метаданных и выполняет семантические запросы по BSL-коду — без чтения сырых файлов.

## Что умеет

- **Структурный поиск** (SQLite + FTS5, мгновенно): поиск функций по имени, список процедур модуля, граф вызовов (что вызывает что), поиск объектов метаданных (справочники, документы, регистры и др.)
- **Семантический поиск** (ChromaDB, векторный): найти код по описанию — "как реализовано проведение", "где логируются ошибки"
- **Два слоя**: SQLite пересобирается за секунды при старте; ChromaDB индексируется один раз в фоне через провайдер эмбеддингов на ваш выбор

## Два режима работы

| | **fast** (по умолчанию) | **full** |
|---|---|---|
| **Что работает** | Структурный поиск: функции, граф вызовов, метаданные | Всё из fast + семантический поиск (ChromaDB) |
| **API-ключ** | Не нужен | Нужен (OpenRouter, OpenAI, Ollama и др.) |
| **Запуск** | Мгновенно | SQLite сразу + векторизация в фоне |
| **Использование** | `INDEXING_MODE=fast` (или не задавать) | `INDEXING_MODE=full` |

**fast** — хороший старт: структурный поиск покрывает большинство задач. Переключитесь на **full** когда понадобится `codesearch` / `helpsearch`.

## Что нужно

**Режим fast:**
- Docker + Docker Compose
- 1С:Предприятие 8.3 (Конфигуратор для выгрузки конфигурации)

**Режим full (дополнительно):**
- API-ключ OpenRouter — [openrouter.ai/keys](https://openrouter.ai/keys) (или другой провайдер)

## Быстрый старт

### 1. Выгрузить конфигурацию

В Конфигураторе: **Конфигурация → Выгрузить конфигурацию в файлы**

Укажите пустую папку, например `C:\my-config\`. После выгрузки появятся сотни XML-файлов и `.bsl`-модулей.

### 2. Скачать конфиг и настроить

```bash
curl -O https://raw.githubusercontent.com/Arman-Kudaibergenov/bsl-atlas/master/docker-compose.yml
curl -O https://raw.githubusercontent.com/Arman-Kudaibergenov/bsl-atlas/master/.env.example
mv .env.example .env
```

Отредактировать `.env`:

```env
SOURCE_PATH=C:\my-config     # папка с выгрузкой конфигурации

# Режим fast (по умолчанию) — API-ключ не нужен:
INDEXING_MODE=fast

# Режим full — добавьте API-ключ:
# INDEXING_MODE=full
# OPENROUTER_API_KEY=sk-or-v1-...
```

### 3. Запустить

```bash
docker compose up -d
```

Образ скачается автоматически с Docker Hub (~500 МБ, один раз). SQLite проиндексируется сразу.
В режиме **full** ChromaDB векторизует в фоне — прогресс: `http://localhost:8000/health`.

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

### Режим индексации

```env
INDEXING_MODE=fast   # только SQLite, без API-ключа (по умолчанию)
INDEXING_MODE=full   # SQLite + ChromaDB векторы, нужен провайдер эмбеддингов
```

В режиме `fast` семантические инструменты (`codesearch`, `helpsearch`, `search_code_filtered`) возвращают подсказку включить `INDEXING_MODE=full`.

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
OLLAMA_MODEL=qwen3-embedding:4b  # лучшая модель для русского/BSL
```

`qwen3-embedding:4b` требует ~2.5 ГБ RAM. Скачать: `ollama pull qwen3-embedding:4b`

> **Бенчмарк:** qwen3-embedding:4b показала результаты, сопоставимые с полноразмерной моделью, при вдвое меньшем потреблении памяти — оптимальный выбор для большинства случаев.

### Модель OpenRouter

По умолчанию используется `qwen/qwen3-embedding-4b` — оптимизирована для русского языка и кириллического кода. Переопределить:

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

Сервер ожидает выгрузку конфигурации по пути `SOURCE_PATH`. Поддерживаются разные структуры — `cf/` подкаталог **не обязателен**:

```
# Вариант 1: выгрузка конфигурации (cf/)
SOURCE_PATH/
└── cf/
    ├── Catalogs/
    ├── Documents/
    └── CommonModules/

# Вариант 2: выгрузка расширения (файлы в корне)
SOURCE_PATH/
├── Catalogs/
├── Documents/
└── CommonModules/

# Вариант 3: расширения через cfe/
SOURCE_PATH/
└── cfe/
    └── МоёРасширение/
        ├── Catalogs/
        └── CommonModules/
```

Это стандартный результат **Конфигуратор → Выгрузить конфигурацию в файлы** (или расширение в файлы).

> **Пути с кириллицей и пробелами (Windows):** Docker Desktop может не смонтировать путь вроде `C:\1С\Выгрузки баз\УТ`. Если при запуске ошибка "SOURCE_PATH does not exist" — создайте символическую ссылку без пробелов: `mklink /D C:\config "C:\1С\Выгрузки баз\УТ"` и укажите `SOURCE_PATH=C:\config`.

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

## Благодарности

- [tree-sitter-bsl](https://github.com/alkoleft/tree-sitter-bsl) — грамматика Tree-sitter для языка 1С (BSL), используется для структурного парсинга кода
