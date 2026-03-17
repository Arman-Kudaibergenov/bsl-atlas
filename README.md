# BSL Atlas

[![Docker Hub](https://img.shields.io/docker/v/armankudaibergenov/bsl-atlas?label=Docker%20Hub&logo=docker)](https://hub.docker.com/r/armankudaibergenov/bsl-atlas)

Публичный MCP-сервер для быстрой индексации и поиска по исходникам 1С. Работает с XML/BSL-выгрузкой конфигурации или расширения и отдает структурный и, при необходимости, семантический поиск для AI-ассистентов.

## Что умеет

- искать функции, процедуры и модули через SQLite/FTS
- искать объекты метаданных, реквизиты и связи
- строить контекст по вызовам и структуре модулей
- работать в `fast` режиме без внешних embedding API
- переиндексировать проект после новой выгрузки

## Режимы

| Режим | Что дает | Что нужно |
|------|----------|------------|
| `fast` | быстрый структурный поиск | Docker и выгруженные исходники 1С |
| `full` | структурный + семантический поиск | Docker, исходники и embedding backend/API key |

`fast` — основной и рекомендуемый стартовый режим.

## Важно: mount исходников обязателен

Если вы запускаете `bsl-atlas` в Docker, контейнер обязан видеть реальные исходники проекта через bind mount `SOURCE_PATH -> /data/source`.

- `SOURCE_PATH` нужен для индексации файлов
- если bind mount настроен неверно, `/data/source` внутри контейнера может существовать, но будет пустым
- в этом случае Atlas честно сообщит, что каталог исходников пустой

Это отдельная тема от RLM: Atlas читает файлы проекта напрямую, поэтому без source mount индексировать нечего.

## Быстрый старт

### 1. Выгрузите исходники 1С

В конфигураторе используйте `Конфигурация -> Выгрузить конфигурацию в файлы` и укажите пустой каталог.

### 2. Скачайте конфиги

```bash
curl -O https://raw.githubusercontent.com/Arman-Kudaibergenov/bsl-atlas/master/docker-compose.yml
curl -O https://raw.githubusercontent.com/Arman-Kudaibergenov/bsl-atlas/master/.env.example
cp .env.example .env
```

### 3. Заполните `.env`

```env
SOURCE_PATH=C:\bsl-src
INDEXING_MODE=fast
```

Для `full` режима дополнительно укажите embedding provider и нужные ключи.

### 4. Запустите контейнер

```bash
docker compose up -d
```

### 5. Подключите MCP в Claude

Добавьте в `claude_desktop_config.json` или в `.mcp.json` проекта:

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

## Windows: что важно

Docker Desktop на Windows часто ломается на путях с пробелами или кириллицей. Если реальный путь выглядит как `C:\1С\Выгрузки\МояКонфигурация`, лучше сначала сделать ASCII-алиас.

```powershell
cmd /c mklink /D C:\bsl-src "C:\1С\Выгрузки\МояКонфигурация"
```

После этого в `.env` используйте:

```env
SOURCE_PATH=C:\bsl-src
```

Если Atlas пишет, что `SOURCE_PATH` пустой, проблема почти всегда в bind mount, а не в самом приложении.

## Поддерживаемые структуры

Каталог исходников может выглядеть так:

```text
SOURCE_PATH/
  cf/
    Catalogs/
    Documents/
    CommonModules/
```

или так:

```text
SOURCE_PATH/
  Catalogs/
  Documents/
  CommonModules/
```

или так:

```text
SOURCE_PATH/
  cfe/
    MyExtension/
      Catalogs/
      CommonModules/
```

## Основные инструменты

- `search_function(name)` — найти функцию или процедуру по имени
- `get_module_functions(path)` — список функций модуля
- `get_function_context(name)` — контекст вызовов
- `metadatasearch(query)` — поиск по объектам метаданных
- `get_object_details(full_name)` — структура объекта
- `codesearch(query)` — семантический поиск в `full` режиме
- `helpsearch(query)` — поиск по help/knowledge слою в `full` режиме
- `reindex(force_chromadb)` — переиндексация после изменений
- `stats()` — статистика индекса

## Переиндексация

После новой выгрузки исходников:

```bash
curl -X POST http://localhost:8000/reindex
```

## Embedding defaults

- рекомендуемое семейство: `qwen3-embedding-4b`
- OpenRouter: `qwen/qwen3-embedding-4b`
- Ollama: `qwen3-embedding:4b`
