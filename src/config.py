"""Configuration management via environment variables."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

EmbeddingProvider = Literal["openai", "openrouter", "ollama", "cohere", "jina", "local"]
IndexingMode = Literal["fast", "full"]


@dataclass
class Config:
    """Application configuration."""

    # Paths
    source_path: Path = field(default_factory=lambda: Path(os.getenv("SOURCE_PATH", "/data/source")))
    chroma_path: Path = field(default_factory=lambda: Path(os.getenv("CHROMA_PATH", "/data/chroma_db")))
    
    # Embedding provider (legacy - for backward compatibility)
    embedding_provider: EmbeddingProvider = field(
        default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "openai")  # type: ignore
    )
    
    # Separate providers for indexing, search, and reindex
    indexing_provider: EmbeddingProvider = field(
        default_factory=lambda: os.getenv("INDEXING_PROVIDER") or os.getenv("EMBEDDING_PROVIDER", "openrouter")  # type: ignore
    )
    search_provider: EmbeddingProvider = field(
        default_factory=lambda: os.getenv("SEARCH_PROVIDER") or os.getenv("EMBEDDING_PROVIDER", "openrouter")  # type: ignore
    )
    reindex_provider: EmbeddingProvider = field(
        default_factory=lambda: os.getenv("REINDEX_PROVIDER") or os.getenv("SEARCH_PROVIDER") or "openrouter"  # type: ignore
    )

    # Ollama settings (optional, for hybrid setup)
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen3-embedding:4b"))
    
    # API Keys
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openrouter_api_key: str | None = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY"))
    cohere_api_key: str | None = field(default_factory=lambda: os.getenv("COHERE_API_KEY"))
    jina_api_key: str | None = field(default_factory=lambda: os.getenv("JINA_API_KEY"))
    
    # Model settings
    embedding_model: str | None = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL") or None)
    openai_api_base: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL") or os.getenv("EMBEDDING_BASE_URL") or None
    )

    # MCP transport
    mcp_transport: str = field(default_factory=lambda: os.getenv("MCP_TRANSPORT", "streamable-http"))
    
    # Indexing settings
    auto_index: bool = field(default_factory=lambda: os.getenv("AUTO_INDEX", "true").lower() == "true")
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "100")))
    max_batch_size: int = field(default_factory=lambda: int(os.getenv("MAX_BATCH_SIZE", "100")))
    
    # Parallel embedding settings
    embedding_concurrency: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_CONCURRENCY", "1")))
    embedding_batch_size: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_BATCH_SIZE", "10")))
    
    # Server settings
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8000")))
    
    # Search settings
    default_search_limit: int = field(default_factory=lambda: int(os.getenv("DEFAULT_SEARCH_LIMIT", "10")))

    # SQLite structural layer
    sqlite_db_path: Path = field(default_factory=lambda: Path(os.getenv("SQLITE_DB_PATH", "/data/bsl_index.db")))
    sqlite_auto_rebuild: bool = field(default_factory=lambda: os.getenv("SQLITE_AUTO_REBUILD", "true").lower() == "true")

    # Reranker (optional cross-encoder)
    reranker_url: str = field(default_factory=lambda: os.getenv("RERANKER_URL", ""))

    # Indexing mode: fast (SQLite only, no ChromaDB) or full (SQLite + ChromaDB vectors)
    indexing_mode: IndexingMode = field(
        default_factory=lambda: os.getenv("INDEXING_MODE", "fast")  # type: ignore
    )

    # ChromaDB indexing control (separate from SQLite)
    # CHROMADB_AUTO_INDEX defaults to AUTO_INDEX for backward compatibility
    chromadb_auto_index: bool = field(
        default_factory=lambda: os.getenv("CHROMADB_AUTO_INDEX", os.getenv("AUTO_INDEX", "false")).lower() == "true"
    )
    chromadb_schedule: str = field(default_factory=lambda: os.getenv("CHROMADB_SCHEDULE", ""))

    # Reranker service URL (optional cross-encoder reranking)
    reranker_url: str = field(default_factory=lambda: os.getenv("RERANKER_URL", ""))

    def get_api_key(self, provider: EmbeddingProvider | None = None) -> str | None:
        """Get API key for the specified or configured embedding provider.
        
        Args:
            provider: Provider name, defaults to embedding_provider
        """
        provider = provider or self.embedding_provider
        match provider:
            case "openai":
                return self.openai_api_key
            case "openrouter":
                return self.openrouter_api_key
            case "cohere":
                return self.cohere_api_key
            case "jina":
                return self.jina_api_key
            case "ollama" | "local":
                return None
        return None

    def validate(self) -> list[str]:
        """Validate configuration, return list of errors."""
        errors = []

        # In fast mode, skip ChromaDB provider validation (not used)
        if self.indexing_mode == "full":
            # Skip API key validation for providers using custom base_url (local models)
            has_custom_base = bool(self.openai_api_base)

            # Validate indexing provider
            if self.indexing_provider not in ("ollama", "local") and not self.get_api_key(self.indexing_provider):
                if not (self.indexing_provider == "openai" and has_custom_base):
                    errors.append(
                        f"API key required for indexing provider '{self.indexing_provider}'. "
                        f"Set {self.indexing_provider.upper()}_API_KEY environment variable."
                    )

            # Validate search provider
            if self.search_provider not in ("ollama", "local") and not self.get_api_key(self.search_provider):
                if not (self.search_provider == "openai" and has_custom_base):
                    errors.append(
                        f"API key required for search provider '{self.search_provider}'. "
                        f"Set {self.search_provider.upper()}_API_KEY environment variable."
                    )

        if not self.source_path.exists():
            errors.append(f"Source path does not exist: {self.source_path}")

        return errors


# Global config instance
config = Config()
