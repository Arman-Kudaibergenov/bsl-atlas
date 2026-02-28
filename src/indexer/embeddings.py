"""Embedding providers abstraction with cloud API support."""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol, runtime_checkable

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of documents."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query."""
        ...


class OpenAIEmbeddings:
    """OpenAI embeddings using text-embedding-3-small.
    
    Also supports OpenAI-compatible APIs via base_url parameter.
    """

    def __init__(
        self, 
        api_key: str, 
        model: str = "text-embedding-3-small",
        base_url: str | None = None,
    ):
        from langchain_openai import OpenAIEmbeddings as LCOpenAIEmbeddings

        self.model = model
        self.base_url = base_url
        
        kwargs = {
            "api_key": api_key,
            "model": model,
        }
        if base_url:
            kwargs["base_url"] = base_url
            
        self._embeddings = LCOpenAIEmbeddings(**kwargs)
        
        provider_info = f" via {base_url}" if base_url else ""
        logger.info(f"Initialized OpenAI embeddings with model: {model}{provider_info}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents."""
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        return self._embeddings.embed_query(text)


class OpenRouterEmbeddings:
    """OpenRouter embeddings - OpenAI-compatible API with multiple models.
    
    Popular models for Russian text:
    - qwen/qwen3-embedding-8b (recommended)
    - openai/text-embedding-3-small
    - cohere/embed-multilingual-v3.0
    """

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self, 
        api_key: str, 
        model: str = "qwen/qwen3-embedding-8b",
    ):
        from langchain_openai import OpenAIEmbeddings as LCOpenAIEmbeddings

        self.model = model
        self._embeddings = LCOpenAIEmbeddings(
            api_key=api_key,
            model=model,
            base_url=self.OPENROUTER_BASE_URL,
        )
        logger.info(f"Initialized OpenRouter embeddings with model: {model}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents."""
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        return self._embeddings.embed_query(text)


class ParallelOpenRouterEmbeddings:
    """OpenRouter embeddings with parallel request processing.
    
    Sends multiple embedding requests in parallel to speed up indexing.
    Uses ThreadPoolExecutor for parallel HTTP requests with retry logic.
    """

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0  # seconds, will be multiplied by attempt number

    def __init__(
        self, 
        api_key: str, 
        model: str = "qwen/qwen3-embedding-8b",
        concurrency: int = 10,
        batch_size: int = 1,
    ):
        """Initialize parallel OpenRouter embeddings.
        
        Args:
            api_key: OpenRouter API key
            model: Model name
            concurrency: Number of parallel requests (default: 10)
            batch_size: Texts per single API request (default: 1)
        """
        import httpx
        
        self.api_key = api_key
        self.model = model
        self.concurrency = concurrency
        self.batch_size = batch_size
        self._client = httpx.Client(timeout=120.0)
        
        logger.info(
            f"Initialized ParallelOpenRouter embeddings: model={model}, "
            f"concurrency={concurrency}, batch_size={batch_size}"
        )

    def _embed_single_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a single batch of texts via API with retry logic."""
        import time
        
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self._client.post(
                    f"{self.OPENROUTER_BASE_URL}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "input": texts,
                    },
                )
                response.raise_for_status()
                data = response.json()
                
                # Validate response format
                if "data" not in data:
                    error_msg = f"Invalid API response format: missing 'data' field. Response: {data}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                if not isinstance(data["data"], list) or len(data["data"]) == 0:
                    error_msg = f"Invalid API response: 'data' is not a non-empty list. Response: {data}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                # Sort by index to maintain order
                sorted_data = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in sorted_data]
                
            except Exception as e:
                last_error = e
                # Log detailed error information
                error_details = {
                    "attempt": attempt + 1,
                    "max_retries": self.MAX_RETRIES,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "batch_size": len(texts),
                }
                
                # Add response details if available
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_details["status_code"] = e.response.status_code
                        error_details["response_body"] = e.response.text[:500]  # First 500 chars
                    except:
                        pass
                
                logger.error(f"Error embedding batch: {error_details}")
                
                if attempt < self.MAX_RETRIES - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    delay = self.RETRY_DELAY * (2 ** attempt)
                    logger.warning(
                        f"Retry {attempt + 1}/{self.MAX_RETRIES} after {type(e).__name__}. "
                        f"Waiting {delay}s..."
                    )
                    time.sleep(delay)
        
        # All retries failed - log final error
        logger.error(
            f"Failed to embed batch after {self.MAX_RETRIES} retries. "
            f"Last error: {type(last_error).__name__}: {last_error}"
        )
        raise last_error

    def _split_into_batches(self, texts: list[str]) -> list[list[str]]:
        """Split texts into batches."""
        return [
            texts[i:i + self.batch_size] 
            for i in range(0, len(texts), self.batch_size)
        ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents in parallel.
        
        Splits texts into batches and processes them concurrently.
        """
        if not texts:
            return []
        
        batches = self._split_into_batches(texts)
        total_batches = len(batches)
        
        if total_batches == 1:
            # Single batch, no need for parallelism
            return self._embed_single_batch(texts)
        
        logger.info(
            f"Processing {len(texts)} texts in {total_batches} batches "
            f"with concurrency={self.concurrency}"
        )
        
        # Use ThreadPoolExecutor for parallel requests
        results = [None] * total_batches
        
        def process_batch(idx: int, batch: list[str]) -> tuple[int, list[list[float]] | None]:
            """Process a single batch with graceful degradation.
            
            Returns:
                Tuple of (batch_index, embeddings or None if failed)
            """
            try:
                embeddings = self._embed_single_batch(batch)
                return (idx, embeddings)
            except Exception as e:
                logger.error(
                    f"Failed to process batch {idx} (size: {len(batch)}) after all retries. "
                    f"Error: {type(e).__name__}: {e}. Returning None for graceful degradation."
                )
                # Return None instead of raising - allows other batches to continue
                return (idx, None)
        
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = [
                executor.submit(process_batch, idx, batch)
                for idx, batch in enumerate(batches)
            ]
            
            for future in futures:
                idx, embeddings = future.result()
                results[idx] = embeddings
        
        # Flatten results maintaining order, skip failed batches
        all_embeddings = []
        failed_batches = []
        for idx, batch_embeddings in enumerate(results):
            if batch_embeddings is None:
                failed_batches.append(idx)
                # Add None placeholders for failed batch
                all_embeddings.extend([None] * len(batches[idx]))
            else:
                all_embeddings.extend(batch_embeddings)
        
        if failed_batches:
            logger.warning(
                f"Completed with {len(failed_batches)} failed batches: {failed_batches}. "
                f"Successfully embedded {len([e for e in all_embeddings if e is not None])}/{len(texts)} texts"
            )
        else:
            logger.info(f"Successfully completed embedding {len(all_embeddings)} texts")
        
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        return self._embed_single_batch([text])[0]


class CohereEmbeddings:
    """Cohere embeddings using embed-multilingual-v3.0."""

    def __init__(self, api_key: str, model: str = "embed-multilingual-v3.0"):
        from langchain_cohere import CohereEmbeddings as LCCohereEmbeddings

        self.model = model
        self._embeddings = LCCohereEmbeddings(
            cohere_api_key=api_key,
            model=model,
        )
        logger.info(f"Initialized Cohere embeddings with model: {model}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents."""
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        return self._embeddings.embed_query(text)


class JinaEmbeddings:
    """Jina AI embeddings using jina-embeddings-v3."""

    def __init__(self, api_key: str, model: str = "jina-embeddings-v3"):
        import httpx

        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.jina.ai/v1/embeddings"
        self._client = httpx.Client(timeout=60.0)
        logger.info(f"Initialized Jina embeddings with model: {model}")

    def _call_api(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        """Call Jina API for embeddings."""
        response = self._client.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": texts,
                "input_type": input_type,
            },
        )
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents."""
        return self._call_api(texts, input_type="document")

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        return self._call_api([text], input_type="query")[0]


class OllamaEmbeddings:
    """Ollama embeddings using local Ollama server.
    
    Supports any Ollama embedding model, recommended: qwen3-embedding:8b
    - 8192 dimensions
    - #1 MTEB multilingual (70.58)
    - 100+ languages support
    """

    def __init__(
        self, 
        model: str = "qwen3-embedding:8b",
        base_url: str = "http://localhost:11434",
    ):
        """Initialize Ollama embeddings.
        
        Args:
            model: Ollama model name (default: qwen3-embedding:8b)
            base_url: Ollama API endpoint (default: http://localhost:11434)
        """
        import httpx
        
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/api/embeddings"
        self._client = httpx.Client(timeout=120.0)
        
        logger.info(f"Initialized Ollama embeddings: model={model}, url={base_url}")
    
    def _call_api(self, text: str) -> list[float]:
        """Call Ollama API for a single text embedding.
        
        Ollama API format:
        Request: {"model": "qwen3-embedding:8b", "prompt": "text"}
        Response: {"embedding": [0.1, 0.2, ...]}
        """
        try:
            response = self._client.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": text,
                },
            )
            response.raise_for_status()
            data = response.json()
            
            if "embedding" not in data:
                error_msg = f"Invalid Ollama response: missing 'embedding' field. Response: {data}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            return data["embedding"]
            
        except Exception as e:
            logger.error(f"Error calling Ollama API: {type(e).__name__}: {e}")
            raise

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents.
        
        Note: Ollama API processes one text at a time, so we call it sequentially.
        For faster indexing, use OpenRouter with parallel processing.
        """
        if not texts:
            return []
        
        embeddings = []
        for i, text in enumerate(texts):
            try:
                embedding = self._call_api(text)
                embeddings.append(embedding)
                
                if (i + 1) % 10 == 0:
                    logger.debug(f"Embedded {i + 1}/{len(texts)} documents via Ollama")
                    
            except Exception as e:
                logger.error(f"Failed to embed document {i}: {e}")
                # Add None for failed embedding to maintain order
                embeddings.append(None)
        
        successful = len([e for e in embeddings if e is not None])
        logger.info(f"Ollama embedded {successful}/{len(texts)} documents")
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        return self._call_api(text)


class LocalEmbeddings:
    """Local embeddings using sentence-transformers (fallback)."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
        try:
            from langchain_huggingface import HuggingFaceEmbeddings

            self._embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
            )
            logger.info(f"Initialized local embeddings with model: {model_name}")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents."""
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        return self._embeddings.embed_query(text)


def create_embedding_provider(
    provider: str,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    concurrency: int = 1,
    batch_size: int = 10,
) -> EmbeddingProvider:
    """Factory function to create embedding provider.

    Args:
        provider: One of "openai", "openrouter", "ollama", "cohere", "jina", "local"
        api_key: API key for the provider (not needed for "ollama" or "local")
        model: Optional model name override
        base_url: Optional base URL override (for OpenAI-compatible APIs and Ollama)
        concurrency: Number of parallel requests (only for openrouter, default: 1)
        batch_size: Texts per API request (only for parallel openrouter, default: 10)

    Returns:
        EmbeddingProvider instance
    """
    match provider:
        case "openai":
            if not api_key:
                raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")
            return OpenAIEmbeddings(
                api_key=api_key,
                model=model or "text-embedding-3-small",
                base_url=base_url,
            )
        case "openrouter":
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY is required for OpenRouter embeddings")
            # Use parallel provider if concurrency > 1
            if concurrency > 1:
                logger.info(f"Using ParallelOpenRouterEmbeddings with concurrency={concurrency}, batch_size={batch_size}")
                return ParallelOpenRouterEmbeddings(
                    api_key=api_key,
                    model=model or "qwen/qwen3-embedding-8b",
                    concurrency=concurrency,
                    batch_size=batch_size,
                )
            return OpenRouterEmbeddings(
                api_key=api_key,
                model=model or "qwen/qwen3-embedding-8b",
            )
        case "ollama":
            return OllamaEmbeddings(
                model=model or "qwen3-embedding:8b",
                base_url=base_url or "http://localhost:11434",
            )
        case "cohere":
            if not api_key:
                raise ValueError("COHERE_API_KEY is required for Cohere embeddings")
            return CohereEmbeddings(
                api_key=api_key,
                model=model or "embed-multilingual-v3.0",
            )
        case "jina":
            if not api_key:
                raise ValueError("JINA_API_KEY is required for Jina embeddings")
            return JinaEmbeddings(
                api_key=api_key,
                model=model or "jina-embeddings-v3",
            )
        case "local":
            return LocalEmbeddings(
                model_name=model or "intfloat/multilingual-e5-small",
            )
        case _:
            raise ValueError(f"Unknown embedding provider: {provider}")


class ChromaDBEmbeddingFunction(EmbeddingFunction):
    """Wrapper to use EmbeddingProvider with ChromaDB."""

    def __init__(self, provider: EmbeddingProvider):
        self._provider = provider

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for ChromaDB."""
        return self._provider.embed_documents(list(input))
