"""Hybrid search combining vector and fulltext search.

Ported from comol/1c_code_metadata_mcp with improvements.
"""

import logging
import re
from typing import Any, Protocol, runtime_checkable

import chromadb

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query."""
        ...


class HybridSearch:
    """Hybrid search with vector and fulltext capabilities."""

    # Query transformations for 1C terminology
    QUERY_TRANSFORMS = {
        "Справочник": "Справочники",
        "Документ": "Документы",
        "Перечисление": "Перечисления",
        "Отчет": "Отчеты",
        "Обработка": "Обработки",
        "РегистрСведений": "РегистрыСведений",
        "РегистрНакопления": "РегистрыНакопления",
        "ПланВидовХарактеристик": "ПланыВидовХарактеристик",
        "ПланСчетов": "ПланыСчетов",
        "ОбщийМодуль": "ОбщиеМодули",
    }

    def __init__(
        self,
        metadata_collection: chromadb.Collection,
        code_collection: chromadb.Collection,
        help_collection: chromadb.Collection,
        search_embedding_provider: EmbeddingProvider | None = None,
    ):
        """Initialize hybrid search.

        Args:
            metadata_collection: ChromaDB collection for metadata
            code_collection: ChromaDB collection for code
            help_collection: ChromaDB collection for help
            search_embedding_provider: Optional separate embedding provider for search queries.
                If provided, will be used instead of collection's embedding function.
        """
        self.metadata_collection = metadata_collection
        self.code_collection = code_collection
        self.help_collection = help_collection
        self.search_embedding_provider = search_embedding_provider
        
        if search_embedding_provider:
            logger.info("Using separate embedding provider for search queries")

    def _prepare_query(self, query: str) -> str:
        """Prepare query with 1C terminology transformations.

        Args:
            query: Original query string

        Returns:
            Transformed query
        """
        result = query

        # Apply transformations
        for singular, plural in self.QUERY_TRANSFORMS.items():
            # Replace singular forms with plural for better matching
            pattern = rf"\b{singular}(?!\w)"
            result = re.sub(pattern, plural, result)

        logger.debug(f"Query transformed: '{query}' -> '{result}'")
        return result

    def _is_single_word_query(self, query: str) -> bool:
        """Check if query is a single word (better for fulltext)."""
        words = query.split()
        return len(words) == 1

    def _perform_hybrid_search(
        self,
        collection: chromadb.Collection,
        query: str,
        limit: int = 10,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Perform hybrid search on a collection.

        Uses fulltext for single-word queries, vector for multi-word.

        Args:
            collection: ChromaDB collection to search
            query: Search query
            limit: Maximum results
            where: Optional ChromaDB metadata filter dict

        Returns:
            List of search results
        """
        prepared_query = self._prepare_query(query)
        results = []

        try:
            if self._is_single_word_query(prepared_query):
                # Fulltext search for single words (exact matching)
                logger.debug(f"Using fulltext search for: {prepared_query}")

                # Search in document content
                if self.search_embedding_provider:
                    # Use separate search provider for query embedding
                    query_embedding = self.search_embedding_provider.embed_query(prepared_query)
                    query_kwargs: dict[str, Any] = {
                        "query_embeddings": [query_embedding],
                        "n_results": limit,
                        "where_document": {"$contains": prepared_query},
                    }
                    if where is not None:
                        query_kwargs["where"] = where
                    fulltext_results = collection.query(**query_kwargs)
                else:
                    # Use collection's embedding function
                    query_kwargs = {
                        "query_texts": [prepared_query],
                        "n_results": limit,
                        "where_document": {"$contains": prepared_query},
                    }
                    if where is not None:
                        query_kwargs["where"] = where
                    fulltext_results = collection.query(**query_kwargs)
            else:
                # Vector search for multi-word queries (semantic)
                logger.debug(f"Using vector search for: {prepared_query}")

                if self.search_embedding_provider:
                    # Use separate search provider for query embedding
                    query_embedding = self.search_embedding_provider.embed_query(prepared_query)
                    query_kwargs = {
                        "query_embeddings": [query_embedding],
                        "n_results": limit,
                    }
                    if where is not None:
                        query_kwargs["where"] = where
                    fulltext_results = collection.query(**query_kwargs)
                else:
                    # Use collection's embedding function
                    query_kwargs = {
                        "query_texts": [prepared_query],
                        "n_results": limit,
                    }
                    if where is not None:
                        query_kwargs["where"] = where
                    fulltext_results = collection.query(**query_kwargs)

            # Process results
            if fulltext_results and fulltext_results.get("documents"):
                documents = fulltext_results["documents"][0]
                metadatas = fulltext_results.get("metadatas", [[]])[0]
                distances = fulltext_results.get("distances", [[]])[0]

                for i, doc in enumerate(documents):
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    distance = distances[i] if i < len(distances) else 1.0

                    results.append(
                        {
                            "content": doc,
                            "metadata": metadata,
                            "score": 1.0 - min(distance, 1.0),  # Convert distance to score
                            "full_path": metadata.get("full_path", ""),
                            "object_type": metadata.get("object_type", ""),
                            "name": metadata.get("name", ""),
                        }
                    )

        except Exception as e:
            logger.error(f"Search error: {e}")

        return results

    def search_metadata(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search metadata collection.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of search results
        """
        logger.info(f"Searching metadata for: {query}")
        return self._perform_hybrid_search(self.metadata_collection, query, limit)

    def search_code(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search code collection.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of search results
        """
        logger.info(f"Searching code for: {query}")
        return self._perform_hybrid_search(self.code_collection, query, limit)

    def search_code_filtered(
        self,
        query: str,
        module_type: str | None = None,
        only_export: bool = False,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search BSL code with structural filters.

        Combines ChromaDB semantic search with metadata filters.
        Filters by module_type and/or only_export if specified.

        Args:
            query: Search query
            module_type: Optional module type filter (e.g. "CommonModule", "ObjectModule")
            only_export: If True, restrict results to exported functions
            limit: Maximum results

        Returns:
            List of search results in the same format as search_code
        """
        where: dict[str, Any] | None = None

        if module_type and only_export:
            where = {
                "$and": [
                    {"module_type": {"$eq": module_type}},
                    {"is_export": {"$eq": True}},
                ]
            }
        elif module_type:
            where = {"module_type": {"$eq": module_type}}
        elif only_export:
            where = {"is_export": {"$eq": True}}

        logger.info(
            f"Searching code (filtered) for: {query!r}, "
            f"module_type={module_type!r}, only_export={only_export}"
        )
        return self._perform_hybrid_search(self.code_collection, query, limit, where=where)

    def search_help(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search help collection.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of search results
        """
        logger.info(f"Searching help for: {query}")
        return self._perform_hybrid_search(self.help_collection, query, limit)

    def search_all(self, query: str, limit: int = 10) -> dict[str, list[dict[str, Any]]]:
        """Search all collections.

        Args:
            query: Search query
            limit: Maximum results per collection

        Returns:
            Dict with results for each collection
        """
        return {
            "metadata": self.search_metadata(query, limit),
            "code": self.search_code(query, limit),
            "help": self.search_help(query, limit),
        }
