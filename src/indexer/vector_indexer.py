"""Vector indexer with ChromaDB and cloud embeddings.

Provides incremental indexing and hybrid search capabilities.
"""

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import Config
from ..parsers import CodeParser, HelpParser, MetadataParser
from .embeddings import ChromaDBEmbeddingFunction, create_embedding_provider
from .file_tracker import FileTracker

logger = logging.getLogger(__name__)


class VectorIndexer:
    """Vector indexer for 1C codebase with cloud embeddings."""

    COLLECTION_METADATA = "metadata"
    COLLECTION_CODE = "code"
    COLLECTION_HELP = "help"

    def __init__(self, config: Config):
        """Initialize vector indexer.

        Args:
            config: Application configuration
        """
        self.config = config

        # Initialize embedding provider for indexing
        indexing_provider = config.indexing_provider
        base_url = config.ollama_base_url if indexing_provider == "ollama" else config.openai_api_base
        model = config.ollama_model if indexing_provider == "ollama" else config.embedding_model
        
        self.embedding_provider = create_embedding_provider(
            provider=indexing_provider,
            api_key=config.get_api_key(indexing_provider),
            model=model,
            base_url=base_url,
            concurrency=config.embedding_concurrency,
            batch_size=config.embedding_batch_size,
        )
        self.embedding_function = ChromaDBEmbeddingFunction(self.embedding_provider)

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(config.chroma_path),
            settings=Settings(anonymized_telemetry=False),
        )

        # Initialize collections
        self._init_collections()

        # Initialize parsers
        self.metadata_parser = MetadataParser()
        self.code_parser = CodeParser()
        self.help_parser = HelpParser()

        # Initialize file tracker
        tracker_db = config.chroma_path / "file_tracker.db"
        self.file_tracker = FileTracker(tracker_db)

        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        logger.info(
            f"Initialized VectorIndexer with indexing provider: {config.indexing_provider}"
        )

    def _init_collections(self):
        """Initialize ChromaDB collections."""
        self.metadata_collection = self.chroma_client.get_or_create_collection(
            name=self.COLLECTION_METADATA,
            embedding_function=self.embedding_function,
            metadata={"description": "1C metadata files"},
        )
        self.code_collection = self.chroma_client.get_or_create_collection(
            name=self.COLLECTION_CODE,
            embedding_function=self.embedding_function,
            metadata={"description": "1C BSL code files"},
        )
        self.help_collection = self.chroma_client.get_or_create_collection(
            name=self.COLLECTION_HELP,
            embedding_function=self.embedding_function,
            metadata={"description": "1C help documentation"},
        )

    def _chunk_document(self, content: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        """Split document into chunks with metadata.

        Args:
            content: Document content
            metadata: Document metadata

        Returns:
            List of chunk dicts with content and metadata
        """
        chunks = self.text_splitter.split_text(content)
        return [
            {
                "content": chunk,
                "metadata": {**metadata, "chunk_index": i},
            }
            for i, chunk in enumerate(chunks)
        ]

    def _add_to_collection_with_retry(
        self,
        collection: chromadb.Collection,
        ids: list[str],
        contents: list[str],
        metadatas: list[dict[str, Any]],
        max_retries: int = 3,
    ):
        """Add documents to ChromaDB collection with retry logic.

        Args:
            collection: ChromaDB collection
            ids: Document IDs
            contents: Document contents
            metadatas: Document metadata
            max_retries: Maximum retry attempts
        """
        import time
        
        last_error = None
        for attempt in range(max_retries):
            try:
                collection.add(
                    ids=ids,
                    documents=contents,
                    metadatas=metadatas,
                )
                return  # Success
                
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} adding to {collection.name}. "
                        f"Error: {type(e).__name__}. Waiting {delay}s..."
                    )
                    time.sleep(delay)
        
        # All retries failed
        logger.error(
            f"Failed to add batch to {collection.name} after {max_retries} retries. "
            f"Last error: {type(last_error).__name__}: {last_error}"
        )
        raise last_error

    def _add_to_collection(
        self,
        collection: chromadb.Collection,
        documents: list[dict[str, Any]],
        id_prefix: str,
    ):
        """Add documents to ChromaDB collection in batches.

        Args:
            collection: ChromaDB collection
            documents: List of document dicts
            id_prefix: Prefix for document IDs
        """
        if not documents:
            return

        batch_size = self.config.max_batch_size

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]

            ids = [f"{id_prefix}_{i + j}" for j in range(len(batch))]
            contents = [doc["content"] for doc in batch]
            metadatas = [doc["metadata"] for doc in batch]

            self._add_to_collection_with_retry(
                collection, ids, contents, metadatas
            )

    def index_metadata_file(self, file_path: Path) -> int:
        """Index a single metadata TXT file.

        Args:
            file_path: Path to metadata file

        Returns:
            Number of indexed chunks
        """
        # Check if should skip
        should_skip, skip_reason = self.should_skip_file(file_path)
        if should_skip:
            logger.debug(f"Skipping file: {file_path} - {skip_reason}")
            self.file_tracker.mark_skipped(file_path, self.COLLECTION_METADATA, skip_reason)
            return 0
        
        status = self.file_tracker.get_file_status(file_path, self.COLLECTION_METADATA)
        if status == "unchanged":
            logger.debug(f"Skipping unchanged file: {file_path}")
            return 0

        try:
            # Parse file
            objects = self.metadata_parser.parse_file(file_path)
            if not objects:
                return 0

            # Process each object
            all_chunks = []
            for obj in objects:
                # Build content string
                content_parts = [
                    f"Путь: {obj['full_path']}",
                    f"Тип: {obj['object_type']}",
                    f"Имя: {obj['name']}",
                ]

                # Add properties
                for key, value in obj.get("properties", {}).items():
                    content_parts.append(f"{key}: {value}")

                content = "\n".join(content_parts)

                # Chunk and add metadata
                chunks = self._chunk_document(
                    content,
                    {
                        "full_path": obj["full_path"],
                        "object_type": obj["object_type"],
                        "name": obj["name"],
                        "source_file": str(file_path),
                    },
                )
                all_chunks.extend(chunks)

            # Add to collection
            id_prefix = file_path.stem.replace(".", "_")
            self._add_to_collection(self.metadata_collection, all_chunks, id_prefix)

            # Mark as indexed
            self.file_tracker.mark_indexed(file_path, self.COLLECTION_METADATA)

            logger.info(f"Indexed metadata file {file_path}: {len(all_chunks)} chunks")
            return len(all_chunks)
            
        except Exception as e:
            logger.error(f"Failed to index metadata file {file_path}: {e}")
            self.file_tracker.mark_failed(file_path, self.COLLECTION_METADATA, str(e))
            raise

    def index_code_file(self, file_path: Path) -> int:
        """Index a single BSL code file.

        Args:
            file_path: Path to code file

        Returns:
            Number of indexed chunks
        """
        # Check if should skip
        should_skip, skip_reason = self.should_skip_file(file_path)
        if should_skip:
            logger.debug(f"Skipping file: {file_path} - {skip_reason}")
            self.file_tracker.mark_skipped(file_path, self.COLLECTION_CODE, skip_reason)
            return 0
        
        status = self.file_tracker.get_file_status(file_path, self.COLLECTION_CODE)
        if status == "unchanged":
            logger.debug(f"Skipping unchanged file: {file_path}")
            return 0

        try:
            # Parse file
            code_objects = self.code_parser.parse_file(file_path)
            if not code_objects:
                return 0

            all_chunks = []
            for obj in code_objects:
                content = obj.get("content", "")
                if not content:
                    continue

                # Add function list to content
                func_names = obj.get("function_names", [])
                if func_names:
                    content = f"Функции/Процедуры: {', '.join(func_names)}\n\n{content}"

                chunks = self._chunk_document(
                    content,
                    {
                        "full_path": obj["full_path"],
                        "object_type": obj["object_type"],
                        "name": obj["name"],
                        "source_file": str(file_path),
                        "functions": ",".join(func_names),
                    },
                )
                all_chunks.extend(chunks)

            # Add to collection
            id_prefix = file_path.stem.replace(".", "_")
            self._add_to_collection(self.code_collection, all_chunks, id_prefix)

            # Mark as indexed
            self.file_tracker.mark_indexed(file_path, self.COLLECTION_CODE)

            logger.info(f"Indexed code file {file_path}: {len(all_chunks)} chunks")
            return len(all_chunks)
            
        except Exception as e:
            logger.error(f"Failed to index code file {file_path}: {e}")
            self.file_tracker.mark_failed(file_path, self.COLLECTION_CODE, str(e))
            raise

    def index_help_file(self, file_path: Path) -> int:
        """Index a single HTML help file.

        Args:
            file_path: Path to help file

        Returns:
            Number of indexed chunks
        """
        # Check if should skip
        should_skip, skip_reason = self.should_skip_file(file_path)
        if should_skip:
            logger.debug(f"Skipping file: {file_path} - {skip_reason}")
            self.file_tracker.mark_skipped(file_path, self.COLLECTION_HELP, skip_reason)
            return 0
        
        status = self.file_tracker.get_file_status(file_path, self.COLLECTION_HELP)
        if status == "unchanged":
            logger.debug(f"Skipping unchanged file: {file_path}")
            return 0

        try:
            # Parse file
            help_objects = self.help_parser.parse_file(file_path)
            if not help_objects:
                return 0

            all_chunks = []
            for obj in help_objects:
                content = obj.get("content", "")
                if not content:
                    continue

                # Add title to content
                title = obj.get("title", "")
                if title:
                    content = f"# {title}\n\n{content}"

                chunks = self._chunk_document(
                    content,
                    {
                        "full_path": obj["full_path"],
                        "object_type": obj["object_type"],
                        "name": obj["name"],
                        "source_file": str(file_path),
                        "title": title,
                    },
                )
                all_chunks.extend(chunks)

            # Add to collection
            id_prefix = file_path.stem.replace(".", "_")
            self._add_to_collection(self.help_collection, all_chunks, id_prefix)

            # Mark as indexed
            self.file_tracker.mark_indexed(file_path, self.COLLECTION_HELP)

            logger.info(f"Indexed help file {file_path}: {len(all_chunks)} chunks")
            return len(all_chunks)
            
        except Exception as e:
            logger.error(f"Failed to index help file {file_path}: {e}")
            self.file_tracker.mark_failed(file_path, self.COLLECTION_HELP, str(e))
            raise

    def _collect_file_chunks(
        self, 
        file_path: Path, 
        collection_name: str,
        parser_method: str,
    ) -> list[dict[str, Any]]:
        """Collect chunks from a file without indexing.
        
        Args:
            file_path: Path to file
            collection_name: Name of collection for tracking
            parser_method: Parser method name ('metadata', 'code', 'help')
            
        Returns:
            List of chunk dicts with content and metadata
        """
        status = self.file_tracker.get_file_status(file_path, collection_name)
        if status == "unchanged":
            return []
            
        all_chunks = []
        
        if parser_method == "metadata":
            objects = self.metadata_parser.parse_file(file_path)
            if not objects:
                return []
            for obj in objects:
                content_parts = [
                    f"Путь: {obj['full_path']}",
                    f"Тип: {obj['object_type']}",
                    f"Имя: {obj['name']}",
                ]
                for key, value in obj.get("properties", {}).items():
                    content_parts.append(f"{key}: {value}")
                content = "\n".join(content_parts)
                chunks = self._chunk_document(
                    content,
                    {
                        "full_path": obj["full_path"],
                        "object_type": obj["object_type"],
                        "name": obj["name"],
                        "source_file": str(file_path),
                    },
                )
                all_chunks.extend(chunks)
                
        elif parser_method == "code":
            # Task-015: function-level chunking when parse_file_functions is available
            functions_indexed = False
            if hasattr(self.code_parser, "parse_file_functions"):
                try:
                    functions = self.code_parser.parse_file_functions(file_path)
                    if functions:
                        functions_indexed = True
                        for fn in functions:
                            # Skip trivially small functions
                            body_lines = fn.body.count("\n") + 1
                            if body_lines < 5:
                                continue
                            header = (
                                f"{'Функция' if fn.type == 'Функция' else 'Процедура'} "
                                f"{fn.name}({', '.join(fn.params)})"
                            )
                            if fn.is_export:
                                header += " Экспорт"
                            content = f"{header}\n\n{fn.body}"
                            chunks = self._chunk_document(
                                content,
                                {
                                    "full_path": fn.module_path,
                                    "object_type": "КодМодуля",
                                    "name": fn.name,
                                    "source_file": str(file_path),
                                    "functions": fn.name,
                                    "module_type": fn.module_type,
                                    "is_export": fn.is_export,
                                },
                            )
                            all_chunks.extend(chunks)
                except Exception as e:
                    logger.warning(f"parse_file_functions failed for {file_path}: {e}, falling back")
                    functions_indexed = False

            if not functions_indexed:
                # Fallback: index whole file
                code_objects = self.code_parser.parse_file(file_path)
                if not code_objects:
                    return []
                for obj in code_objects:
                    content = obj.get("content", "")
                    if not content:
                        continue
                    func_names = obj.get("function_names", [])
                    if func_names:
                        content = f"Функции/Процедуры: {', '.join(func_names)}\n\n{content}"
                    chunks = self._chunk_document(
                        content,
                        {
                            "full_path": obj["full_path"],
                            "object_type": obj["object_type"],
                            "name": obj["name"],
                            "source_file": str(file_path),
                            "functions": ",".join(func_names),
                            "module_type": "",
                            "is_export": False,
                        },
                    )
                    all_chunks.extend(chunks)
                
        elif parser_method == "help":
            help_objects = self.help_parser.parse_file(file_path)
            if not help_objects:
                return []
            for obj in help_objects:
                content = obj.get("content", "")
                if not content:
                    continue
                title = obj.get("title", "")
                if title:
                    content = f"# {title}\n\n{content}"
                chunks = self._chunk_document(
                    content,
                    {
                        "full_path": obj["full_path"],
                        "object_type": obj["object_type"],
                        "name": obj["name"],
                        "source_file": str(file_path),
                        "title": title,
                    },
                )
                all_chunks.extend(chunks)
        
        # Add file path to each chunk for tracking
        for chunk in all_chunks:
            chunk["_file_path"] = file_path
            
        return all_chunks

    # ChromaDB has a limit of ~5461 documents per add() call
    CHROMA_MAX_BATCH = 5000

    def _batch_index_chunks(
        self,
        collection: chromadb.Collection,
        chunks: list[dict[str, Any]],
        collection_name: str,
        id_prefix: str,
    ) -> int:
        """Index a batch of chunks with embedding, respecting ChromaDB limits.
        
        Args:
            collection: ChromaDB collection
            chunks: List of chunks to index
            collection_name: Collection name for file tracking
            id_prefix: Prefix for document IDs
            
        Returns:
            Number of indexed chunks
        """
        if not chunks:
            return 0
            
        # Extract unique files for tracking
        files_to_mark = set()
        for chunk in chunks:
            if "_file_path" in chunk:
                files_to_mark.add(chunk["_file_path"])
        
        total_indexed = 0
        
        # Split into sub-batches to respect ChromaDB limit
        for batch_idx in range(0, len(chunks), self.CHROMA_MAX_BATCH):
            batch_chunks = chunks[batch_idx:batch_idx + self.CHROMA_MAX_BATCH]
            
            # Prepare data for ChromaDB
            ids = [f"{id_prefix}_{batch_idx + i}" for i in range(len(batch_chunks))]
            contents = [chunk["content"] for chunk in batch_chunks]
            metadatas = [{k: v for k, v in chunk["metadata"].items()} for chunk in batch_chunks]
            
            try:
                # This will trigger embedding for batch_chunks at once
                collection.add(
                    ids=ids,
                    documents=contents,
                    metadatas=metadatas,
                )
                total_indexed += len(batch_chunks)
                logger.info(
                    f"Batch indexed {len(batch_chunks)} chunks to {collection.name} "
                    f"(sub-batch {batch_idx // self.CHROMA_MAX_BATCH + 1})"
                )
                
            except Exception as e:
                logger.error(f"Error batch indexing to {collection.name}: {e}")
                raise
        
        # Mark all files as indexed after successful indexing
        for file_path in files_to_mark:
            self.file_tracker.mark_indexed(file_path, collection_name)
                
        return total_indexed

    def _parallel_collect_chunks(
        self,
        files: list[Path],
        collection_name: str,
        parser_method: str,
        max_workers: int = 8,
    ) -> list[dict[str, Any]]:
        """Collect chunks from files in parallel.
        
        Args:
            files: List of file paths
            collection_name: Collection name for tracking
            parser_method: Parser method name
            max_workers: Number of parallel workers
            
        Returns:
            List of all chunks from all files
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        all_chunks = []
        errors = 0
        
        def process_file(file_path: Path) -> list[dict[str, Any]]:
            try:
                return self._collect_file_chunks(file_path, collection_name, parser_method)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                return []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_file, f): f for f in files}
            
            for future in as_completed(futures):
                chunks = future.result()
                all_chunks.extend(chunks)
        
        logger.info(f"Parallel collected {len(all_chunks)} chunks from {len(files)} files")
        return all_chunks

    def index_directory(self, directory: Path | None = None, sqlite_enabled: bool = False) -> dict[str, int]:
        """Index all files in directory using parallel batch processing.

        Args:
            directory: Directory to index (default: config.source_path)
            sqlite_enabled: If True, skip metadata TXT files (handled by SQLite layer)

        Returns:
            Dict with counts per collection
        """
        directory = directory or self.config.source_path
        logger.info(f"Starting parallel batch indexing of {directory}")

        stats = {"metadata": 0, "code": 0, "help": 0}

        if sqlite_enabled:
            logger.info("SQLite layer active — skipping metadata files from ChromaDB indexing")
        else:
            # Collect metadata files with parallel parsing
            logger.info("Collecting metadata files (parallel)...")
            metadata_files = list(directory.rglob("*.txt"))
            logger.info(f"Found {len(metadata_files)} metadata files")

            metadata_chunks = self._parallel_collect_chunks(
                metadata_files, self.COLLECTION_METADATA, "metadata"
            )

            # Index metadata in sub-batches
            if metadata_chunks:
                stats["metadata"] = self._batch_index_chunks(
                    self.metadata_collection,
                    metadata_chunks,
                    self.COLLECTION_METADATA,
                    "metadata_all",
                )

        # Collect code files with parallel parsing
        logger.info("Collecting code files (parallel)...")
        code_files = list(directory.rglob("*.bsl"))
        logger.info(f"Found {len(code_files)} code files")
        
        code_chunks = self._parallel_collect_chunks(
            code_files, self.COLLECTION_CODE, "code"
        )
        
        # Index code in sub-batches
        if code_chunks:
            stats["code"] = self._batch_index_chunks(
                self.code_collection,
                code_chunks,
                self.COLLECTION_CODE,
                "code_all",
            )

        # Collect help files with parallel parsing
        logger.info("Collecting help files (parallel)...")
        help_files = list(directory.rglob("*.html")) + list(directory.rglob("*.htm"))
        logger.info(f"Found {len(help_files)} help files")
        
        help_chunks = self._parallel_collect_chunks(
            help_files, self.COLLECTION_HELP, "help"
        )
        
        # Index help in sub-batches
        if help_chunks:
            stats["help"] = self._batch_index_chunks(
                self.help_collection,
                help_chunks,
                self.COLLECTION_HELP,
                "help_final",
            )

        logger.info(
            f"Indexing complete: {stats['metadata']} metadata, "
            f"{stats['code']} code, {stats['help']} help chunks"
        )
        return stats
    
    def should_skip_file(self, file_path: Path) -> tuple[bool, str | None]:
        """Check if file should be skipped based on filter patterns.
        
        Args:
            file_path: Path to check
            
        Returns:
            Tuple of (should_skip, reason)
        """
        # Skip patterns - add more as needed
        skip_patterns = [
            "**/РегламентныеОтчеты/**",
            "**/РегламентныеОтчёты/**",
            "**/.git/**",
            "**/__pycache__/**",
            "**/node_modules/**",
        ]
        
        path_str = str(file_path)
        for pattern in skip_patterns:
            import fnmatch
            if fnmatch.fnmatch(path_str, pattern):
                return True, f"Matched skip pattern: {pattern}"
        
        return False, None
    
    def retry_failed_files(self, max_retries: int = 3) -> dict[str, int]:
        """Retry indexing files that previously failed.
        
        Args:
            max_retries: Maximum retry attempts per file
            
        Returns:
            Dict with retry statistics per collection
        """
        logger.info(f"Starting retry of failed files (max_retries={max_retries})")
        
        stats = {"metadata": 0, "code": 0, "help": 0, "errors": 0}
        
        # Retry metadata files
        failed_metadata = self.file_tracker.get_failed_files(
            self.COLLECTION_METADATA, max_retries
        )
        logger.info(f"Found {len(failed_metadata)} failed metadata files to retry")
        
        for path_str, error_msg, retry_count in failed_metadata:
            file_path = Path(path_str)
            if not file_path.exists():
                logger.warning(f"Skipping deleted file: {file_path}")
                continue
            
            try:
                logger.info(f"Retrying metadata file (attempt {retry_count + 1}): {file_path}")
                chunks = self.index_metadata_file(file_path)
                stats["metadata"] += chunks
            except Exception as e:
                logger.error(f"Retry failed for {file_path}: {e}")
                self.file_tracker.mark_failed(
                    file_path, self.COLLECTION_METADATA, str(e)
                )
                stats["errors"] += 1
        
        # Retry code files
        failed_code = self.file_tracker.get_failed_files(
            self.COLLECTION_CODE, max_retries
        )
        logger.info(f"Found {len(failed_code)} failed code files to retry")
        
        for path_str, error_msg, retry_count in failed_code:
            file_path = Path(path_str)
            if not file_path.exists():
                logger.warning(f"Skipping deleted file: {file_path}")
                continue
            
            try:
                logger.info(f"Retrying code file (attempt {retry_count + 1}): {file_path}")
                chunks = self.index_code_file(file_path)
                stats["code"] += chunks
            except Exception as e:
                logger.error(f"Retry failed for {file_path}: {e}")
                self.file_tracker.mark_failed(
                    file_path, self.COLLECTION_CODE, str(e)
                )
                stats["errors"] += 1
        
        # Retry help files
        failed_help = self.file_tracker.get_failed_files(
            self.COLLECTION_HELP, max_retries
        )
        logger.info(f"Found {len(failed_help)} failed help files to retry")
        
        for path_str, error_msg, retry_count in failed_help:
            file_path = Path(path_str)
            if not file_path.exists():
                logger.warning(f"Skipping deleted file: {file_path}")
                continue
            
            try:
                logger.info(f"Retrying help file (attempt {retry_count + 1}): {file_path}")
                chunks = self.index_help_file(file_path)
                stats["help"] += chunks
            except Exception as e:
                logger.error(f"Retry failed for {file_path}: {e}")
                self.file_tracker.mark_failed(
                    file_path, self.COLLECTION_HELP, str(e)
                )
                stats["errors"] += 1
        
        logger.info(
            f"Retry complete: {stats['metadata']} metadata, {stats['code']} code, "
            f"{stats['help']} help chunks, {stats['errors']} errors"
        )
        return stats

    def get_stats(self) -> dict[str, Any]:
        """Get indexer statistics.

        Returns:
            Dict with collection counts and file tracker stats
        """
        return {
            "collections": {
                "metadata": self.metadata_collection.count(),
                "code": self.code_collection.count(),
                "help": self.help_collection.count(),
            },
            "tracked_files": self.file_tracker.get_stats(),
            "embedding_provider": self.config.embedding_provider,
        }

    def clear_all(self):
        """Clear all indexed data."""
        self.chroma_client.delete_collection(self.COLLECTION_METADATA)
        self.chroma_client.delete_collection(self.COLLECTION_CODE)
        self.chroma_client.delete_collection(self.COLLECTION_HELP)
        self._init_collections()

        self.file_tracker.clear_collection(self.COLLECTION_METADATA)
        self.file_tracker.clear_collection(self.COLLECTION_CODE)
        self.file_tracker.clear_collection(self.COLLECTION_HELP)

        logger.info("Cleared all indexed data")
