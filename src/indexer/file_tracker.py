"""File tracking for incremental indexing.

Uses SQLite to track file hashes and detect changes.
"""

import hashlib
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

FileStatus = Literal["new", "modified", "unchanged", "deleted"]
IndexStatus = Literal["indexed", "failed", "skipped"]


class FileTracker:
    """Tracks file changes using content hashes stored in SQLite."""

    def __init__(self, db_path: Path):
        """Initialize file tracker.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            # Check if table exists and needs migration
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='file_hashes'"
            )
            table_exists = cursor.fetchone() is not None
            
            if table_exists:
                # Check if new columns exist
                cursor = conn.execute("PRAGMA table_info(file_hashes)")
                columns = {row[1] for row in cursor.fetchall()}
                
                # Add new columns if missing (migration)
                if "status" not in columns:
                    conn.execute("ALTER TABLE file_hashes ADD COLUMN status TEXT DEFAULT 'indexed'")
                    logger.info("Added 'status' column to file_hashes table")
                
                if "error_message" not in columns:
                    conn.execute("ALTER TABLE file_hashes ADD COLUMN error_message TEXT")
                    logger.info("Added 'error_message' column to file_hashes table")
                
                if "retry_count" not in columns:
                    conn.execute("ALTER TABLE file_hashes ADD COLUMN retry_count INTEGER DEFAULT 0")
                    logger.info("Added 'retry_count' column to file_hashes table")
            else:
                # Create new table with all columns
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS file_hashes (
                        path TEXT PRIMARY KEY,
                        hash TEXT NOT NULL,
                        indexed_at TIMESTAMP NOT NULL,
                        collection TEXT NOT NULL,
                        status TEXT DEFAULT 'indexed',
                        error_message TEXT,
                        retry_count INTEGER DEFAULT 0
                    )
                """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_collection 
                ON file_hashes(collection)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status 
                ON file_hashes(status)
            """)
            conn.commit()

    def _compute_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file content."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def get_file_status(self, file_path: Path, collection: str) -> FileStatus:
        """Check if file is new, modified, or unchanged.

        Args:
            file_path: Path to check
            collection: Collection name (metadata, code, help)

        Returns:
            Status: "new", "modified", or "unchanged"
        """
        if not file_path.exists():
            return "deleted"

        current_hash = self._compute_hash(file_path)
        path_str = str(file_path.resolve())

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT hash FROM file_hashes WHERE path = ? AND collection = ?",
                (path_str, collection),
            )
            row = cursor.fetchone()

        if row is None:
            return "new"
        elif row[0] != current_hash:
            return "modified"
        else:
            return "unchanged"

    def mark_indexed(self, file_path: Path, collection: str):
        """Mark file as indexed with current hash.

        Args:
            file_path: Path that was indexed
            collection: Collection name
        """
        current_hash = self._compute_hash(file_path)
        path_str = str(file_path.resolve())
        now = datetime.utcnow().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO file_hashes 
                (path, hash, indexed_at, collection, status, error_message, retry_count)
                VALUES (?, ?, ?, ?, 'indexed', NULL, 0)
                """,
                (path_str, current_hash, now, collection),
            )
            conn.commit()
    
    def mark_failed(self, file_path: Path, collection: str, error_message: str):
        """Mark file as failed to index.

        Args:
            file_path: Path that failed to index
            collection: Collection name
            error_message: Error description
        """
        current_hash = self._compute_hash(file_path)
        path_str = str(file_path.resolve())
        now = datetime.utcnow().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            # Get current retry count
            cursor = conn.execute(
                "SELECT retry_count FROM file_hashes WHERE path = ? AND collection = ?",
                (path_str, collection),
            )
            row = cursor.fetchone()
            retry_count = (row[0] + 1) if row else 1
            
            conn.execute(
                """
                INSERT OR REPLACE INTO file_hashes 
                (path, hash, indexed_at, collection, status, error_message, retry_count)
                VALUES (?, ?, ?, ?, 'failed', ?, ?)
                """,
                (path_str, current_hash, now, collection, error_message, retry_count),
            )
            conn.commit()
            logger.warning(f"Marked file as failed (retry {retry_count}): {path_str}")
    
    def mark_skipped(self, file_path: Path, collection: str, reason: str):
        """Mark file as skipped (e.g., filtered out).

        Args:
            file_path: Path that was skipped
            collection: Collection name
            reason: Reason for skipping
        """
        current_hash = self._compute_hash(file_path)
        path_str = str(file_path.resolve())
        now = datetime.utcnow().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO file_hashes 
                (path, hash, indexed_at, collection, status, error_message, retry_count)
                VALUES (?, ?, ?, ?, 'skipped', ?, 0)
                """,
                (path_str, current_hash, now, collection, reason),
            )
            conn.commit()
    
    def get_failed_files(self, collection: str, max_retries: int = 3) -> list[tuple[str, str, int]]:
        """Get files that failed indexing and haven't exceeded max retries.

        Args:
            collection: Collection name
            max_retries: Maximum number of retry attempts

        Returns:
            List of (path, error_message, retry_count) tuples
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT path, error_message, retry_count 
                FROM file_hashes 
                WHERE collection = ? AND status = 'failed' AND retry_count < ?
                ORDER BY retry_count ASC, indexed_at ASC
                """,
                (collection, max_retries),
            )
            return cursor.fetchall()

    def get_indexed_files(self, collection: str) -> set[str]:
        """Get all indexed file paths for a collection.

        Args:
            collection: Collection name

        Returns:
            Set of indexed file paths
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT path FROM file_hashes WHERE collection = ?",
                (collection,),
            )
            return {row[0] for row in cursor.fetchall()}

    def remove_deleted_files(self, collection: str, current_files: set[Path]) -> list[str]:
        """Remove entries for files that no longer exist.

        Args:
            collection: Collection name
            current_files: Set of currently existing file paths

        Returns:
            List of removed file paths
        """
        indexed = self.get_indexed_files(collection)
        current_paths = {str(f.resolve()) for f in current_files}
        to_remove = indexed - current_paths

        if to_remove:
            with sqlite3.connect(self.db_path) as conn:
                for path in to_remove:
                    conn.execute(
                        "DELETE FROM file_hashes WHERE path = ? AND collection = ?",
                        (path, collection),
                    )
                conn.commit()
            logger.info(f"Removed {len(to_remove)} deleted files from {collection}")

        return list(to_remove)

    def clear_collection(self, collection: str):
        """Clear all entries for a collection.

        Args:
            collection: Collection name to clear
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM file_hashes WHERE collection = ?",
                (collection,),
            )
            conn.commit()
        logger.info(f"Cleared all entries for collection: {collection}")

    def get_stats(self) -> dict[str, dict[str, int]]:
        """Get statistics about tracked files.

        Returns:
            Dict with counts per collection and status
            Example: {"metadata": {"indexed": 100, "failed": 2, "skipped": 5}}
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT collection, status, COUNT(*) 
                FROM file_hashes 
                GROUP BY collection, status
                """
            )
            
            stats = {}
            for collection, status, count in cursor.fetchall():
                if collection not in stats:
                    stats[collection] = {}
                stats[collection][status] = count
            
            return stats
