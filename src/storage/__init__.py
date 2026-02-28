"""Storage layer — SQLite structural index.

Note: SQLiteStore is NOT re-exported here to avoid a circular import cycle:
  parsers.code → storage.models → storage.__init__ → sqlite_store → parsers.code
Import SQLiteStore directly: `from storage.sqlite_store import SQLiteStore`
"""

from .models import (
    BSLFunction,
    Attribute,
    TabPart,
    MetadataObject,
    FunctionInfo,
    FunctionContext,
    MetadataInfo,
    ObjectDetails,
    ReferenceInfo,
    IndexStats,
)

__all__ = [
    "BSLFunction",
    "Attribute",
    "TabPart",
    "MetadataObject",
    "FunctionInfo",
    "FunctionContext",
    "MetadataInfo",
    "ObjectDetails",
    "ReferenceInfo",
    "IndexStats",
]
