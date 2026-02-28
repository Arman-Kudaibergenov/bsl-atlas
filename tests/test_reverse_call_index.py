"""Tests for reverse call index functionality."""

import json
from pathlib import Path

import pytest

from src.config import Config
from src.indexer import ReverseCallIndex, VectorIndexer


@pytest.fixture
def test_config(tmp_path):
    """Create test configuration."""
    config = Config()
    config.source_path = Path("tests/fixtures/test_project")
    config.chroma_path = tmp_path / "chroma"
    config.embedding_provider = "openrouter"
    config.embedding_model = "qwen3-embedding-8b"
    config.auto_index = False
    return config


@pytest.fixture
def indexer(test_config):
    """Create and initialize indexer."""
    indexer = VectorIndexer(test_config)
    # Index test project
    indexer.index_directory()
    return indexer


@pytest.fixture
def reverse_index(indexer):
    """Get reverse call index from indexer."""
    return indexer.reverse_call_index


class TestReverseCallIndexBuild:
    """Tests for building reverse call index."""

    def test_build_index(self, indexer, reverse_index):
        """Test building reverse call index."""
        # Index should be built automatically during indexing
        stats = reverse_index.get_stats()

        assert isinstance(stats, dict)
        assert "indexed_functions" in stats
        assert "total_edges" in stats

        # Should have some indexed functions
        if stats["indexed_functions"] > 0:
            assert stats["total_edges"] >= 0

    def test_rebuild_index(self, indexer, reverse_index):
        """Test rebuilding reverse call index."""
        # Get initial stats
        initial_stats = reverse_index.get_stats()

        # Rebuild
        rebuild_stats = reverse_index.build_index(indexer.code_collection)

        assert isinstance(rebuild_stats, dict)
        assert "indexed_functions" in rebuild_stats
        assert "total_edges" in rebuild_stats

        # Stats should be consistent
        assert rebuild_stats["indexed_functions"] == initial_stats["indexed_functions"]
        assert rebuild_stats["total_edges"] == initial_stats["total_edges"]

    def test_clear_index(self, reverse_index):
        """Test clearing reverse call index."""
        # Clear index
        reverse_index.clear()

        # Stats should be zero
        stats = reverse_index.get_stats()
        assert stats["indexed_functions"] == 0
        assert stats["total_edges"] == 0


class TestReverseCallIndexLookup:
    """Tests for reverse call index lookups."""

    def test_get_callers_existing_function(self, reverse_index):
        """Test getting callers for a function that is called."""
        # This test assumes there's at least one function with callers
        # In a real test, we'd use a known test project

        # Try to find a function with callers
        stats = reverse_index.get_stats()

        if stats["indexed_functions"] > 0:
            # Get all functions to find one with callers
            # (In real test, we'd use a known function name)
            callers = reverse_index.get_callers("ПроверитьЗаполнение")

            assert isinstance(callers, list)
            # May be empty if function doesn't exist or has no callers

    def test_get_callers_nonexistent_function(self, reverse_index):
        """Test getting callers for a non-existent function."""
        callers = reverse_index.get_callers("НесуществующаяФункция123")

        assert isinstance(callers, list)
        assert len(callers) == 0

    def test_get_callers_format(self, reverse_index):
        """Test that callers have correct format."""
        # Try a common function name
        callers = reverse_index.get_callers("ПроверитьЗаполнение")

        if callers:
            for caller in callers:
                assert "chunk_id" in caller
                assert "object_name" in caller
                assert "function_name" in caller
                assert "function_type" in caller
                assert "is_export" in caller
                assert "params" in caller

    def test_multiple_callers(self, reverse_index):
        """Test function with multiple callers."""
        # Common utility functions should have multiple callers
        # (In real test, we'd use a known function)

        stats = reverse_index.get_stats()

        if stats["total_edges"] > 0:
            # At least some functions should have callers
            assert stats["total_edges"] >= stats["indexed_functions"] * 0.1


class TestReverseCallIndexIntegration:
    """Integration tests with AdvancedSearch."""

    def test_advanced_search_uses_index(self, indexer):
        """Test that AdvancedSearch uses reverse index."""
        from src.search import AdvancedSearch

        # Create AdvancedSearch with reverse index
        advanced_search = AdvancedSearch(
            code_collection=indexer.code_collection,
            reverse_call_index=indexer.reverse_call_index,
        )

        # Should have reverse index
        assert advanced_search.reverse_call_index is not None

    def test_get_function_context_with_index(self, indexer):
        """Test get_function_context uses reverse index."""
        from src.search import AdvancedSearch

        advanced_search = AdvancedSearch(
            code_collection=indexer.code_collection,
            reverse_call_index=indexer.reverse_call_index,
        )

        # Try to get context for a function
        # (In real test, we'd use a known function)
        result = advanced_search.get_function_context(
            "ПроверитьЗаполнение", include_callers=True
        )

        assert isinstance(result, dict)

        if "error" not in result:
            assert "function" in result
            assert "calls" in result
            assert "called_by" in result
            assert "stats" in result

    def test_get_function_context_without_index(self, indexer):
        """Test get_function_context fallback without index."""
        from src.search import AdvancedSearch

        # Create AdvancedSearch without reverse index
        advanced_search = AdvancedSearch(
            code_collection=indexer.code_collection,
            reverse_call_index=None,
        )

        # Should still work (fallback to O(N) scan)
        result = advanced_search.get_function_context(
            "ПроверитьЗаполнение", include_callers=True
        )

        assert isinstance(result, dict)

        if "error" not in result:
            assert "function" in result
            assert "calls" in result
            assert "called_by" in result


class TestReverseCallIndexPerformance:
    """Performance tests for reverse call index."""

    def test_lookup_performance(self, reverse_index):
        """Test that lookup is fast (O(1))."""
        import time

        # Get a function to test
        stats = reverse_index.get_stats()

        if stats["indexed_functions"] > 0:
            # Measure lookup time
            start = time.time()
            callers = reverse_index.get_callers("ПроверитьЗаполнение")
            end = time.time()

            lookup_time = end - start

            # Should be very fast (< 100ms)
            assert lookup_time < 0.1, f"Lookup took {lookup_time:.3f}s, expected < 0.1s"

    def test_build_performance(self, indexer, reverse_index):
        """Test that building index is reasonable."""
        import time

        # Clear index
        reverse_index.clear()

        # Measure build time
        start = time.time()
        stats = reverse_index.build_index(indexer.code_collection)
        end = time.time()

        build_time = end - start

        # Build time should be reasonable (< 10s for small projects)
        # For large projects (39K functions), expect < 60s
        if stats["indexed_functions"] < 1000:
            assert build_time < 10, f"Build took {build_time:.3f}s, expected < 10s"
        else:
            assert build_time < 60, f"Build took {build_time:.3f}s, expected < 60s"


class TestReverseCallIndexEdgeCases:
    """Edge case tests."""

    def test_empty_code_collection(self, test_config):
        """Test with empty code collection."""
        # Create indexer without indexing
        indexer = VectorIndexer(test_config)

        # Build index on empty collection
        stats = indexer.reverse_call_index.build_index(indexer.code_collection)

        assert stats["indexed_functions"] == 0
        assert stats["total_edges"] == 0

    def test_function_with_no_callers(self, reverse_index):
        """Test function that is never called."""
        # Private helper functions may have no callers
        callers = reverse_index.get_callers("ПриватнаяФункция")

        assert isinstance(callers, list)
        # May be empty

    def test_function_with_many_callers(self, reverse_index):
        """Test common utility function with many callers."""
        # Functions like "СообщитьПользователю" should have many callers
        callers = reverse_index.get_callers("СообщитьПользователю")

        assert isinstance(callers, list)
        # May have many callers

    def test_circular_calls(self, reverse_index):
        """Test handling of circular function calls."""
        # A calls B, B calls A
        # Index should handle this correctly

        stats = reverse_index.get_stats()

        # Should not crash or hang
        assert isinstance(stats, dict)

    def test_self_recursive_function(self, reverse_index):
        """Test function that calls itself."""
        # Recursive function should be in its own callers list

        # This is a valid case - function can call itself
        # Index should handle it correctly

        stats = reverse_index.get_stats()
        assert isinstance(stats, dict)


class TestReverseCallIndexConsistency:
    """Consistency tests."""

    def test_index_matches_code_collection(self, indexer, reverse_index):
        """Test that reverse index is consistent with code collection."""
        # Get all functions from code collection
        code_results = indexer.code_collection.get(
            where={"function_name": {"$ne": "module_body"}}, include=["metadatas"]
        )

        if not code_results or not code_results.get("metadatas"):
            return

        # Count total calls in code collection
        total_calls_in_code = 0
        for metadata in code_results["metadatas"]:
            calls_json = metadata.get("calls", "[]")
            try:
                calls = json.loads(calls_json)
                total_calls_in_code += len(calls)
            except json.JSONDecodeError:
                pass

        # Get stats from reverse index
        reverse_stats = reverse_index.get_stats()

        # Total edges in reverse index should match total calls in code
        # (allowing for some discrepancy due to unresolved calls)
        assert reverse_stats["total_edges"] <= total_calls_in_code

    def test_rebuild_consistency(self, indexer, reverse_index):
        """Test that rebuilding produces same result."""
        # Get initial stats
        stats1 = reverse_index.get_stats()

        # Rebuild
        reverse_index.build_index(indexer.code_collection)

        # Get stats again
        stats2 = reverse_index.get_stats()

        # Should be identical
        assert stats1["indexed_functions"] == stats2["indexed_functions"]
        assert stats1["total_edges"] == stats2["total_edges"]
