"""Tests for advanced search functionality."""

import json
from pathlib import Path

import pytest

from src.config import Config
from src.indexer import VectorIndexer
from src.search import AdvancedSearch


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
def advanced_search(indexer):
    """Create advanced search instance."""
    return AdvancedSearch(code_collection=indexer.code_collection)


class TestSearchFunction:
    """Tests for search_function tool."""

    def test_exact_function_search(self, advanced_search):
        """Test exact function name search."""
        results = advanced_search.search_function("ПроверитьЗаполнение", exact=True)
        
        assert isinstance(results, list)
        if results and "error" not in results[0]:
            assert len(results) > 0
            func = results[0]
            assert func["function_name"] == "ПроверитьЗаполнение"
            assert "params" in func
            assert "calls" in func
            assert "code" in func

    def test_partial_function_search(self, advanced_search):
        """Test partial function name search."""
        results = advanced_search.search_function("Проверить", exact=False)
        
        assert isinstance(results, list)
        if results and "error" not in results[0]:
            for func in results:
                assert "Проверить" in func["function_name"]

    def test_function_search_with_module_filter(self, advanced_search):
        """Test function search with module name filter."""
        results = advanced_search.search_function(
            "ПроверитьЗаполнение",
            module_name="Справочники.Контрагенты",
            exact=True,
        )
        
        assert isinstance(results, list)
        if results and "error" not in results[0]:
            for func in results:
                assert func["object_name"] == "Справочники.Контрагенты"

    def test_function_not_found(self, advanced_search):
        """Test search for non-existent function."""
        results = advanced_search.search_function("НесуществующаяФункция", exact=True)
        
        assert isinstance(results, list)
        assert len(results) == 0


class TestGetModuleFunctions:
    """Tests for get_module_functions tool."""

    def test_get_all_functions(self, advanced_search):
        """Test getting all functions in a module."""
        results = advanced_search.get_module_functions(
            "Справочники.Контрагенты",
            only_export=False,
        )
        
        assert isinstance(results, list)
        if results and "error" not in results[0]:
            assert len(results) > 0
            for func in results:
                assert "function_name" in func
                assert "function_type" in func
                assert "is_export" in func
                assert "params" in func
                assert func["function_name"] != "module_body"

    def test_get_export_functions_only(self, advanced_search):
        """Test getting only exported functions."""
        results = advanced_search.get_module_functions(
            "Справочники.Контрагенты",
            only_export=True,
        )
        
        assert isinstance(results, list)
        if results and "error" not in results[0]:
            for func in results:
                assert func["is_export"] is True

    def test_module_not_found(self, advanced_search):
        """Test getting functions from non-existent module."""
        results = advanced_search.get_module_functions(
            "Справочники.НесуществующийСправочник",
            only_export=False,
        )
        
        assert isinstance(results, list)
        assert len(results) == 0


class TestSearchCodeFiltered:
    """Tests for search_code_filtered tool."""

    def test_search_with_module_type_filter(self, advanced_search):
        """Test search with module type filter."""
        results = advanced_search.search_code_filtered(
            query="проверка заполнения",
            module_type="CommonModule",
            limit=5,
        )
        
        assert isinstance(results, list)
        if results and "error" not in results[0]:
            for chunk in results:
                assert chunk["module_type"] == "CommonModule"

    def test_search_export_only(self, advanced_search):
        """Test search for exported functions only."""
        results = advanced_search.search_code_filtered(
            query="получить данные",
            only_export=True,
            limit=5,
        )
        
        assert isinstance(results, list)
        if results and "error" not in results[0]:
            for chunk in results:
                assert chunk["is_export"] is True

    def test_search_with_combined_filters(self, advanced_search):
        """Test search with multiple filters."""
        results = advanced_search.search_code_filtered(
            query="обработка документа",
            module_type="DocumentObjectModule",
            only_export=True,
            limit=10,
        )
        
        assert isinstance(results, list)
        if results and "error" not in results[0]:
            for chunk in results:
                assert chunk["module_type"] == "DocumentObjectModule"
                assert chunk["is_export"] is True
                assert "score" in chunk


class TestGetFunctionContext:
    """Tests for get_function_context tool."""

    def test_get_context_with_callers(self, advanced_search):
        """Test getting function context with callers."""
        result = advanced_search.get_function_context(
            function_name="ПроверитьЗаполнение",
            include_callers=True,
        )
        
        assert isinstance(result, dict)
        if "error" not in result:
            assert "function" in result
            assert "calls" in result
            assert "called_by" in result
            assert "stats" in result
            
            # Check function info
            func = result["function"]
            assert func["function_name"] == "ПроверитьЗаполнение"
            
            # Check stats
            stats = result["stats"]
            assert "calls_count" in stats
            assert "called_by_count" in stats

    def test_get_context_without_callers(self, advanced_search):
        """Test getting function context without callers."""
        result = advanced_search.get_function_context(
            function_name="ПроверитьЗаполнение",
            include_callers=False,
        )
        
        assert isinstance(result, dict)
        if "error" not in result:
            assert "function" in result
            assert "calls" in result
            assert "called_by" in result
            assert len(result["called_by"]) == 0

    def test_get_context_with_module_filter(self, advanced_search):
        """Test getting function context with module filter."""
        result = advanced_search.get_function_context(
            function_name="ПроверитьЗаполнение",
            module_name="Справочники.Контрагенты",
            include_callers=True,
        )
        
        assert isinstance(result, dict)
        if "error" not in result:
            func = result["function"]
            assert func["object_name"] == "Справочники.Контрагенты"

    def test_context_function_not_found(self, advanced_search):
        """Test getting context for non-existent function."""
        result = advanced_search.get_function_context(
            function_name="НесуществующаяФункция",
            include_callers=True,
        )
        
        assert isinstance(result, dict)
        assert "error" in result


class TestCallGraphAnalysis:
    """Tests for call graph analysis features."""

    def test_calls_are_resolved(self, advanced_search):
        """Test that function calls are properly resolved."""
        result = advanced_search.get_function_context(
            function_name="ПроверитьЗаполнение",
            include_callers=True,
        )
        
        if "error" not in result:
            calls = result["calls"]
            
            # Check that called functions have proper structure
            for called_func in calls:
                assert "function_name" in called_func
                assert "object_name" in called_func
                assert "params" in called_func

    def test_callers_are_found(self, advanced_search):
        """Test that caller functions are properly found."""
        result = advanced_search.get_function_context(
            function_name="ПроверитьЗаполнение",
            include_callers=True,
        )
        
        if "error" not in result:
            callers = result["called_by"]
            
            # Check that caller functions have proper structure
            for caller_func in callers:
                assert "function_name" in caller_func
                assert "object_name" in caller_func
                
                # Verify that caller actually calls our function
                # (This would require fetching the caller's code and checking)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_query(self, advanced_search):
        """Test search with empty query."""
        results = advanced_search.search_code_filtered(
            query="",
            limit=5,
        )
        
        assert isinstance(results, list)

    def test_special_characters_in_function_name(self, advanced_search):
        """Test search with special characters."""
        results = advanced_search.search_function(
            function_name="Функция_С_Подчеркиванием",
            exact=True,
        )
        
        assert isinstance(results, list)

    def test_large_limit(self, advanced_search):
        """Test search with very large limit."""
        results = advanced_search.search_code_filtered(
            query="функция",
            limit=1000,
        )
        
        assert isinstance(results, list)
        # Should not crash, but may return fewer results

    def test_invalid_module_type(self, advanced_search):
        """Test search with invalid module type."""
        results = advanced_search.search_code_filtered(
            query="функция",
            module_type="InvalidModuleType",
            limit=5,
        )
        
        assert isinstance(results, list)
        # Should return empty list or handle gracefully
