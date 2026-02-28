"""Integration test for BSL Parser + Vector Indexer."""

import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Direct imports
import importlib.util

def load_module(module_name, file_path):
    """Load module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load BSL Parser
bsl_parser_path = Path(__file__).parent.parent / "src" / "parsers" / "bsl_parser.py"
bsl_parser_module = load_module("bsl_parser", bsl_parser_path)
BSLParser = bsl_parser_module.BSLParser


def test_bsl_parser_enriched_output():
    """Test that BSL Parser produces enriched output suitable for indexing."""
    
    content = """
// Получает данные клиента по коду
Функция ПолучитьДанныеКлиента(КодКлиента, ДатаНачала = Неопределено) Экспорт
    Запрос = Новый Запрос;
    Запрос.Текст = "ВЫБРАТЬ * ИЗ Справочник.Контрагенты";
    Результат = ОбщегоНазначения.ВыполнитьЗапрос(Запрос);
    Возврат Результат;
КонецФункции

// Сохраняет данные клиента
Процедура СохранитьДанныеКлиента(Клиент)
    Клиент.Записать();
КонецПроцедуры
"""
    
    parser = BSLParser()
    
    # Create temp file in src/Catalogs structure
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create directory structure
        catalog_path = tmpdir_path / "src" / "Catalogs" / "Контрагенты" / "Ext"
        catalog_path.mkdir(parents=True)
        
        test_file = catalog_path / "ObjectModule.bsl"
        test_file.write_text(content, encoding='utf-8')
        
        # Parse file
        chunks = parser.parse_file_with_ast(test_file)
        
        # Verify chunks
        assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}"
        
        # Check first function chunk
        func_chunk = chunks[0]
        assert func_chunk['module_type'] == 'CatalogObjectModule'
        assert func_chunk['object_name'] == 'Справочники.Контрагенты'
        assert func_chunk['function_name'] == 'ПолучитьДанныеКлиента'
        assert func_chunk['function_type'] == 'Функция'
        assert func_chunk['is_export'] == True
        assert len(func_chunk['params']) == 2
        assert 'КодКлиента' in func_chunk['params']
        assert 'ДатаНачала' in func_chunk['params']
        assert 'Получает данные клиента' in func_chunk['comments']
        assert len(func_chunk['calls']) > 0
        assert 'ОбщегоНазначения.ВыполнитьЗапрос' in func_chunk['calls']
        
        # Check second function chunk
        proc_chunk = chunks[1]
        assert proc_chunk['function_name'] == 'СохранитьДанныеКлиента'
        assert proc_chunk['function_type'] == 'Процедура'
        assert proc_chunk['is_export'] == False
        
        print("[OK] test_bsl_parser_enriched_output passed")
        print(f"  Chunks: {len(chunks)}")
        print(f"  First chunk metadata:")
        print(f"    Module: {func_chunk['module_type']}")
        print(f"    Object: {func_chunk['object_name']}")
        print(f"    Function: {func_chunk['function_name']}")
        print(f"    Export: {func_chunk['is_export']}")
        print(f"    Params: {func_chunk['params']}")
        print(f"    Calls: {func_chunk['calls']}")
        print(f"    Comments: {func_chunk['comments'][:50]}...")


def test_enriched_text_format():
    """Test that enriched text is properly formatted for embeddings."""
    
    content = """
// Проверяет доступ пользователя
Функция ПроверитьДоступ(Пользователь) Экспорт
    Возврат Истина;
КонецФункции
"""
    
    parser = BSLParser()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create directory structure
        module_path = tmpdir_path / "src" / "CommonModules" / "ОбщегоНазначения" / "Ext"
        module_path.mkdir(parents=True)
        
        test_file = module_path / "Module.bsl"
        test_file.write_text(content, encoding='utf-8')
        
        # Parse file
        chunks = parser.parse_file_with_ast(test_file)
        
        assert len(chunks) >= 1
        chunk = chunks[0]
        
        # Build enriched text (как в indexer)
        import json
        enriched_parts = []
        enriched_parts.append(f"Модуль: {chunk['object_name']} ({chunk['module_type']})")
        
        if chunk['function_name']:
            params_str = ', '.join(chunk['params']) if chunk['params'] else ''
            export_str = ' Экспорт' if chunk['is_export'] else ''
            enriched_parts.append(
                f"{chunk['function_type']}: {chunk['function_name']}({params_str}){export_str}"
            )
        
        if chunk['comments']:
            enriched_parts.append(f"// {chunk['comments']}")
        
        enriched_parts.append('')
        enriched_parts.append(chunk['code'])
        
        enriched_text = '\n'.join(enriched_parts)
        
        # Verify enriched text structure
        assert 'Модуль: ОбщиеМодули.ОбщегоНазначения' in enriched_text
        assert 'Функция: ПроверитьДоступ(Пользователь) Экспорт' in enriched_text
        assert '// Проверяет доступ пользователя' in enriched_text
        assert 'Возврат Истина' in enriched_text
        
        # Verify metadata can be serialized to JSON
        metadata = {
            "chunk_id": chunk['chunk_id'],
            "module_type": chunk['module_type'],
            "object_name": chunk['object_name'],
            "function_name": chunk['function_name'],
            "function_type": chunk['function_type'],
            "is_export": chunk['is_export'],
            "params": json.dumps(chunk['params'], ensure_ascii=False),
            "calls": json.dumps(chunk['calls'], ensure_ascii=False),
        }
        
        # Should not raise
        json_str = json.dumps(metadata, ensure_ascii=False)
        assert len(json_str) > 0
        
        print("[OK] test_enriched_text_format passed")
        print(f"  Enriched text length: {len(enriched_text)} chars")
        print(f"  Enriched text preview:")
        for line in enriched_text.split('\n')[:5]:
            print(f"    {line}")


def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("Running Integration Tests (BSL Parser + Indexer)")
    print("=" * 60)
    print()
    
    tests = [
        test_bsl_parser_enriched_output,
        test_enriched_text_format,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
            print()
        except Exception as e:
            print(f"[FAIL] {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
