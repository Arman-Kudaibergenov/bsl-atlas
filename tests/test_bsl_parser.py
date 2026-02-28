"""Tests for BSL Parser with AST-based chunking."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Direct import to avoid __init__.py issues
import importlib.util
spec = importlib.util.spec_from_file_location(
    "bsl_parser",
    Path(__file__).parent.parent / "src" / "parsers" / "bsl_parser.py"
)
bsl_parser_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bsl_parser_module)
BSLParser = bsl_parser_module.BSLParser


def test_simple_function():
    """Test parsing a simple function."""
    content = """
// Получает данные клиента
Функция ПолучитьДанныеКлиента(КодКлиента) Экспорт
    Запрос = Новый Запрос;
    Запрос.Текст = "ВЫБРАТЬ * ИЗ Справочник.Контрагенты";
    Результат = Запрос.Выполнить();
    Возврат Результат;
КонецФункции
"""
    
    parser = BSLParser()
    
    # Create temp file
    temp_file = Path("temp_test.bsl")
    temp_file.write_text(content, encoding='utf-8')
    
    try:
        chunks = parser.parse_file_with_ast(temp_file)
        
        # Should have 1 function (module_body may or may not be present)
        func_chunks = [c for c in chunks if c['function_name'] is not None]
        assert len(func_chunks) == 1, f"Expected 1 function chunk, got {len(func_chunks)}"
        
        chunk = func_chunks[0]
        assert chunk['function_name'] == 'ПолучитьДанныеКлиента'
        assert chunk['function_type'] == 'Функция'
        assert chunk['is_export'] == True
        assert chunk['params'] == ['КодКлиента']
        assert 'Получает данные клиента' in chunk['comments']
        assert len(chunk['calls']) > 0  # Should detect Новый, Запрос.Выполнить
        
        print("[OK] test_simple_function passed")
        print(f"  Function: {chunk['function_name']}")
        print(f"  Type: {chunk['function_type']}")
        print(f"  Export: {chunk['is_export']}")
        print(f"  Params: {chunk['params']}")
        print(f"  Comments: {chunk['comments']}")
        print(f"  Calls: {chunk['calls']}")
        
    finally:
        temp_file.unlink(missing_ok=True)


def test_multiple_functions():
    """Test parsing multiple functions."""
    content = """
// Модуль для работы с клиентами
Перем КэшДанных;

// Получает клиента по коду
Функция ПолучитьКлиента(Код) Экспорт
    Возврат НайтиВСправочнике(Код);
КонецФункции

// Сохраняет клиента
Процедура СохранитьКлиента(Клиент)
    Клиент.Записать();
КонецПроцедуры

// Внутренняя функция поиска
Функция НайтиВСправочнике(Код)
    Запрос = Новый Запрос;
    Возврат Запрос.Выполнить();
КонецФункции
"""
    
    parser = BSLParser()
    
    temp_file = Path("temp_test.bsl")
    temp_file.write_text(content, encoding='utf-8')
    
    try:
        chunks = parser.parse_file_with_ast(temp_file)
        
        # Should have 3 functions + 1 module body
        assert len(chunks) == 4, f"Expected 4 chunks, got {len(chunks)}"
        
        # Check functions
        func_names = [c['function_name'] for c in chunks if c['function_name']]
        assert 'ПолучитьКлиента' in func_names
        assert 'СохранитьКлиента' in func_names
        assert 'НайтиВСправочнике' in func_names
        
        # Check module body
        module_body_chunks = [c for c in chunks if c['function_name'] is None]
        assert len(module_body_chunks) == 1
        assert 'КэшДанных' in module_body_chunks[0]['code']
        
        # Check export flags
        export_funcs = [c for c in chunks if c.get('is_export')]
        assert len(export_funcs) == 1
        assert export_funcs[0]['function_name'] == 'ПолучитьКлиента'
        
        print("[OK] test_multiple_functions passed")
        print(f"  Functions: {func_names}")
        print(f"  Module body present: Yes")
        
    finally:
        temp_file.unlink(missing_ok=True)


def test_function_calls_extraction():
    """Test extraction of function calls."""
    content = """
Функция ОбработатьДанные(Данные) Экспорт
    Результат = ОбщегоНазначения.ПроверитьДоступ(Данные);
    Если Результат Тогда
        СохранитьВБазу(Данные);
        ОтправитьУведомление();
    КонецЕсли;
    Возврат Результат;
КонецФункции
"""
    
    parser = BSLParser()
    
    temp_file = Path("temp_test.bsl")
    temp_file.write_text(content, encoding='utf-8')
    
    try:
        chunks = parser.parse_file_with_ast(temp_file)
        
        assert len(chunks) == 1
        chunk = chunks[0]
        
        # Check calls
        calls = chunk['calls']
        assert 'ОбщегоНазначения.ПроверитьДоступ' in calls
        assert 'СохранитьВБазу' in calls
        assert 'ОтправитьУведомление' in calls
        
        # Should NOT include keywords
        assert 'Если' not in calls
        assert 'Возврат' not in calls
        
        print("[OK] test_function_calls_extraction passed")
        print(f"  Calls: {calls}")
        
    finally:
        temp_file.unlink(missing_ok=True)


def test_module_type_detection():
    """Test module type detection from file path."""
    parser = BSLParser()
    
    test_cases = [
        (Path("src/CommonModules/ОбщегоНазначения/Ext/Module.bsl"), "CommonModule"),
        (Path("src/Catalogs/Контрагенты/Ext/ObjectModule.bsl"), "CatalogObjectModule"),
        (Path("src/Documents/ЗаказКлиента/Ext/ObjectModule.bsl"), "DocumentObjectModule"),
        (Path("src/DataProcessors/Обработка1/Ext/ObjectModule.bsl"), "DataProcessorModule"),
        (Path("src/Catalogs/Контрагенты/Ext/ManagerModule.bsl"), "ManagerModule"),
    ]
    
    for file_path, expected_type in test_cases:
        module_type = parser._determine_module_type(file_path)
        assert module_type == expected_type, f"Expected {expected_type}, got {module_type} for {file_path}"
    
    print("[OK] test_module_type_detection passed")
    for file_path, expected_type in test_cases:
        print(f"  {file_path.name} -> {expected_type}")


def test_object_name_extraction():
    """Test object name extraction from file path."""
    parser = BSLParser()
    
    test_cases = [
        (Path("src/Catalogs/Контрагенты/Ext/ObjectModule.bsl"), "Справочники.Контрагенты"),
        (Path("src/Documents/ЗаказКлиента/Ext/ObjectModule.bsl"), "Документы.ЗаказКлиента"),
        (Path("src/CommonModules/ОбщегоНазначения/Ext/Module.bsl"), "ОбщиеМодули.ОбщегоНазначения"),
        (Path("src/DataProcessors/Обработка1/Ext/ObjectModule.bsl"), "Обработки.Обработка1"),
    ]
    
    for file_path, expected_name in test_cases:
        object_name = parser._extract_object_name(file_path)
        assert object_name == expected_name, f"Expected {expected_name}, got {object_name} for {file_path}"
    
    print("[OK] test_object_name_extraction passed")
    for file_path, expected_name in test_cases:
        print(f"  {file_path} -> {expected_name}")


def test_real_file():
    """Test parsing a real BSL file from MinimKG."""
    parser = BSLParser()
    
    # Find a real BSL file
    minimkg_path = Path(r"C:\Users\Arman\workspace\MinimKG\src\cf")
    if not minimkg_path.exists():
        print("[SKIP] test_real_file skipped (MinimKG not found)")
        return
    
    # Find first BSL file
    bsl_files = list(minimkg_path.rglob("*.bsl"))
    if not bsl_files:
        print("[SKIP] test_real_file skipped (no BSL files found)")
        return
    
    test_file = bsl_files[0]
    print(f"Testing with real file: {test_file}")
    
    try:
        chunks = parser.parse_file_with_ast(test_file)
        
        print(f"[OK] test_real_file passed")
        print(f"  File: {test_file.name}")
        print(f"  Chunks: {len(chunks)}")
        
        if chunks:
            # Show first chunk details
            chunk = chunks[0]
            print(f"  First chunk:")
            print(f"    Module type: {chunk['module_type']}")
            print(f"    Object name: {chunk['object_name']}")
            print(f"    Function: {chunk['function_name']}")
            print(f"    Export: {chunk['is_export']}")
            print(f"    Params: {chunk['params']}")
            print(f"    Calls: {len(chunk['calls'])} calls")
            print(f"    Lines: {chunk['line_start']}-{chunk['line_end']}")
        
    except Exception as e:
        print(f"[FAIL] test_real_file failed: {e}")
        raise


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running BSL Parser Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_simple_function,
        test_multiple_functions,
        test_function_calls_extraction,
        test_module_type_detection,
        test_object_name_extraction,
        test_real_file,
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
            failed += 1
            print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
