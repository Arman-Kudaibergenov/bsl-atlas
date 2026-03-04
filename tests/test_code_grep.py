"""Tests for code_grep — BSL grep with AST context."""
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.search.code_grep import CodeGrep, _find_function_regex

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BSL_A = """\
Процедура ПровестиДокумент(Отказ, РежимПроведения) Экспорт
\tСтавка = ПолучитьСтавку(Дата);
\tСформироватьДвижения(Ставка);
КонецПроцедуры

Функция ПолучитьСтавку(Дата) Экспорт
\tЗапрос = Новый Запрос;
\tЗапрос.Текст = "ВЫБРАТЬ СтавкиНДС ИЗ РегистрСведений.Ставки";
\tРезультат = Запрос.Выполнить();
\tВозврат Результат;
КонецФункции

Процедура СформироватьДвижения(Ставка)
\tДвижения.ВзаиморасчетыПоДоговорам.Записать();
КонецПроцедуры
"""

BSL_B = """\
Функция РассчитатьСумму(Количество, Цена) Экспорт
\tВозврат Количество * Цена;
КонецФункции

// module-level call СтавкиНДС here
СтавкиНДС = 0.2;

Процедура ПроверитьЗаполнение(Отказ) Экспорт
\tЕсли СтавкиНДС = 0 Тогда
\t\tОтказ = Истина;
\tКонецЕсли;
КонецПроцедуры
"""


@pytest.fixture
def source_dir(tmp_path: Path) -> Path:
    """Create temp dir with two .bsl files."""
    (tmp_path / "CommonModules" / "ОбщийМодуль" / "Ext").mkdir(parents=True)
    f_a = tmp_path / "CommonModules" / "ОбщийМодуль" / "Ext" / "Module.bsl"
    f_a.write_text(BSL_A, encoding="utf-8")

    (tmp_path / "Catalogs" / "Контрагенты" / "Ext").mkdir(parents=True)
    f_b = tmp_path / "Catalogs" / "Контрагенты" / "Ext" / "ObjectModule.bsl"
    f_b.write_text(BSL_B, encoding="utf-8")

    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_grep_finds_matches(source_dir: Path):
    results = CodeGrep().search("СтавкиНДС", source_dir)
    assert len(results) >= 2, "Expected matches in both files"


def test_grep_result_fields(source_dir: Path):
    results = CodeGrep().search("СтавкиНДС", source_dir)
    for r in results:
        assert "file" in r
        assert "line" in r
        assert "text" in r
        assert "function" in r
        assert "module_type" in r
        assert "context" in r


def test_grep_function_context(source_dir: Path):
    results = CodeGrep().search("ВЫБРАТЬ СтавкиНДС", source_dir, case_sensitive=False)
    assert results, "Expected at least one match"
    r = results[0]
    assert r["function"] == "ПолучитьСтавку"


def test_grep_module_level(source_dir: Path):
    """Match at module level (outside any function) returns 'module level'."""
    results = CodeGrep().search("// module-level call", source_dir, case_sensitive=True)
    assert results
    r = results[0]
    assert r["function"] == "module level"


def test_grep_module_type(source_dir: Path):
    results = CodeGrep().search("РассчитатьСумму", source_dir)
    assert results
    by_file = {r["file"]: r for r in results}
    obj_match = next((r for r in results if "ObjectModule.bsl" in r["file"]), None)
    assert obj_match is not None
    assert obj_match["module_type"] == "ObjectModule"


def test_grep_case_insensitive(source_dir: Path):
    lower = CodeGrep().search("ставкиндс", source_dir, case_sensitive=False)
    upper = CodeGrep().search("СТАВКИНДС", source_dir, case_sensitive=False)
    assert len(lower) == len(upper)


def test_grep_case_sensitive_no_match(source_dir: Path):
    results = CodeGrep().search("СТАВКИНДС", source_dir, case_sensitive=True)
    # Cyrillic upper СТАВКИНДС shouldn't match lower-cased content in BSL_B
    # (the variable is written as СтавкиНДС, not all-caps)
    assert all("СТАВКИНДС" in r["text"] for r in results)


def test_grep_limit(source_dir: Path):
    results = CodeGrep().search("КонецПроцедуры", source_dir, limit=1)
    assert len(results) <= 1


def test_grep_no_pattern_returns_empty(source_dir: Path):
    results = CodeGrep().search("", source_dir)
    assert results == []


def test_grep_missing_source():
    results = CodeGrep().search("anything", Path("/nonexistent/path/xyz"))
    assert results == []


def test_grep_context_lines(source_dir: Path):
    results = CodeGrep().search("ВЫБРАТЬ СтавкиНДС", source_dir, case_sensitive=False)
    assert results
    ctx = results[0]["context"]
    # context should have multiple lines
    assert "\n" in ctx


# ---------------------------------------------------------------------------
# Regex fallback unit tests
# ---------------------------------------------------------------------------


def test_regex_finds_function():
    lines = BSL_A.splitlines()
    # Line 8 is inside ПолучитьСтавку (1-based)
    assert _find_function_regex(lines, 8) == "ПолучитьСтавку"


def test_regex_module_level():
    lines = BSL_B.splitlines()
    # Line 6: "СтавкиНДС = 0.2;" — outside any function
    assert _find_function_regex(lines, 6) == "module level"
