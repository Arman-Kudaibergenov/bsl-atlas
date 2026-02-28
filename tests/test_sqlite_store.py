"""Smoke tests for SQLiteStore structural layer.

Tests the key methods of SQLiteStore against real BSL content
parsed by CodeParser. No ChromaDB or network calls needed.
"""

import sys
import tempfile
from pathlib import Path

import pytest

# Add project root so `src` is importable as a package (preserves relative imports)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.sqlite_store import SQLiteStore
from src.storage.models import Attribute, MetadataObject, TabPart

# ---------------------------------------------------------------------------
# Shared BSL fixtures
# ---------------------------------------------------------------------------

BSL_MODULE_A = """\
Процедура ПровестиДокумент(Отказ, РежимПроведения) Экспорт
\tСтавка = ПолучитьСтавку(Дата);
\tСформироватьДвижения(Ставка);
КонецПроцедуры

Функция ПолучитьСтавку(Дата) Экспорт
\tЗапрос = Новый Запрос;
\tЗапрос.Текст = "ВЫБРАТЬ Ставка ИЗ РегистрСведений.Ставки";
\tРезультат = Запрос.Выполнить();
\tВозврат Результат;
КонецФункции

Процедура СформироватьДвижения(Ставка)
\tДвижения.ВзаиморасчетыПоДоговорам.Записать();
КонецПроцедуры
"""

BSL_MODULE_B = """\
Функция РассчитатьСумму(Количество, Цена) Экспорт
\tВозврат Количество * Цена;
КонецФункции

Процедура ПроверитьЗаполнение(Отказ) Экспорт
\tСумма = РассчитатьСумму(1, 100);
\tЕсли Сумма = 0 Тогда
\t\tОтказ = Истина;
\tКонецЕсли;
КонецПроцедуры
"""


@pytest.fixture
def bsl_files(tmp_path):
    """Create temp BSL files in a minimal 1C directory structure."""
    doc_dir = tmp_path / "Documents" / "ЛизинговыйДоговор" / "Ext"
    doc_dir.mkdir(parents=True)
    module_a = doc_dir / "ObjectModule.bsl"
    module_a.write_text(BSL_MODULE_A, encoding="utf-8")

    common_dir = tmp_path / "CommonModules" / "РасчетСумм" / "Ext"
    common_dir.mkdir(parents=True)
    module_b = common_dir / "Module.bsl"
    module_b.write_text(BSL_MODULE_B, encoding="utf-8")

    return [module_a, module_b]


@pytest.fixture
def metadata_objects():
    """Sample MetadataObject list."""
    return [
        MetadataObject(
            name="ЛизинговыйДоговор",
            object_type="Документ",
            synonym="Лизинговый договор",
            attributes=[
                Attribute("Контрагент", "СправочникСсылка.Контрагенты", is_required=True),
                Attribute("Сумма", "Число"),
                Attribute("Дата", "Дата"),
            ],
            tab_parts=[
                TabPart(
                    name="ГрафикПлатежей",
                    attributes=[
                        Attribute("ДатаПлатежа", "Дата"),
                        Attribute("Сумма", "Число"),
                    ],
                )
            ],
            registers=["РегистрНакопления.ВзаиморасчетыПоДоговорам"],
        ),
        MetadataObject(
            name="Контрагенты",
            object_type="Справочник",
            synonym="Контрагент",
            attributes=[
                Attribute("ИНН", "Строка"),
                Attribute("КПП", "Строка"),
            ],
        ),
    ]


@pytest.fixture
def store(tmp_path, bsl_files, metadata_objects):
    """Built SQLiteStore with test data."""
    db_path = tmp_path / "test_index.db"
    s = SQLiteStore(db_path=db_path)
    s.rebuild(bsl_files, metadata_objects)
    return s


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_has_data_after_rebuild(self, store):
        assert store.has_data()

    def test_stats_counts(self, store):
        s = store.stats()
        assert s.files == 2
        assert s.symbols >= 5       # 3 in A + 2 in B
        assert s.objects == 2
        assert s.attributes >= 5   # 3 + 2

    def test_empty_store_has_no_data(self, tmp_path):
        s = SQLiteStore(db_path=tmp_path / "empty.db")
        assert not s.has_data()


# ---------------------------------------------------------------------------
# find_function
# ---------------------------------------------------------------------------


class TestFindFunction:
    def test_exact_match(self, store):
        results = store.find_function("ПолучитьСтавку", exact=True)
        assert len(results) == 1
        fn = results[0]
        assert fn.name == "ПолучитьСтавку"
        assert fn.type == "Функция"
        assert fn.is_export

    def test_exact_match_case_insensitive(self, store):
        # NOCASE collation
        results = store.find_function("получитьставку", exact=True)
        assert len(results) == 1

    def test_exact_no_match(self, store):
        results = store.find_function("НесуществующаяФункция", exact=True)
        assert results == []

    def test_fuzzy_match(self, store):
        # FTS5 prefix / fuzzy
        results = store.find_function("Провести", exact=False)
        assert any("Провести" in r.name for r in results)

    def test_procedure_found(self, store):
        results = store.find_function("ПровестиДокумент")
        assert len(results) == 1
        assert results[0].type == "Процедура"
        assert results[0].is_export

    def test_private_function_found(self, store):
        results = store.find_function("СформироватьДвижения")
        assert len(results) == 1
        assert not results[0].is_export

    def test_module_path_populated(self, store):
        results = store.find_function("РассчитатьСумму")
        assert len(results) == 1
        assert results[0].module_path  # non-empty
        assert results[0].module_type  # non-empty


# ---------------------------------------------------------------------------
# get_module_functions
# ---------------------------------------------------------------------------


class TestGetModuleFunctions:
    def test_returns_all_functions(self, store):
        results = store.get_module_functions("ObjectModule")
        assert len(results) == 3

    def test_ordered_by_line(self, store):
        results = store.get_module_functions("ObjectModule")
        lines = [r.line_start for r in results]
        assert lines == sorted(lines)

    def test_partial_path_match(self, store):
        results = store.get_module_functions("РасчетСумм")
        assert len(results) == 2

    def test_no_match_returns_empty(self, store):
        results = store.get_module_functions("НесуществующийМодуль")
        assert results == []


# ---------------------------------------------------------------------------
# get_function_context
# ---------------------------------------------------------------------------


class TestGetFunctionContext:
    def test_calls_extracted(self, store):
        ctx = store.get_function_context("ПровестиДокумент")
        assert ctx is not None
        assert "ПолучитьСтавку" in ctx.calls
        assert "СформироватьДвижения" in ctx.calls

    def test_called_by(self, store):
        # ПолучитьСтавку is called by ПровестиДокумент
        ctx = store.get_function_context("ПолучитьСтавку")
        assert ctx is not None
        assert "ПровестиДокумент" in ctx.called_by

    def test_not_found_returns_none(self, store):
        ctx = store.get_function_context("НесуществующаяФункция")
        assert ctx is None

    def test_context_contains_function_info(self, store):
        ctx = store.get_function_context("ПровестиДокумент")
        assert ctx.function.name == "ПровестиДокумент"
        assert ctx.function.is_export

    def test_private_has_caller(self, store):
        ctx = store.get_function_context("СформироватьДвижения")
        assert ctx is not None
        assert "ПровестиДокумент" in ctx.called_by


# ---------------------------------------------------------------------------
# search_metadata
# ---------------------------------------------------------------------------


class TestSearchMetadata:
    def test_find_by_name(self, store):
        results = store.search_metadata("ЛизинговыйДоговор")
        assert len(results) >= 1
        assert any(r.name == "ЛизинговыйДоговор" for r in results)

    def test_find_by_synonym(self, store):
        results = store.search_metadata("Лизинговый договор")
        assert len(results) >= 1

    def test_find_catalog(self, store):
        results = store.search_metadata("Контрагент")
        assert any(r.object_type == "Справочник" for r in results)

    def test_no_match_returns_empty(self, store):
        results = store.search_metadata("НесуществующийОбъект12345")
        assert results == []

    def test_result_has_full_name(self, store):
        results = store.search_metadata("Контрагент")
        for r in results:
            assert "." in r.full_name  # e.g. "Справочник.Контрагенты"


# ---------------------------------------------------------------------------
# get_object_attributes
# ---------------------------------------------------------------------------


class TestGetObjectAttributes:
    def test_find_by_full_name(self, store):
        details = store.get_object_attributes("Документ.ЛизинговыйДоговор")
        assert details is not None
        assert details.name == "ЛизинговыйДоговор"

    def test_find_by_name_only(self, store):
        details = store.get_object_attributes("Контрагенты")
        assert details is not None

    def test_attributes_returned(self, store):
        details = store.get_object_attributes("Документ.ЛизинговыйДоговор")
        attr_names = {a.name for a in details.attributes}
        assert "Контрагент" in attr_names
        assert "Сумма" in attr_names

    def test_attribute_type_ref(self, store):
        details = store.get_object_attributes("Документ.ЛизинговыйДоговор")
        kontragent = next(a for a in details.attributes if a.name == "Контрагент")
        assert kontragent.type_ref == "СправочникСсылка.Контрагенты"
        assert kontragent.is_required

    def test_tab_parts_returned(self, store):
        details = store.get_object_attributes("Документ.ЛизинговыйДоговор")
        assert len(details.tab_parts) == 1
        tp = details.tab_parts[0]
        assert tp.name == "ГрафикПлатежей"
        assert len(tp.attributes) == 2

    def test_registers_returned(self, store):
        details = store.get_object_attributes("Документ.ЛизинговыйДоговор")
        assert len(details.registers) == 1
        assert "ВзаиморасчетыПоДоговорам" in details.registers[0]

    def test_not_found_returns_none(self, store):
        details = store.get_object_attributes("НесуществующийОбъект.Фиктивный")
        assert details is None


# ---------------------------------------------------------------------------
# find_references_to
# ---------------------------------------------------------------------------


class TestFindReferences:
    def test_find_refs_to_catalog(self, store):
        refs = store.find_references_to("Справочник.Контрагенты")
        assert len(refs) >= 1
        ref = refs[0]
        assert ref.referencing_object  # not empty
        assert ref.attribute_name     # not empty
        assert "Контрагент" in ref.attribute_type

    def test_no_refs_returns_empty(self, store):
        refs = store.find_references_to("Справочник.НесуществующийОбъект")
        assert refs == []


# ---------------------------------------------------------------------------
# rebuild / update
# ---------------------------------------------------------------------------


class TestRebuild:
    def test_rebuild_clears_old_data(self, tmp_path, bsl_files, metadata_objects):
        db_path = tmp_path / "rebuild_test.db"
        s = SQLiteStore(db_path=db_path)
        s.rebuild(bsl_files, metadata_objects)
        s1 = s.stats()

        # Rebuild again — should produce same counts
        s.rebuild(bsl_files, metadata_objects)
        s2 = s.stats()

        assert s1.symbols == s2.symbols
        assert s1.objects == s2.objects

    def test_rebuild_returns_stats(self, tmp_path, bsl_files, metadata_objects):
        s = SQLiteStore(db_path=tmp_path / "s.db")
        stats = s.rebuild(bsl_files, metadata_objects)
        assert stats.files == 2
        assert stats.symbols >= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
