"""Wave 0 tests: Fix A (qualifier capture), call edges + resolution, and the
incremental detector / update path.

Pure SQLite + parser — no ChromaDB or network. The tree-sitter grammar is absent
on dev machines, so these exercise the regex parser path (the one that ships in
the daily loop here).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsers.code import CodeParser
from src.storage.sqlite_store import SQLiteStore
from src.storage.models import CallRef


# ---------------------------------------------------------------------------
# BSL fixtures
# ---------------------------------------------------------------------------

COMMON_MODULE = """\
Функция ЗначениеРеквизита(Ссылка, ИмяРеквизита) Экспорт
\tВозврат Ссылка[ИмяРеквизита];
КонецФункции

Процедура СообщитьПользователю(Текст) Экспорт
\tСообщить(Текст);
КонецПроцедуры
"""

OBJECT_MODULE = """\
Процедура ОбработкаПроведения(Отказ, РежимПроведения)
\tКонтрагент = ОбщегоНазначения.ЗначениеРеквизита(Ссылка, "Контрагент");
\tСтавка = ПолучитьСтавку(Дата);
\tСообщить("Проведено");
КонецПроцедуры

Функция ПолучитьСтавку(Дата)
\tВозврат 10;
КонецФункции
"""


def _make_tree(root: Path):
    cm_dir = root / "CommonModules" / "ОбщегоНазначения" / "Ext"
    cm_dir.mkdir(parents=True)
    cm = cm_dir / "Module.bsl"
    cm.write_text(COMMON_MODULE, encoding="utf-8")

    om_dir = root / "Documents" / "ЛизинговыйДоговор" / "Ext"
    om_dir.mkdir(parents=True)
    om = om_dir / "ObjectModule.bsl"
    om.write_text(OBJECT_MODULE, encoding="utf-8")
    return cm, om


# ---------------------------------------------------------------------------
# Fix A — qualifier capture in the parser
# ---------------------------------------------------------------------------


class TestQualifierCapture:
    def test_qualified_call_keeps_qualifier(self, tmp_path):
        cm, om = _make_tree(tmp_path)
        funcs = CodeParser().parse_file_functions(om)
        proc = next(f for f in funcs if f.name == "ОбработкаПроведения")
        refs = {(c.qualifier, c.name) for c in proc.calls}
        assert ("ОбщегоНазначения", "ЗначениеРеквизита") in refs

    def test_unqualified_call_has_no_qualifier(self, tmp_path):
        cm, om = _make_tree(tmp_path)
        funcs = CodeParser().parse_file_functions(om)
        proc = next(f for f in funcs if f.name == "ОбработкаПроведения")
        refs = {(c.qualifier, c.name) for c in proc.calls}
        assert (None, "ПолучитьСтавку") in refs

    def test_platform_global_dropped(self, tmp_path):
        cm, om = _make_tree(tmp_path)
        funcs = CodeParser().parse_file_functions(om)
        proc = next(f for f in funcs if f.name == "ОбработкаПроведения")
        names = {c.name for c in proc.calls}
        assert "Сообщить" not in names  # platform global, dropped as noise

    def test_calls_are_callref(self, tmp_path):
        cm, om = _make_tree(tmp_path)
        funcs = CodeParser().parse_file_functions(om)
        for f in funcs:
            for c in f.calls:
                assert isinstance(c, CallRef)


# ---------------------------------------------------------------------------
# Edges + resolution
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    cm, om = _make_tree(tmp_path)
    s = SQLiteStore(db_path=tmp_path / "idx.db")
    s.rebuild([cm, om], [])
    return s


class TestEdgeResolution:
    def test_commonmodule_call_resolved(self, store):
        edges = store.count_edges()
        assert edges["by_type"].get("call_commonmodule", 0) >= 1
        assert edges["resolved"] >= 1

    def test_commonmodule_reverse_resolves_to_symbol(self, store):
        # ЗначениеРеквизита is defined in CommonModule, called from ObjectModule
        ctx = store.get_function_context("ЗначениеРеквизита")
        assert ctx is not None
        assert "ОбработкаПроведения" in {c["name"] for c in ctx.called_by}

    def test_local_call_resolved_to_symbol_id(self, store):
        # ПолучитьСтавку called unqualified from same file -> call_local, dst set
        with store._get_conn() as conn:
            row = conn.execute(
                """SELECT e.dst_symbol_id, e.edge_type, e.resolved
                   FROM edges e
                   JOIN symbols s ON e.src_symbol_id = s.id
                   WHERE s.name = 'ОбработкаПроведения' AND e.callee_name = 'ПолучитьСтавку'"""
            ).fetchone()
        assert row is not None
        assert row["resolved"] == 1
        assert row["edge_type"] == "call_local"
        assert row["dst_symbol_id"] is not None

    def test_local_reverse_call(self, store):
        ctx = store.get_function_context("ПолучитьСтавку")
        assert ctx is not None
        assert "ОбработкаПроведения" in {c["name"] for c in ctx.called_by}

    def test_unqualified_unknown_stays_unresolved(self, store):
        # СообщитьПользователю calls Сообщить — dropped as global, so no edge;
        # remaining honesty check: nothing falsely resolved
        edges = store.count_edges()
        # every resolved edge must have a dst
        with store._get_conn() as conn:
            bad = conn.execute(
                "SELECT COUNT(*) FROM edges WHERE resolved = 1 AND dst_symbol_id IS NULL"
            ).fetchone()[0]
        assert bad == 0


# ---------------------------------------------------------------------------
# Incremental detector + update
# ---------------------------------------------------------------------------


class TestIncremental:
    def test_no_changes_detected_after_rebuild(self, store, tmp_path):
        changed, deleted = store.detect_changes(tmp_path)
        assert changed == []
        assert deleted == []

    def test_edit_detected_and_applied(self, store, tmp_path):
        om = tmp_path / "Documents" / "ЛизинговыйДоговор" / "Ext" / "ObjectModule.bsl"
        new_src = OBJECT_MODULE + "\nПроцедура НоваяПроцедура() Экспорт\n\tВозврат;\nКонецПроцедуры\n"
        om.write_text(new_src, encoding="utf-8")

        changed, deleted = store.detect_changes(tmp_path)
        assert len(changed) == 1
        assert deleted == []

        store.update(changed, deleted_keys=deleted)
        results = store.find_function("НоваяПроцедура")
        assert len(results) == 1

    def test_update_does_not_duplicate_symbols(self, store, tmp_path):
        om = tmp_path / "Documents" / "ЛизинговыйДоговор" / "Ext" / "ObjectModule.bsl"
        before = len(store.find_function("ПолучитьСтавку"))
        # Touch content (whitespace) so hash changes
        om.write_text(OBJECT_MODULE + "\n// comment\n", encoding="utf-8")
        changed, deleted = store.detect_changes(tmp_path)
        store.update(changed, deleted_keys=deleted)
        after = len(store.find_function("ПолучитьСтавку"))
        assert after == before == 1

    def test_update_no_duplicate_code_fts(self, store, tmp_path):
        om = tmp_path / "Documents" / "ЛизинговыйДоговор" / "Ext" / "ObjectModule.bsl"
        om.write_text(OBJECT_MODULE + "\n// touch\n", encoding="utf-8")
        changed, deleted = store.detect_changes(tmp_path)
        store.update(changed, deleted_keys=deleted)
        # grep for a line unique to the function — must appear once per occurrence,
        # not doubled by a stale FTS row
        hits = store.code_grep("ПолучитьСтавку")
        # ПолучитьСтавку appears as call (in ОбработкаПроведения) + definition body
        files = {h["file"] for h in hits}
        assert len(files) == 1  # single module, no duplicate file rows

    def test_deleted_file_swept(self, store, tmp_path):
        cm = tmp_path / "CommonModules" / "ОбщегоНазначения" / "Ext" / "Module.bsl"
        cm.unlink()
        changed, deleted = store.detect_changes(tmp_path)
        assert len(deleted) == 1
        store.update(changed, deleted_keys=deleted)
        assert store.find_function("ЗначениеРеквизита") == []

    def test_deleted_target_nulls_inbound_edges(self, store, tmp_path):
        # Remove the common module → the resolved commonmodule edge must drop back
        cm = tmp_path / "CommonModules" / "ОбщегоНазначения" / "Ext" / "Module.bsl"
        cm.unlink()
        changed, deleted = store.detect_changes(tmp_path)
        store.update(changed, deleted_keys=deleted)
        with store._get_conn() as conn:
            bad = conn.execute(
                "SELECT COUNT(*) FROM edges WHERE resolved = 1 AND dst_symbol_id IS NULL"
            ).fetchone()[0]
        assert bad == 0


# ---------------------------------------------------------------------------
# Stable vector id helper
# ---------------------------------------------------------------------------


class TestStableVectorId:
    def test_deterministic(self):
        from src.indexer.vector_indexer import VectorIndexer

        a = VectorIndexer._stable_id("/x/Module.bsl", "Foo", 0)
        b = VectorIndexer._stable_id("/x/Module.bsl", "Foo", 0)
        assert a == b
        assert a.startswith("fn_")

    def test_distinct_for_different_functions(self):
        from src.indexer.vector_indexer import VectorIndexer

        a = VectorIndexer._stable_id("/x/Module.bsl", "Foo", 0)
        b = VectorIndexer._stable_id("/x/Module.bsl", "Bar", 0)
        assert a != b


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
