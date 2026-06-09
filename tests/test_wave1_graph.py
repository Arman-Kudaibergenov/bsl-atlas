"""Wave 1 tests: indirect edges (subscriptions / entry points / query refs),
PageRank repomap, context assembly, verify oracle, and deterministic cards.

Builds a small but realistic 1C export tree (BSL + metadata XML + an event
subscription + a scheduled job) and exercises the graph tools end to end.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsers.code import CodeParser
from src.parsers.metadata_xml import MetadataXMLParser
from src.storage.sqlite_store import SQLiteStore


# ---------------------------------------------------------------------------
# Fixtures: a tiny 1C export tree
# ---------------------------------------------------------------------------

COMMON_HUB = """\
Функция ЗначениеРеквизита(Ссылка, ИмяРеквизита) Экспорт
\tВозврат Ссылка[ИмяРеквизита];
КонецФункции
"""

COMMON_HANDLERS = """\
Процедура ПередЗаписьюРеализации(Источник, Отказ) Экспорт
\tЕсли Источник.Сумма = 0 Тогда
\t\tОтказ = Истина;
\tКонецЕсли;
КонецПроцедуры
"""

DOC_OBJECT_MODULE = """\
Процедура ОбработкаПроведения(Отказ, РежимПроведения)
\tСтавка = ОбщегоНазначения.ЗначениеРеквизита(Ссылка, "Ставка");
\tСумма = ПолучитьСумму();
\tЗапрос = Новый Запрос;
\tЗапрос.Текст = "ВЫБРАТЬ Цена ИЗ Справочник.Номенклатура КАК Номенклатура";
КонецПроцедуры

Функция ПолучитьСумму()
\tВозврат ОбщегоНазначения.ЗначениеРеквизита(Ссылка, "Сумма");
КонецФункции

Процедура ПередЗаписью(Отказ)
\tСообщить("Запись");
КонецПроцедуры
"""

DOC_XML = """<?xml version="1.0" encoding="UTF-8"?>
<MetaDataObject>
  <Document uuid="doc-1">
    <Properties>
      <Name>РеализацияТоваров</Name>
      <Synonym><value>Реализация товаров</value></Synonym>
      <RegisterRecords>
        <RegisterRecord>AccumulationRegisterRecord.Продажи</RegisterRecord>
      </RegisterRecords>
    </Properties>
    <ChildObjects>
      <Attribute>
        <Properties><Name>Сумма</Name><Type><Type>xs:decimal</Type></Type></Properties>
      </Attribute>
      <Attribute>
        <Properties><Name>Контрагент</Name><Type><Type>cfg:CatalogRef.Контрагенты</Type></Type></Properties>
      </Attribute>
    </ChildObjects>
  </Document>
</MetaDataObject>
"""

SUBSCRIPTION_XML = """<?xml version="1.0" encoding="UTF-8"?>
<MetaDataObject>
  <EventSubscription uuid="sub-1">
    <Properties>
      <Name>ПередЗаписьюРеализации</Name>
      <Source><Type>cfg:DocumentObject.РеализацияТоваров</Type></Source>
      <Event>BeforeWrite</Event>
      <Handler>ОбщийМодуль.ОбработчикиСобытий.ПередЗаписьюРеализации</Handler>
    </Properties>
  </EventSubscription>
</MetaDataObject>
"""

SCHEDULED_JOB_XML = """<?xml version="1.0" encoding="UTF-8"?>
<MetaDataObject>
  <ScheduledJob uuid="job-1">
    <Properties>
      <Name>ЗакрытиеМесяца</Name>
      <MethodName>ОбщийМодуль.ОбработчикиСобытий.ПередЗаписьюРеализации</MethodName>
    </Properties>
  </ScheduledJob>
</MetaDataObject>
"""


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def tree(tmp_path):
    bsl = [
        _write(tmp_path / "CommonModules" / "ОбщегоНазначения" / "Ext" / "Module.bsl", COMMON_HUB),
        _write(tmp_path / "CommonModules" / "ОбработчикиСобытий" / "Ext" / "Module.bsl", COMMON_HANDLERS),
        _write(tmp_path / "Documents" / "РеализацияТоваров" / "Ext" / "ObjectModule.bsl", DOC_OBJECT_MODULE),
    ]
    _write(tmp_path / "Documents" / "РеализацияТоваров.xml", DOC_XML)
    _write(tmp_path / "EventSubscriptions" / "ПередЗаписьюРеализации.xml", SUBSCRIPTION_XML)
    _write(tmp_path / "ScheduledJobs" / "ЗакрытиеМесяца.xml", SCHEDULED_JOB_XML)
    return tmp_path, bsl


@pytest.fixture
def store(tree):
    root, bsl = tree
    xml = MetadataXMLParser()
    metas = xml.parse_directory(root)
    subs = xml.parse_event_subscriptions(root)
    eps = xml.parse_entry_points(root)
    s = SQLiteStore(db_path=root / "idx.db")
    s.rebuild(bsl, metas, subs, eps)
    return s


# ---------------------------------------------------------------------------
# Query refs (parser)
# ---------------------------------------------------------------------------


class TestQueryRefs:
    def test_extracts_object_from_query(self, tree):
        _, bsl = tree
        om = next(f for f in bsl if f.name == "ObjectModule.bsl")
        funcs = CodeParser().parse_file_functions(om)
        proc = next(f for f in funcs if f.name == "ОбработкаПроведения")
        assert "Справочник.Номенклатура" in proc.query_refs


# ---------------------------------------------------------------------------
# Metadata XML: subscriptions + entry points
# ---------------------------------------------------------------------------


class TestIndirectParsing:
    def test_subscription_parsed(self, tree):
        root, _ = tree
        subs = MetadataXMLParser().parse_event_subscriptions(root)
        assert len(subs) == 1
        s = subs[0]
        assert s["event"] == "ПередЗаписью"          # BeforeWrite normalised
        assert s["handler_module"] == "ОбработчикиСобытий"
        assert s["handler_method"] == "ПередЗаписьюРеализации"
        assert "РеализацияТоваров" in s["source"]

    def test_entry_point_parsed(self, tree):
        root, _ = tree
        eps = MetadataXMLParser().parse_entry_points(root)
        assert len(eps) == 1
        assert eps[0]["kind"] == "scheduled_job"
        assert eps[0]["handler_method"] == "ПередЗаписьюРеализации"


# ---------------------------------------------------------------------------
# Indirect handler resolution
# ---------------------------------------------------------------------------


class TestHandlerResolution:
    def test_subscription_handler_resolved(self, store):
        with store._get_conn() as conn:
            row = conn.execute(
                "SELECT handler_symbol_id FROM subscriptions WHERE name = 'ПередЗаписьюРеализации'"
            ).fetchone()
        assert row["handler_symbol_id"] is not None

    def test_entry_point_handler_resolved(self, store):
        with store._get_conn() as conn:
            row = conn.execute("SELECT handler_symbol_id FROM entry_points LIMIT 1").fetchone()
        assert row["handler_symbol_id"] is not None


# ---------------------------------------------------------------------------
# PageRank / repomap
# ---------------------------------------------------------------------------


class TestRepomap:
    def test_hub_has_highest_centrality(self, store):
        with store._get_conn() as conn:
            hub = conn.execute(
                "SELECT centrality FROM symbols WHERE name = 'ЗначениеРеквизита'"
            ).fetchone()["centrality"]
            leaf = conn.execute(
                "SELECT centrality FROM symbols WHERE name = 'ПередЗаписью'"
            ).fetchone()["centrality"]
        assert hub > leaf

    def test_repomap_ranks_hub_first(self, store):
        rows = store.repomap(token_budget=2000)
        assert rows
        assert "ЗначениеРеквизита" in rows[0]["signature"]

    def test_repomap_scope_filter(self, store):
        rows = store.repomap(scope="ОбщегоНазначения")
        assert all(r["owner"] == "ОбщегоНазначения" for r in rows)
        assert any("ЗначениеРеквизита" in r["signature"] for r in rows)

    def test_repomap_budget_bounds_output(self, store):
        small = store.repomap(token_budget=1)
        assert len(small) <= 2  # budget cuts it down


# ---------------------------------------------------------------------------
# context_for
# ---------------------------------------------------------------------------


class TestContextFor:
    def test_bundle_has_callees(self, store):
        ctx = store.context_for("ОбработкаПроведения")
        assert ctx is not None
        callee_names = {c["name"] for c in ctx["callees"]}
        assert "ЗначениеРеквизита" in callee_names
        assert "ПолучитьСумму" in callee_names

    def test_bundle_has_touched_objects(self, store):
        ctx = store.context_for("ОбработкаПроведения")
        assert "Справочник.Номенклатура" in ctx["touched_objects"]

    def test_bundle_has_registers(self, store):
        ctx = store.context_for("ОбработкаПроведения")
        assert any("Продажи" in r for r in ctx["registers"])

    def test_bundle_has_body(self, store):
        ctx = store.context_for("ОбработкаПроведения")
        assert "ЗначениеРеквизита" in ctx["body"]

    def test_callers_resolved(self, store):
        # ПолучитьСумму is called by ОбработкаПроведения (local edge)
        ctx = store.context_for("ПолучитьСумму")
        assert any(c["name"] == "ОбработкаПроведения" for c in ctx["callers"])

    def test_missing_symbol_returns_none(self, store):
        assert store.context_for("НетТакой") is None


# ---------------------------------------------------------------------------
# triggers_on_write — the killer query
# ---------------------------------------------------------------------------


class TestTriggersOnWrite:
    def test_object_handlers(self, store):
        res = store.triggers_on_write("Документ.РеализацияТоваров")
        handlers = {h["handler"] for h in res["object_handlers"]}
        assert "ОбработкаПроведения" in handlers
        assert "ПередЗаписью" in handlers

    def test_subscriptions_included(self, store):
        res = store.triggers_on_write("Документ.РеализацияТоваров")
        assert any(s["name"] == "ПередЗаписьюРеализации" for s in res["subscriptions"])
        assert all(s["resolved"] for s in res["subscriptions"])

    def test_register_movements(self, store):
        res = store.triggers_on_write("Документ.РеализацияТоваров")
        assert any("Продажи" in m for m in res["register_movements"])

    def test_event_filter(self, store):
        res = store.triggers_on_write("Документ.РеализацияТоваров", event="ПередЗаписью")
        handlers = {h["handler"] for h in res["object_handlers"]}
        assert "ПередЗаписью" in handlers
        assert "ОбработкаПроведения" not in handlers  # filtered out


# ---------------------------------------------------------------------------
# verify oracle
# ---------------------------------------------------------------------------


class TestVerify:
    def test_verify_call_true(self, store):
        v = store.verify_call("ОбработкаПроведения", "ЗначениеРеквизита")
        assert v["holds"]
        assert v["edge_type"] == "call_commonmodule"

    def test_verify_call_false(self, store):
        v = store.verify_call("ОбработкаПроведения", "НесуществующийМетод")
        assert not v["holds"]

    def test_verify_field_true(self, store):
        v = store.verify_field("Документ.РеализацияТоваров", "Сумма")
        assert v["holds"]

    def test_verify_field_false(self, store):
        v = store.verify_field("Документ.РеализацияТоваров", "ФиктивноеПоле")
        assert not v["holds"]
        assert "available" in v


# ---------------------------------------------------------------------------
# cards
# ---------------------------------------------------------------------------


class TestCard:
    def test_card_basic(self, store):
        c = store.card("ОбработкаПроведения")
        assert c is not None
        assert c["owner"] == "РеализацияТоваров"
        assert "ЗначениеРеквизита" in c["calls"]
        assert "Справочник.Номенклатура" in c["touches"]

    def test_card_summary_string(self, store):
        c = store.card("ЗначениеРеквизита")
        assert "ЗначениеРеквизита" in c["summary"]
        assert c["owner"] == "ОбщегоНазначения"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
