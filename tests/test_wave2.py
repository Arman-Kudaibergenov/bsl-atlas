"""Wave 2 tests: Form.xml parser, СКД parser, RRF fusion, bulk-load correctness."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsers.form_xml import FormXMLParser
from src.parsers.skd_xml import SKDXMLParser
from src.search.hybrid import rrf_fuse
from src.storage.sqlite_store import SQLiteStore

# ---------------------------------------------------------------------------
# Form.xml
# ---------------------------------------------------------------------------

FORM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Form>
  <Properties><Name>ФормаДокумента</Name></Properties>
  <Events>
    <Event name="OnCreateAtServer">ПриСозданииНаСервере</Event>
    <Event name="BeforeWriteAtServer">ПередЗаписьюНаСервере</Event>
  </Events>
  <Attributes>
    <Attribute name="Объект"/>
  </Attributes>
  <Commands>
    <Command name="Провести"/>
  </Commands>
  <ChildItems>
    <InputField name="Сумма">
      <Events><Event name="OnChange">СуммаПриИзменении</Event></Events>
    </InputField>
    <Button name="ПровестиКнопка"/>
  </ChildItems>
</Form>
"""

FORM_MODULE = """\
Процедура ПриСозданииНаСервере(Отказ, СтандартнаяОбработка)
	Заголовок = "Документ";
КонецПроцедуры

Процедура СуммаПриИзменении(Элемент)
	ПересчитатьИтоги();
КонецПроцедуры
"""


@pytest.fixture
def form_path(tmp_path):
    ext = tmp_path / "Documents" / "РеализацияТоваров" / "Forms" / "ФормаДокумента" / "Ext"
    ext.mkdir(parents=True)
    (ext / "Form.xml").write_text(FORM_XML, encoding="utf-8")
    (ext / "Form").mkdir()
    (ext / "Form" / "Module.bsl").write_text(FORM_MODULE, encoding="utf-8")
    return ext / "Form.xml"


class TestFormParser:
    def test_name(self, form_path):
        info = FormXMLParser().parse_file(form_path)
        assert info["name"] == "ФормаДокумента"

    def test_attributes_commands_items(self, form_path):
        info = FormXMLParser().parse_file(form_path)
        assert "Объект" in info["attributes"]
        assert "Провести" in info["commands"]
        assert "Сумма" in info["items"]

    def test_handlers_declared(self, form_path):
        info = FormXMLParser().parse_file(form_path)
        methods = {h["handler"] for h in info["handlers"]}
        assert {"ПриСозданииНаСервере", "СуммаПриИзменении", "ПередЗаписьюНаСервере"} <= methods

    def test_resolved_handlers_intersect_module(self, form_path):
        info = FormXMLParser().parse_file(form_path)
        # ПередЗаписьюНаСервере declared but NOT in module -> excluded
        assert "ПриСозданииНаСервере" in info["resolved_handlers"]
        assert "СуммаПриИзменении" in info["resolved_handlers"]
        assert "ПередЗаписьюНаСервере" not in info["resolved_handlers"]


# ---------------------------------------------------------------------------
# СКД
# ---------------------------------------------------------------------------

SKD_XML = """<?xml version="1.0" encoding="UTF-8"?>
<DataCompositionSchema>
  <dataSource><name>ИсточникДанных1</name></dataSource>
  <dataSet>
    <name>НаборДанных1</name>
    <query>ВЫБРАТЬ Цена ИЗ Справочник.Номенклатура КАК Номенклатура</query>
    <field><dataPath>Цена</dataPath></field>
  </dataSet>
  <parameter><name>Период</name></parameter>
</DataCompositionSchema>
"""


@pytest.fixture
def skd_path(tmp_path):
    p = tmp_path / "Template.xml"
    p.write_text(SKD_XML, encoding="utf-8")
    return p


class TestSKDParser:
    def test_dataset_query(self, skd_path):
        info = SKDXMLParser().parse_file(skd_path)
        assert len(info["datasets"]) == 1
        assert "Номенклатура" in info["datasets"][0]["query"]

    def test_query_refs(self, skd_path):
        info = SKDXMLParser().parse_file(skd_path)
        assert "Справочник.Номенклатура" in info["query_refs"]

    def test_parameters(self, skd_path):
        info = SKDXMLParser().parse_file(skd_path)
        assert "Период" in info["parameters"]

    def test_non_skd_returns_none(self, tmp_path):
        p = tmp_path / "notskd.xml"
        p.write_text("<Root><Foo/></Root>", encoding="utf-8")
        assert SKDXMLParser().parse_file(p) is None


# ---------------------------------------------------------------------------
# RRF
# ---------------------------------------------------------------------------


class TestRRF:
    def test_fusion_order(self):
        a = [{"id": 1}, {"id": 2}, {"id": 3}]
        b = [{"id": 3}, {"id": 1}]
        fused = rrf_fuse([a, b])
        ids = [x["id"] for x in fused]
        # id1 appears high in both; id3 boosted by being top of b; id2 only once
        assert ids[0] == 1
        assert ids[-1] == 2
        assert all("rrf_score" in x for x in fused)

    def test_top_k(self):
        a = [{"id": i} for i in range(10)]
        fused = rrf_fuse([a], top_k=3)
        assert len(fused) == 3

    def test_empty(self):
        assert rrf_fuse([]) == []


# ---------------------------------------------------------------------------
# Bulk-load correctness (deferred-index rebuild)
# ---------------------------------------------------------------------------

HUB = "Функция Ц(Ссылка, П) Экспорт\n\tВозврат Ссылка[П];\nКонецФункции\n"
CALLER = (
    "Процедура ОбработкаПроведения(Отказ)\n"
    "\tЗ = ОбщегоНазначения.Ц(Ссылка, \"П\");\n"
    "КонецПроцедуры\n"
)


class TestBulkLoad:
    @pytest.fixture
    def store(self, tmp_path):
        cm = tmp_path / "CommonModules" / "ОбщегоНазначения" / "Ext" / "Module.bsl"
        cm.parent.mkdir(parents=True)
        cm.write_text(HUB, encoding="utf-8")
        om = tmp_path / "Documents" / "Реализация" / "Ext" / "ObjectModule.bsl"
        om.parent.mkdir(parents=True)
        om.write_text(CALLER, encoding="utf-8")
        s = SQLiteStore(db_path=tmp_path / "idx.db")
        s.rebuild([cm, om], [])
        return s

    def test_secondary_indices_built_after_rebuild(self, store):
        with store._get_conn() as conn:
            rows = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                ).fetchall()
            }
        assert "idx_edges_src" in rows
        assert "idx_symbols_name" in rows

    def test_data_correct_after_bulk_load(self, store):
        assert store.has_data()
        ctx = store.get_function_context("Ц")
        assert ctx is not None
        assert "ОбработкаПроведения" in {c["name"] for c in ctx.called_by}  # commonmodule edge resolved

    def test_centrality_computed(self, store):
        with store._get_conn() as conn:
            hub = conn.execute("SELECT centrality FROM symbols WHERE name='Ц'").fetchone()[0]
        assert hub > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
