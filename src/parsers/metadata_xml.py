"""Parser for 1C metadata XML files (configurator dump format).

Supports the standard 1C:Enterprise Configurator XML dump structure:
  <MetaDataObject>
    <Catalog|Document|...>
      <Properties><Name>...</Name><Synonym>...</Synonym></Properties>
      <ChildObjects>
        <Attribute|Dimension|Resource><Properties>...</Properties></Attribute>
        <TabularSection><Properties>...</Properties><ChildObjects>...</ChildObjects></TabularSection>
      </ChildObjects>
    </Catalog>
  </MetaDataObject>
"""

import logging
from pathlib import Path
from xml.etree import ElementTree as ET

from ..storage.models import MetadataObject, Attribute, TabPart

logger = logging.getLogger(__name__)


class MetadataXMLParser:

    FOLDER_TO_TYPE = {
        "Catalogs": "Справочник",
        "Documents": "Документ",
        "Enums": "Перечисление",
        "Reports": "Отчет",
        "DataProcessors": "Обработка",
        "InformationRegisters": "РегистрСведений",
        "AccumulationRegisters": "РегистрНакопления",
        "AccountingRegisters": "РегистрБухгалтерии",
        "CalculationRegisters": "РегистрРасчета",
        "ChartsOfCharacteristicTypes": "ПланВидовХарактеристик",
        "ChartsOfAccounts": "ПланСчетов",
        "BusinessProcesses": "БизнесПроцесс",
        "Tasks": "Задача",
        "ExchangePlans": "ПланОбмена",
        "Constants": "Константа",
        "Sequences": "Последовательность",
    }

    ELEMENT_TO_TYPE = {
        "Catalog": "Справочник",
        "Document": "Документ",
        "Enum": "Перечисление",
        "Report": "Отчет",
        "DataProcessor": "Обработка",
        "InformationRegister": "РегистрСведений",
        "AccumulationRegister": "РегистрНакопления",
        "AccountingRegister": "РегистрБухгалтерии",
        "CalculationRegister": "РегистрРасчета",
        "ChartOfCharacteristicTypes": "ПланВидовХарактеристик",
        "ChartOfAccounts": "ПланСчетов",
        "BusinessProcess": "БизнесПроцесс",
        "Task": "Задача",
        "ExchangePlan": "ПланОбмена",
        "Constant": "Константа",
        "Sequence": "Последовательность",
        "CommonModule": "ОбщийМодуль",
    }

    TYPE_PREFIX_MAP = {
        "CatalogRef": "СправочникСсылка",
        "DocumentRef": "ДокументСсылка",
        "EnumRef": "ПеречислениеСсылка",
        "ChartOfCharacteristicTypesRef": "ПланВидовХарактеристикСсылка",
        "ChartOfAccountsRef": "ПланСчетовСсылка",
        "BusinessProcessRef": "БизнесПроцессСсылка",
        "TaskRef": "ЗадачаСсылка",
        "ExchangePlanRef": "ПланОбменаСсылка",
        "InformationRegisterRecord": "РегистрСведенийЗапись",
        "InformationRegisterRecordSet": "РегистрСведенийНаборЗаписей",
        "AccumulationRegisterRecord": "РегистрНакопленияЗапись",
        "AccumulationRegisterRecordSet": "РегистрНакопленияНаборЗаписей",
        "AccountingRegisterRecord": "РегистрБухгалтерииЗапись",
        "CalculationRegisterRecord": "РегистрРасчетаЗапись",
        "DocumentTabularSectionRow": "ДокументТабличнаяЧастьСтрока",
    }

    PRIMITIVE_TYPE_MAP = {
        "xs:string": "Строка",
        "xs:decimal": "Число",
        "xs:boolean": "Булево",
        "xs:dateTime": "Дата",
        "xs:date": "Дата",
        "v8:ValueStorage": "ХранилищеЗначения",
        "v8:UUID": "УникальныйИдентификатор",
    }

    # Elements in ChildObjects that represent attribute-like fields
    ATTRIBUTE_ELEMENTS = {"Attribute", "Dimension", "Resource"}

    # ------------------------------------------------------------------
    # Namespace-agnostic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_ns(tag: str) -> str:
        """Remove {namespace} prefix from an XML tag."""
        if tag.startswith("{"):
            return tag[tag.index("}") + 1:]
        return tag

    def _find_child(self, el: ET.Element, name: str) -> ET.Element | None:
        for child in el:
            if self._strip_ns(child.tag) == name:
                return child
        return None

    def _find_children(self, el: ET.Element, name: str) -> list[ET.Element]:
        return [child for child in el if self._strip_ns(child.tag) == name]

    def _find_children_multi(self, el: ET.Element, names: set[str]) -> list[ET.Element]:
        """Find all children whose local tag name is in the given set."""
        return [child for child in el if self._strip_ns(child.tag) in names]

    def _get_child_text(self, el: ET.Element, child_name: str, default: str = "") -> str:
        child = self._find_child(el, child_name)
        if child is not None and child.text:
            return child.text.strip()
        return default

    # ------------------------------------------------------------------
    # Properties helper (Configurator dump format)
    # ------------------------------------------------------------------

    def _get_properties(self, obj_el: ET.Element) -> ET.Element | None:
        """Get the <Properties> child of an object element."""
        return self._find_child(obj_el, "Properties")

    def _get_child_objects(self, obj_el: ET.Element) -> ET.Element | None:
        """Get the <ChildObjects> child of an object element."""
        return self._find_child(obj_el, "ChildObjects")

    # ------------------------------------------------------------------
    # Synonym
    # ------------------------------------------------------------------

    def _parse_synonym(self, props_el: ET.Element) -> str:
        """Parse synonym from a Properties element.

        Handles two formats:
        - Configurator dump: <Synonym><v8:item><v8:lang>ru</v8:lang><v8:content>TEXT</v8:content></v8:item></Synonym>
        - Simple: <Synonym><key>ru</key><value>TEXT</value></Synonym> or <Synonym>TEXT</Synonym>
        """
        syn_el = self._find_child(props_el, "Synonym")
        if syn_el is None:
            return ""

        # Format 1 (Configurator dump): <v8:item><v8:content>TEXT</v8:content></v8:item>
        item_el = self._find_child(syn_el, "item")
        if item_el is not None:
            content_el = self._find_child(item_el, "content")
            if content_el is not None and content_el.text:
                return content_el.text.strip()

        # Format 2: <key>ru</key><value>TEXT</value>
        value_el = self._find_child(syn_el, "value")
        if value_el is not None and value_el.text:
            return value_el.text.strip()

        # Format 3: <Synonym>TEXT</Synonym>
        if syn_el.text:
            return syn_el.text.strip()

        return ""

    # ------------------------------------------------------------------
    # Type reference
    # ------------------------------------------------------------------

    def _translate_type_text(self, text: str) -> str:
        """Translate a single type string using the type maps."""
        text = text.strip()

        # Strip cfg: prefix (Configurator dump format)
        if text.startswith("cfg:"):
            text = text[4:]

        if text in self.PRIMITIVE_TYPE_MAP:
            return self.PRIMITIVE_TYPE_MAP[text]

        # Try prefix translation: "CatalogRef.SomeName" -> "СправочникСсылка.SomeName"
        if "." in text:
            prefix, rest = text.split(".", 1)
            if prefix in self.TYPE_PREFIX_MAP:
                return f"{self.TYPE_PREFIX_MAP[prefix]}.{rest}"

        return text

    def _parse_type_from_props(self, props_el: ET.Element) -> str:
        """Parse <Type> element from Properties.

        Configurator dump format:
          <Type><v8:Type>xs:string</v8:Type>...</Type>
        or with multiple types:
          <Type><v8:Type>cfg:CatalogRef.Foo</v8:Type><v8:Type>cfg:CatalogRef.Bar</v8:Type></Type>
        """
        type_el = self._find_child(props_el, "Type")
        if type_el is None:
            return ""

        parts: list[str] = []

        # Look for v8:Type children (namespace-stripped to "Type")
        for child in type_el:
            local = self._strip_ns(child.tag)
            if local == "Type" and child.text:
                parts.append(self._translate_type_text(child.text.strip()))

        # Also check TypeSet (some formats)
        if not parts:
            type_set = self._find_child(type_el, "TypeSet")
            if type_set is not None:
                for child in type_set:
                    local = self._strip_ns(child.tag)
                    if local in ("Type", "Reference") and child.text:
                        parts.append(self._translate_type_text(child.text.strip()))

        # Fallback: direct text
        if not parts and type_el.text and type_el.text.strip():
            parts.append(self._translate_type_text(type_el.text.strip()))

        return ", ".join(parts)

    # ------------------------------------------------------------------
    # Attribute parsing (Configurator dump format)
    # ------------------------------------------------------------------

    def _parse_attribute_from_child_obj(self, attr_el: ET.Element) -> Attribute | None:
        """Parse an Attribute/Dimension/Resource element from ChildObjects.

        Structure: <Attribute uuid="..."><Properties><Name>X</Name><Type>...</Type></Properties></Attribute>
        """
        props = self._get_properties(attr_el)
        if props is None:
            # Fallback: try direct children (simple format)
            name = self._get_child_text(attr_el, "Name")
            if not name:
                return None
            type_ref = ""
            type_el = self._find_child(attr_el, "Type")
            if type_el is not None:
                type_ref = self._parse_type_from_props(attr_el)
            is_required = self._get_child_text(attr_el, "FillChecking") == "ErrorIfNotFilled"
            return Attribute(name=name, type_ref=type_ref, is_required=is_required)

        name = self._get_child_text(props, "Name")
        if not name:
            return None

        type_ref = self._parse_type_from_props(props)
        is_required = self._get_child_text(props, "FillChecking") == "ErrorIfNotFilled"

        return Attribute(name=name, type_ref=type_ref, is_required=is_required)

    def _parse_attributes_from_child_objects(self, obj_el: ET.Element) -> list[Attribute]:
        """Parse all Attribute/Dimension/Resource elements from ChildObjects."""
        child_objects = self._get_child_objects(obj_el)
        if child_objects is None:
            return []

        result = []
        for el in self._find_children_multi(child_objects, self.ATTRIBUTE_ELEMENTS):
            attr = self._parse_attribute_from_child_obj(el)
            if attr is not None:
                result.append(attr)
        return result

    # ------------------------------------------------------------------
    # Tabular sections (Configurator dump format)
    # ------------------------------------------------------------------

    def _parse_tab_sections_from_child_objects(self, obj_el: ET.Element) -> list[TabPart]:
        """Parse TabularSection elements from ChildObjects.

        Structure:
          <ChildObjects>
            <TabularSection uuid="...">
              <Properties><Name>X</Name></Properties>
              <ChildObjects>
                <Attribute uuid="..."><Properties>...</Properties></Attribute>
              </ChildObjects>
            </TabularSection>
          </ChildObjects>
        """
        child_objects = self._get_child_objects(obj_el)
        if child_objects is None:
            return []

        result = []
        for ts_el in self._find_children(child_objects, "TabularSection"):
            props = self._get_properties(ts_el)
            if props is None:
                continue
            name = self._get_child_text(props, "Name")
            if not name:
                continue

            # Parse attributes of the tabular section
            attrs = self._parse_attributes_from_child_objects(ts_el)
            result.append(TabPart(name=name, attributes=attrs))

        return result

    # ------------------------------------------------------------------
    # Register records (for documents)
    # ------------------------------------------------------------------

    def _parse_register_records(self, props_el: ET.Element) -> list[str]:
        """Parse RegisterRecords from Properties element."""
        container = self._find_child(props_el, "RegisterRecords")
        if container is None:
            return []
        result = []
        for child in container:
            local = self._strip_ns(child.tag)
            if child.text and child.text.strip():
                result.append(self._translate_type_text(child.text.strip()))
            elif local:
                # Some formats list records as element names
                result.append(local)
        return result

    # ------------------------------------------------------------------
    # File-level parse
    # ------------------------------------------------------------------

    def parse_file(self, file_path: str | Path) -> MetadataObject | None:
        file_path = Path(file_path)
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
        except Exception as e:
            logger.warning(f"Failed to parse XML {file_path}: {e}")
            return None

        root_tag = self._strip_ns(root.tag)

        # Determine the object element and its type
        obj_el: ET.Element | None = None
        object_type: str = ""

        if root_tag == "MetaDataObject":
            for child in root:
                child_tag = self._strip_ns(child.tag)
                if child_tag in self.ELEMENT_TO_TYPE:
                    obj_el = child
                    object_type = self.ELEMENT_TO_TYPE[child_tag]
                    break
        elif root_tag in self.ELEMENT_TO_TYPE:
            obj_el = root
            object_type = self.ELEMENT_TO_TYPE[root_tag]

        # Fallback: infer type from parent folder name
        if object_type == "" and obj_el is None:
            for part in file_path.parts:
                if part in self.FOLDER_TO_TYPE:
                    object_type = self.FOLDER_TO_TYPE[part]
                    obj_el = root
                    break

        if obj_el is None:
            logger.debug(f"No recognized metadata element in {file_path}")
            return None

        # --- Configurator dump format: data lives inside <Properties> ---
        props_el = self._get_properties(obj_el)

        if props_el is not None:
            # Configurator dump format
            name = self._get_child_text(props_el, "Name")
            if not name:
                logger.debug(f"No Name in Properties of {file_path}")
                return None

            synonym = self._parse_synonym(props_el)
            attributes = self._parse_attributes_from_child_objects(obj_el)
            tab_parts = self._parse_tab_sections_from_child_objects(obj_el)
            registers = self._parse_register_records(props_el)
        else:
            # Simple/legacy format: data is direct children of obj_el
            name = self._get_child_text(obj_el, "Name")
            if not name:
                name = obj_el.get("Name", "")
            if not name:
                logger.debug(f"No Name found in {file_path}")
                return None

            synonym = self._parse_synonym(obj_el)
            attributes = self._parse_attributes_from_child_objects(obj_el)
            tab_parts = self._parse_tab_sections_from_child_objects(obj_el)
            registers = self._parse_register_records(obj_el)

        return MetadataObject(
            name=name,
            object_type=object_type,
            synonym=synonym,
            attributes=attributes,
            tab_parts=tab_parts,
            registers=registers,
        )

    # ------------------------------------------------------------------
    # Directory-level parse
    # ------------------------------------------------------------------

    def parse_directory(self, directory: str | Path) -> list[MetadataObject]:
        directory = Path(directory)
        results: list[MetadataObject] = []

        # Build search roots: directory itself + standard subdirs (cf/, src/)
        # For cfe/: each child directory is a separate extension
        search_roots = [directory]
        for subdir in directory.iterdir():
            if not subdir.is_dir():
                continue
            if subdir.name in ("cf", "src"):
                search_roots.append(subdir)
            elif subdir.name == "cfe":
                # cfe/ contains extension subdirectories: cfe/ExtName/Catalogs/...
                for ext_dir in subdir.iterdir():
                    if ext_dir.is_dir():
                        search_roots.append(ext_dir)

        for root in search_roots:
            for folder_name in self.FOLDER_TO_TYPE:
                folder_path = root / folder_name
                if not folder_path.exists():
                    continue
                for xml_file in folder_path.glob("*.xml"):
                    try:
                        obj = self.parse_file(xml_file)
                        if obj is not None:
                            results.append(obj)
                    except Exception as e:
                        logger.error(f"Error parsing {xml_file}: {e}")

        logger.info(f"Parsed {len(results)} metadata objects from {directory} ({len(search_roots)} search roots)")
        return results
