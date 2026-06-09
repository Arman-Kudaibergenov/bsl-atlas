"""Parser for 1C managed form definitions (Form.xml).

Managed-form XML is large and schema-heavy; this is a tolerant, namespace-agnostic
extractor for the parts that matter for navigation: form attributes, commands,
items, and — most usefully — event handlers. Handlers are linked to the actual
procedures in the sibling form module (Ext/Form/Module.bsl), which is precise
because it resolves within ONE file (no cross-form name collisions).
"""

import logging
from pathlib import Path
from xml.etree import ElementTree as ET

from .code import CodeParser

logger = logging.getLogger(__name__)


class FormXMLParser:
    """Extract structure + event handlers from a managed Form.xml."""

    def __init__(self) -> None:
        self.code_parser = CodeParser()

    @staticmethod
    def _strip_ns(tag: str) -> str:
        return tag[tag.index("}") + 1:] if tag.startswith("{") else tag

    def _iter(self, root: ET.Element):
        """Depth-first walk yielding (local_tag, element)."""
        stack = [root]
        while stack:
            el = stack.pop()
            yield self._strip_ns(el.tag), el
            stack.extend(list(el))

    def _name_of(self, el: ET.Element) -> str:
        """Element name from a `name`/`Name` attribute or a <Name> child."""
        for key in ("name", "Name"):
            if el.get(key):
                return el.get(key)
        for child in el:
            if self._strip_ns(child.tag) == "Name" and child.text:
                return child.text.strip()
        # Configurator dump: name lives under Properties/Name
        for child in el:
            if self._strip_ns(child.tag) == "Properties":
                for gc in child:
                    if self._strip_ns(gc.tag) == "Name" and gc.text:
                        return gc.text.strip()
        return ""

    def parse_file(self, file_path: str | Path) -> dict | None:
        """Parse a Form.xml into a structured dict.

        Returns: {name, attributes, commands, items, handlers, resolved_handlers}.
        resolved_handlers = handler methods that actually exist in the form module.
        """
        file_path = Path(file_path)
        try:
            root = ET.parse(file_path).getroot()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to parse Form.xml {file_path}: {e}")
            return None

        attributes: list[str] = []
        commands: list[str] = []
        items: list[str] = []
        handlers: list[dict] = []
        form_name = ""

        for tag, el in self._iter(root):
            if tag in ("Form",) and not form_name:
                form_name = self._name_of(el)
            elif tag == "Event":
                ev = el.get("name") or el.get("Name") or ""
                method = (el.text or "").strip()
                # Configurator dump sometimes stores handler in CallType/Name children
                if not method:
                    for child in el:
                        if self._strip_ns(child.tag) in ("Name", "Handler") and child.text:
                            method = child.text.strip()
                            break
                if method:
                    handlers.append({"event": ev, "handler": method})
            elif tag == "Attribute":
                nm = self._name_of(el)
                if nm:
                    attributes.append(nm)
            elif tag == "Command":
                nm = self._name_of(el)
                if nm:
                    commands.append(nm)
            elif tag in (
                "InputField", "Button", "Table", "Group", "LabelField",
                "CheckBoxField", "RadioButtonField", "PictureField",
            ):
                nm = self._name_of(el)
                if nm:
                    items.append(nm)

        if not form_name:
            # Forms/<Name>/Ext/Form.xml -> <Name>
            parts = file_path.parts
            if "Forms" in parts:
                i = parts.index("Forms")
                if i + 1 < len(parts):
                    form_name = parts[i + 1]

        resolved = self._resolve_handlers(file_path, handlers)

        return {
            "name": form_name,
            "attributes": attributes,
            "commands": commands,
            "items": items,
            "handlers": handlers,
            "resolved_handlers": resolved,
            "module_path": str(self._module_path(file_path)),
        }

    @staticmethod
    def _module_path(form_xml: Path) -> Path:
        """Sibling form module: <...>/Ext/Form.xml -> <...>/Ext/Form/Module.bsl."""
        return form_xml.parent / "Form" / "Module.bsl"

    def _resolve_handlers(self, form_xml: Path, handlers: list[dict]) -> list[str]:
        """Which declared handler methods actually exist in the form module."""
        module = self._module_path(form_xml)
        if not module.exists():
            return []
        try:
            funcs = self.code_parser.parse_file_functions(module)
        except Exception:  # noqa: BLE001
            return []
        proc_names = {f.name for f in funcs}
        declared = {h["handler"] for h in handlers}
        return sorted(declared & proc_names)
