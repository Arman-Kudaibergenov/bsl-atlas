"""Parser for 1C DataCompositionSchema (СКД) templates.

Reports/SKD live as Template XML (dataCompositionSchema namespace). The valuable,
low-ambiguity extraction for navigation is: each dataset's query text, the object
refs inside it (code->data, reusing the BSL query-ref miner), the available fields,
and the schema parameters. Read-on-demand — not pushed into the global graph.
"""

import logging
from pathlib import Path
from xml.etree import ElementTree as ET

from .code import CodeParser

logger = logging.getLogger(__name__)


class SKDXMLParser:
    """Extract datasets / query text / fields / parameters from a СКД template."""

    @staticmethod
    def _strip_ns(tag: str) -> str:
        return tag[tag.index("}") + 1:] if tag.startswith("{") else tag

    def _iter(self, root: ET.Element):
        stack = [root]
        while stack:
            el = stack.pop()
            yield self._strip_ns(el.tag), el
            stack.extend(list(el))

    def parse_file(self, file_path: str | Path) -> dict | None:
        """Parse a СКД template XML.

        Returns: {datasets: [{name, query, query_refs, fields}], parameters,
        query_refs (union)}. Returns None if the file isn't a parseable schema.
        """
        file_path = Path(file_path)
        try:
            root = ET.parse(file_path).getroot()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to parse СКД XML {file_path}: {e}")
            return None

        # A schema is recognised by having at least one <query> or <dataSet>
        tags = {self._strip_ns(el.tag) for _, el in self._iter(root)}
        if "dataSet" not in tags and "query" not in tags:
            return None

        datasets: list[dict] = []
        parameters: list[str] = []
        all_refs: set[str] = set()

        for tag, el in self._iter(root):
            if tag == "dataSet":
                ds_name = self._child_text(el, "name")
                query = ""
                fields: list[str] = []
                for ctag, child in self._iter(el):
                    if ctag == "query" and child.text:
                        query = child.text.strip()
                    elif ctag in ("dataPath", "field") and child.text:
                        val = child.text.strip()
                        if val and val not in fields:
                            fields.append(val)
                refs = CodeParser._extract_query_refs(query) if query else []
                all_refs.update(refs)
                datasets.append({
                    "name": ds_name,
                    "query": query,
                    "query_refs": refs,
                    "fields": fields,
                })
            elif tag == "parameter":
                pname = self._child_text(el, "name") or self._child_text(el, "dataPath")
                if pname:
                    parameters.append(pname)

        # Fallback: schemas that put <query> outside an explicit <dataSet>
        if not datasets:
            for tag, el in self._iter(root):
                if tag == "query" and el.text and el.text.strip():
                    q = el.text.strip()
                    refs = CodeParser._extract_query_refs(q)
                    all_refs.update(refs)
                    datasets.append({"name": "", "query": q, "query_refs": refs, "fields": []})

        return {
            "datasets": datasets,
            "parameters": parameters,
            "query_refs": sorted(all_refs),
        }

    def _child_text(self, el: ET.Element, name: str) -> str:
        for child in el:
            if self._strip_ns(child.tag) == name and child.text:
                return child.text.strip()
        return ""
