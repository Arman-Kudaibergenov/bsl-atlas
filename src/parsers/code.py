"""Parser for 1C BSL code files.

Parses .bsl files and extracts function/procedure information.
"""

import logging
import re
from pathlib import Path
from typing import Any

import chardet

from ..storage.models import BSLFunction
from . import tree_sitter_parser

logger = logging.getLogger(__name__)


class CodeParser:
    """Parser for 1C BSL (Built-in Scripting Language) code files."""

    # Patterns for BSL code
    PROCEDURE_PATTERN = re.compile(
        r"^\s*(Процедура|Procedure)\s+(\w+)\s*\((.*?)\)",
        re.IGNORECASE | re.MULTILINE,
    )
    FUNCTION_PATTERN = re.compile(
        r"^\s*(Функция|Function)\s+(\w+)\s*\((.*?)\)",
        re.IGNORECASE | re.MULTILINE,
    )
    END_PATTERN = re.compile(
        r"^\s*(КонецПроцедуры|КонецФункции|EndProcedure|EndFunction)\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    COMMENT_PATTERN = re.compile(r"//.*$", re.MULTILINE)
    EXPORT_PATTERN = re.compile(r"\bЭкспорт\b|\bExport\b", re.IGNORECASE)

    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding using chardet."""
        with open(file_path, "rb") as f:
            raw_data = f.read(10000)
            result = chardet.detect(raw_data)
            return result.get("encoding", "utf-8") or "utf-8"

    def _read_file(self, file_path: Path) -> str:
        """Read file with automatic encoding detection."""
        encodings = ["utf-8-sig", "utf-8", "cp1251", "windows-1251", "utf-16"]

        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding, errors="strict") as f:
                    content = f.read()
                    logger.debug(f"Read {file_path} with encoding: {encoding}")
                    return content
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                logger.warning(f"Error reading {file_path} with {encoding}: {e}")
                continue

        # Fallback to chardet
        detected = self._detect_encoding(file_path)
        try:
            with open(file_path, "r", encoding=detected, errors="replace") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return ""

    def _extract_object_path(self, file_path: Path) -> str:
        """Extract 1C object path from file path.

        Example: src/Catalogs/Контрагенты/Ext/ObjectModule.bsl
        -> Справочники.Контрагенты.МодульОбъекта
        """
        parts = file_path.parts
        path_parts = []

        # Find the metadata type folder
        type_mappings = {
            "Catalogs": "Справочники",
            "Documents": "Документы",
            "CommonModules": "ОбщиеМодули",
            "Reports": "Отчеты",
            "DataProcessors": "Обработки",
            "InformationRegisters": "РегистрыСведений",
            "AccumulationRegisters": "РегистрыНакопления",
            "Enums": "Перечисления",
            "ChartsOfCharacteristicTypes": "ПланыВидовХарактеристик",
            "ChartsOfAccounts": "ПланыСчетов",
            "BusinessProcesses": "БизнесПроцессы",
            "Tasks": "Задачи",
            "ExchangePlans": "ПланыОбмена",
        }

        module_mappings = {
            "ObjectModule.bsl": "МодульОбъекта",
            "ManagerModule.bsl": "МодульМенеджера",
            "Module.bsl": "Модуль",
            "RecordSetModule.bsl": "МодульНабораЗаписей",
            "CommandModule.bsl": "МодульКоманды",
        }

        for i, part in enumerate(parts):
            if part in type_mappings:
                path_parts.append(type_mappings[part])
                # Next part is the object name
                if i + 1 < len(parts):
                    path_parts.append(parts[i + 1])

        # Add module type from filename
        filename = file_path.name
        if filename in module_mappings:
            path_parts.append(module_mappings[filename])
        else:
            path_parts.append(file_path.stem)

        return ".".join(path_parts) if path_parts else str(file_path)

    def _extract_functions(self, content: str) -> list[dict[str, Any]]:
        """Extract function and procedure definitions from BSL code."""
        functions = []

        # Find all procedure starts
        for match in self.PROCEDURE_PATTERN.finditer(content):
            func_info = {
                "type": "Процедура",
                "name": match.group(2),
                "params": match.group(3).strip(),
                "is_export": bool(self.EXPORT_PATTERN.search(content[match.end() : match.end() + 50])),
                "start_pos": match.start(),
            }
            functions.append(func_info)

        # Find all function starts
        for match in self.FUNCTION_PATTERN.finditer(content):
            func_info = {
                "type": "Функция",
                "name": match.group(2),
                "params": match.group(3).strip(),
                "is_export": bool(self.EXPORT_PATTERN.search(content[match.end() : match.end() + 50])),
                "start_pos": match.start(),
            }
            functions.append(func_info)

        # Sort by position
        functions.sort(key=lambda x: x["start_pos"])
        return functions

    _CALL_PATTERN = re.compile(r"([А-Яа-яA-Za-z_]\w*)\s*\(")
    _BSL_KEYWORDS = {
        "Если", "Пока", "Для", "Попытка", "Исключение", "Возврат", "Новый",
        "ИначеЕсли", "Иначе", "КонецЕсли", "КонецПока", "КонецДля",
        "КонецЦикла", "КонецПопытки", "КонецПроцедуры", "КонецФункции",
        "Function", "Procedure", "EndProcedure", "EndFunction",
        "If", "While", "For", "Try", "Except", "Return", "New",
        "ElsIf", "Else", "EndIf", "EndWhile", "EndFor", "EndTry",
    }

    @staticmethod
    def _extract_module_type(file_path: Path) -> str:
        """Map file path to module type string."""
        parts = file_path.parts
        str_path = "/".join(parts)

        if "Forms" in parts:
            idx = parts.index("Forms")
            # Forms/<FormName>/Ext/Form/Module.bsl
            if file_path.name == "Module.bsl" and "Form" in parts:
                return "FormModule"

        if file_path.name == "ObjectModule.bsl":
            return "ObjectModule"
        if file_path.name == "ManagerModule.bsl":
            return "ManagerModule"
        if file_path.name == "RecordSetModule.bsl":
            return "RecordSetModule"
        if file_path.name == "CommandModule.bsl":
            return "CommandModule"

        if "CommonModules" in parts and file_path.name == "Module.bsl":
            return "CommonModule"

        return "Module"

    @staticmethod
    def _parse_params(params_str: str) -> list[str]:
        """Parse parameter string into list of parameter names."""
        result = []
        for raw in params_str.split(","):
            param = raw.strip()
            if not param:
                continue
            # Strip "Знач " prefix
            if param.startswith("Знач ") or param.startswith("знач "):
                param = param[5:].strip()
            # Strip default value
            if "=" in param:
                param = param[:param.index("=")].strip()
            if param:
                result.append(param)
        return result

    def parse_file_functions(self, file_path: str | Path) -> list[BSLFunction]:
        """Parse BSL file and return enriched BSLFunction list for SQLite layer.

        Uses tree-sitter for precise AST parsing when available,
        falls back to regex parser otherwise.
        """
        file_path = Path(file_path)
        content = self._read_file(file_path)
        if not content:
            return []

        # Determine module_path relative to closest src/ parent
        try:
            parts = file_path.parts
            src_idx = None
            for i in range(len(parts) - 1, -1, -1):
                if parts[i] == "src":
                    src_idx = i
                    break
            if src_idx is not None:
                module_path = "/".join(parts[src_idx + 1:])
            else:
                module_path = str(file_path)
        except Exception:
            module_path = str(file_path)

        module_type = self._extract_module_type(file_path)

        # --- Try tree-sitter first ---
        ts_results = tree_sitter_parser.parse_functions(
            file_path, content.encode("utf-8", errors="replace")
        )
        if ts_results is not None:
            return [
                BSLFunction(
                    name=f["name"],
                    type=f["type"],
                    params=f["params"],
                    is_export=f["is_export"],
                    line_start=f["line_start"],
                    line_end=f["line_end"],
                    calls=f["calls"],
                    body=f["body"],
                    module_path=module_path,
                    module_type=module_type,
                )
                for f in ts_results
            ]

        # --- Regex fallback ---
        # Collect all function/procedure starts
        starts = []
        for match in self.PROCEDURE_PATTERN.finditer(content):
            starts.append({
                "type": "Процедура",
                "name": match.group(2),
                "params_str": match.group(3).strip(),
                "is_export": bool(self.EXPORT_PATTERN.search(content[match.end(): match.end() + 100])),
                "start_pos": match.start(),
                "sig_end": match.end(),
            })
        for match in self.FUNCTION_PATTERN.finditer(content):
            starts.append({
                "type": "Функция",
                "name": match.group(2),
                "params_str": match.group(3).strip(),
                "is_export": bool(self.EXPORT_PATTERN.search(content[match.end(): match.end() + 100])),
                "start_pos": match.start(),
                "sig_end": match.end(),
            })
        starts.sort(key=lambda x: x["start_pos"])

        # Collect all end markers as (start_of_match, match) pairs
        end_markers = [(m.start(), m) for m in self.END_PATTERN.finditer(content)]

        results: list[BSLFunction] = []
        for func in starts:
            start_pos = func["start_pos"]
            line_start = content[:start_pos].count("\n") + 1

            # Find first end marker after this function's start
            end_match = None
            for end_start, m in end_markers:
                if end_start > start_pos:
                    end_match = m
                    break

            if end_match is not None:
                end_pos = end_match.start()
                body = content[start_pos: end_match.end()]
                line_end = content[:end_pos].count("\n") + 1
            else:
                body = content[start_pos:]
                line_end = content.count("\n") + 1

            params = self._parse_params(func["params_str"])

            # Extract calls from body
            calls = []
            seen = set()
            func_name_lower = func["name"].lower()
            for call_match in self._CALL_PATTERN.finditer(body):
                name = call_match.group(1)
                if name in self._BSL_KEYWORDS:
                    continue
                if name.lower() == func_name_lower:
                    continue
                if name not in seen:
                    seen.add(name)
                    calls.append(name)

            results.append(BSLFunction(
                name=func["name"],
                type=func["type"],
                params=params,
                is_export=func["is_export"],
                line_start=line_start,
                line_end=line_end,
                calls=calls,
                body=body,
                module_path=module_path,
                module_type=module_type,
            ))

        return results

    def parse_file(self, file_path: str | Path) -> list[dict[str, Any]]:
        """Parse a BSL code file.

        Args:
            file_path: Path to the .bsl file

        Returns:
            List containing file info and extracted functions
        """
        file_path = Path(file_path)
        content = self._read_file(file_path)

        if not content:
            logger.warning(f"Empty or unreadable file: {file_path}")
            return []

        object_path = self._extract_object_path(file_path)
        functions = self._extract_functions(content)

        # Return a single document with full file content and metadata
        result = {
            "full_path": object_path,
            "object_type": "КодМодуля",
            "name": file_path.stem,
            "source_file": str(file_path),
            "content": content,
            "functions": functions,
            "function_names": [f["name"] for f in functions],
            "export_functions": [f["name"] for f in functions if f["is_export"]],
            "line_count": content.count("\n") + 1,
        }

        logger.info(
            f"Parsed {file_path}: {len(functions)} functions/procedures, "
            f"{result['line_count']} lines"
        )
        return [result]

    def parse_directory(
        self,
        directory: str | Path,
        extensions: tuple[str, ...] = (".bsl",),
    ) -> list[dict[str, Any]]:
        """Parse all BSL files in a directory recursively.

        Args:
            directory: Root directory to scan
            extensions: File extensions to process

        Returns:
            List of parsed code objects
        """
        directory = Path(directory)
        results = []

        for ext in extensions:
            for file_path in directory.rglob(f"*{ext}"):
                try:
                    parsed = self.parse_file(file_path)
                    results.extend(parsed)
                except Exception as e:
                    logger.error(f"Error parsing {file_path}: {e}")

        logger.info(f"Parsed {len(results)} code files from {directory}")
        return results
