"""Parser for 1C metadata text report files.

Ported from comol/1c_code_metadata_mcp with improvements.
"""

import logging
import re
from pathlib import Path
from typing import Any

import chardet

logger = logging.getLogger(__name__)


class MetadataParser:
    """Parser for 1C metadata TXT files."""

    # Mapping from plural to singular forms
    PLURAL_TO_SINGULAR = {
        "Справочники": "Справочник",
        "Документы": "Документ",
        "Перечисления": "Перечисление",
        "Отчеты": "Отчет",
        "Обработки": "Обработка",
        "ПланыВидовХарактеристик": "ПланВидовХарактеристик",
        "ПланыСчетов": "ПланСчетов",
        "РегистрыСведений": "РегистрСведений",
        "РегистрыНакопления": "РегистрНакопления",
        "БизнесПроцессы": "БизнесПроцесс",
        "Задачи": "Задача",
        "Константы": "Константа",
        "ОбщиеМодули": "ОбщийМодуль",
        "ОбщиеФормы": "ОбщаяФорма",
        "ОбщиеКартинки": "ОбщаяКартинка",
        "Роли": "Роль",
        "Интерфейсы": "Интерфейс",
        "КритерииОтбора": "КритерийОтбора",
        "ПланыОбмена": "ПланОбмена",
        "ПодпискиНаСобытия": "ПодпискаНаСобытие",
        "РегламентныеЗадания": "РегламентноеЗадание",
        "ХранилищаНастроек": "ХранилищеНастроек",
        "ФункциональныеОпции": "ФункциональнаяОпция",
        "ПараметрыФункциональныхОпций": "ПараметрФункциональныхОпций",
        "ОпределяемыеТипы": "ОпределяемыйТип",
        "Стили": "Стиль",
        "Языки": "Язык",
    }

    # Sub-element types
    SUB_ELEMENT_TYPES = {
        "Реквизиты": "Реквизит",
        "ТабличныеЧасти": "ТабличнаяЧасть",
        "Формы": "Форма",
        "Команды": "Команда",
        "Макеты": "Макет",
        "Значения": "ЗначениеПеречисления",
    }

    def _get_indentation(self, line: str) -> int:
        """Count leading whitespace characters."""
        count = 0
        for char in line:
            if char == " ":
                count += 1
            elif char == "\t":
                count += 4
            else:
                break
        return count

    def _parse_full_path(self, path_str: str) -> tuple[str, str]:
        """Derive object_type and name from a full_path string.

        Example: "Справочники.APA_Модели.Реквизиты.Провайдер"
        -> object_type="Реквизит", name="Провайдер"
        """
        parts = path_str.split(".")
        if not parts:
            return "НеизвестныйТип", "НеизвестноеИмя"

        name = parts[-1]
        obj_type = "НеизвестныйТипОбъекта"

        if len(parts) == 1:
            obj_type = self.PLURAL_TO_SINGULAR.get(parts[0], parts[0])
        elif len(parts) > 1:
            if parts[0] in self.PLURAL_TO_SINGULAR:
                if len(parts) == 2:
                    obj_type = parts[0]
                elif len(parts) > 2 and parts[-2] in self.SUB_ELEMENT_TYPES:
                    obj_type = self.SUB_ELEMENT_TYPES[parts[-2]]
                else:
                    obj_type = parts[-2]
            else:
                if len(parts) > 2 and parts[-2] in self.SUB_ELEMENT_TYPES:
                    obj_type = self.SUB_ELEMENT_TYPES[parts[-2]]
                else:
                    obj_type = parts[0]

        return obj_type, name

    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding using chardet."""
        with open(file_path, "rb") as f:
            raw_data = f.read(10000)
            result = chardet.detect(raw_data)
            encoding = result.get("encoding", "utf-8")
            confidence = result.get("confidence", 0)
            logger.debug(f"Detected encoding: {encoding} (confidence: {confidence})")
            return encoding or "utf-8"

    def _read_file(self, file_path: Path) -> list[str]:
        """Read file with automatic encoding detection."""
        # Priority encodings for 1C files
        encodings = ["utf-16-le", "utf-16", "utf-8", "cp1251", "windows-1251"]

        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding, errors="strict") as f:
                    lines = f.readlines()
                    logger.debug(f"Successfully read {file_path} with encoding: {encoding}")
                    return lines
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                logger.warning(f"Error reading {file_path} with {encoding}: {e}")
                continue

        # Fallback to chardet
        detected = self._detect_encoding(file_path)
        try:
            with open(file_path, "r", encoding=detected, errors="replace") as f:
                return f.readlines()
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return []

    def parse_file(self, file_path: str | Path) -> list[dict[str, Any]]:
        """Parse a plain text metadata report file.

        Args:
            file_path: Path to the metadata TXT file

        Returns:
            List of parsed metadata objects
        """
        file_path = Path(file_path)
        objects: list[dict[str, Any]] = []
        current_obj: dict[str, Any] | None = None
        current_obj_indent: int = -1

        file_lines = self._read_file(file_path)
        if not file_lines:
            logger.error(f"Failed to read file: {file_path}")
            return objects

        idx = 0
        while idx < len(file_lines):
            line = file_lines[idx].rstrip("\r\n")
            stripped = line.strip()
            indent = self._get_indentation(line)

            if not stripped:
                idx += 1
                continue

            # New object definition (starts with "- ")
            if stripped.startswith("- "):
                if current_obj:
                    objects.append(current_obj)

                full_path = stripped[2:].strip()
                obj_type, name = self._parse_full_path(full_path)
                current_obj = {
                    "full_path": full_path,
                    "object_type": obj_type,
                    "name": name,
                    "properties": {},
                    "source_file": str(file_path),
                }
                current_obj_indent = indent
                idx += 1
                continue

            if not current_obj:
                idx += 1
                continue

            # End of current object's attributes
            if indent <= current_obj_indent:
                if current_obj:
                    objects.append(current_obj)
                    current_obj = None
                    current_obj_indent = -1
                continue

            # Parse attribute: Key: "Value"
            match_simple = re.match(r'^([^:]+):\s*"(.*)"\s*$', stripped)
            if match_simple:
                key, value = match_simple.group(1).strip(), match_simple.group(2)
                current_obj["properties"][key] = value
                idx += 1
                continue

            # Parse multiline attribute
            match_key = re.match(r"^([^:]+):\s*$", stripped)
            if match_key:
                key = match_key.group(1).strip()
                multiline_parts = []
                key_indent = indent

                val_idx = idx + 1
                first_val_indent = -1

                while val_idx < len(file_lines):
                    next_line = file_lines[val_idx].rstrip("\r\n")
                    next_stripped = next_line.strip()
                    next_indent = self._get_indentation(next_line)

                    if not next_stripped and first_val_indent != -1:
                        multiline_parts.append("")
                        val_idx += 1
                        continue

                    if first_val_indent == -1:
                        if next_indent > key_indent and next_stripped.startswith('"'):
                            first_val_indent = next_indent
                        else:
                            break

                    if next_indent < first_val_indent or (
                        next_indent == first_val_indent
                        and ":" in next_stripped
                        and not next_stripped.startswith('"')
                    ):
                        break

                    part = next_stripped
                    is_last = False
                    if part.startswith('"'):
                        part = part[1:]
                    if part.endswith('"') and not part.endswith('\\"'):
                        part = part[:-1]
                        is_last = True

                    multiline_parts.append(part)
                    val_idx += 1
                    if is_last:
                        break

                idx = val_idx
                current_obj["properties"][key] = "\n".join(multiline_parts)
                continue

            idx += 1

        if current_obj:
            objects.append(current_obj)

        logger.info(f"Parsed {file_path}: found {len(objects)} objects")
        return objects
