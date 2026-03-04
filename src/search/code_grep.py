"""BSL code grep with AST context.

Searches text patterns across .bsl files in SOURCE_PATH.
For each match, determines the containing function/procedure
via tree-sitter (or regex fallback).
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Regex fallback: match Процедура/Функция / Procedure/Function keyword lines
_FUNC_RE = re.compile(
    r"^\s*(?:Процедура|Функция|Procedure|Function)\s+(\w+)",
    re.IGNORECASE,
)
_ENDFUNC_RE = re.compile(
    r"^\s*(?:КонецПроцедуры|КонецФункции|EndProcedure|EndFunction)\b",
    re.IGNORECASE,
)


def _find_function_regex(lines: list[str], target_line: int) -> str:
    """Return function name containing target_line (1-based) using regex."""
    current_func: str | None = None
    for idx, line in enumerate(lines, start=1):
        m = _FUNC_RE.match(line)
        if m:
            current_func = m.group(1)
        if _ENDFUNC_RE.match(line):
            if idx >= target_line and current_func:
                return current_func
            current_func = None
        if idx == target_line:
            return current_func or "module level"
    return current_func or "module level"


def _find_function_ts(file_path: Path, src: bytes, target_line: int) -> str | None:
    """Return function name using tree-sitter. None if unavailable."""
    try:
        from src.parsers import tree_sitter_parser as ts

        if not ts.is_available():
            return None
        functions = ts.parse_functions(file_path, src)
        if functions is None:
            return None
        for fn in functions:
            if fn["line_start"] <= target_line <= fn["line_end"]:
                return fn["name"]
        return "module level"
    except Exception:
        return None


def _extract_module_type(file_path: Path) -> str:
    """Map file path to module type string (mirrors code.py logic)."""
    parts = file_path.parts
    if "Forms" in parts and file_path.name == "Module.bsl" and "Form" in parts:
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


def _grep_file(
    bsl_file: Path,
    source_path: Path,
    pattern: str,
    case_sensitive: bool,
) -> list[dict[str, Any]]:
    """Search a single BSL file. Returns list of match dicts."""
    try:
        raw = bsl_file.read_bytes()
        text = raw.decode("utf-8-sig", errors="replace")
    except OSError:
        return []

    search_text = text if case_sensitive else text.lower()
    search_pat = pattern if case_sensitive else pattern.lower()

    if search_pat not in search_text:
        return []

    lines = text.splitlines()
    module_type = _extract_module_type(bsl_file)
    rel_path = str(bsl_file.relative_to(source_path)).replace("\\", "/")

    matches: list[dict[str, Any]] = []
    for line_no, line in enumerate(lines, start=1):
        check = line if case_sensitive else line.lower()
        if search_pat not in check:
            continue

        # Determine containing function
        func_name = _find_function_ts(bsl_file, raw, line_no)
        if func_name is None:
            func_name = _find_function_regex(lines, line_no)

        # Context: 2 lines before + 2 lines after
        ctx_start = max(0, line_no - 3)
        ctx_end = min(len(lines), line_no + 2)
        context_lines = lines[ctx_start:ctx_end]

        matches.append({
            "file": rel_path,
            "line": line_no,
            "text": line.strip(),
            "function": func_name,
            "module_type": module_type,
            "context": "\n".join(context_lines),
        })

    return matches


class CodeGrep:
    """Grep .bsl files with AST context enrichment."""

    def search(
        self,
        pattern: str,
        source_path: Path,
        case_sensitive: bool = False,
        limit: int = 20,
        max_workers: int = 8,
    ) -> list[dict[str, Any]]:
        """Search for pattern across all .bsl files under source_path.

        Args:
            pattern: Substring to search for.
            source_path: Root directory with .bsl files.
            case_sensitive: If False (default), performs case-insensitive search.
            limit: Maximum matches to return.
            max_workers: Thread pool size for parallel file scanning.

        Returns:
            List of match dicts (file, line, text, function, module_type, context).
        """
        if not pattern:
            return []

        bsl_files = list(source_path.rglob("*.bsl"))
        if not bsl_files:
            logger.warning(f"No .bsl files found in {source_path}")
            return []

        logger.debug(f"code_grep: scanning {len(bsl_files)} files for {pattern!r}")

        results: list[dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_grep_file, f, source_path, pattern, case_sensitive): f
                for f in bsl_files
            }
            for future in as_completed(futures):
                try:
                    matches = future.result()
                    results.extend(matches)
                except Exception as e:
                    logger.debug(f"code_grep file error: {e}")

                if len(results) >= limit * 10:
                    # Early exit — collected enough raw results before dedup/trim
                    break

        results.sort(key=lambda r: (r["file"], r["line"]))
        return results[:limit]
