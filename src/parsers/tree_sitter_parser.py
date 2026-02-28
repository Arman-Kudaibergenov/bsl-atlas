"""Tree-sitter based BSL parser.

Provides precise AST-level parsing as a drop-in replacement for the regex parser.
Falls back gracefully if the compiled grammar library is not available.
"""
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Path where the compiled grammar is baked into the Docker image
_LIB_PATH = Path("/app/lib/bsl.so")

_language = None
_parser = None
_available = False


def _init() -> bool:
    global _language, _parser, _available
    if _available:
        return True
    try:
        import ctypes
        from tree_sitter import Language, Parser  # type: ignore

        if not _LIB_PATH.exists():
            logger.debug(f"tree-sitter-bsl library not found at {_LIB_PATH} — using regex fallback")
            return False

        # tree-sitter >= 0.22 API: load .so via ctypes, pass raw pointer to Language()
        lib = ctypes.cdll.LoadLibrary(str(_LIB_PATH))
        fn = lib.tree_sitter_bsl
        fn.restype = ctypes.c_void_p
        _language = Language(fn())
        _parser = Parser(_language)
        _available = True
        logger.info("tree-sitter-bsl initialized — precise AST parsing enabled")
        return True
    except Exception as e:
        logger.debug(f"tree-sitter-bsl unavailable: {e} — using regex fallback")
        return False


def is_available() -> bool:
    return _available


def _text(node, src: bytes) -> str:
    return src[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


_FUNC_TYPES = {"function_definition", "procedure_definition"}

# Call node types in BSL grammar
_CALL_TYPES = {"method_call", "call_expression", "call_statement"}

_KEYWORDS_LOWER = {
    "если", "пока", "для", "попытка", "исключение", "возврат", "новый",
    "иначеесли", "иначе", "конецесли", "конецпока", "конецдля",
    "конеццикла", "конецпопытки", "конецпроцедуры", "конецфункции",
    "if", "while", "for", "try", "except", "return", "new",
    "elsif", "else", "endif", "endwhile", "endfor", "endtry",
}


def _extract_params(params_node, src: bytes) -> list[str]:
    """Extract parameter names from the parameters AST node."""
    if params_node is None:
        return []
    result = []
    for child in params_node.named_children:
        # Each child is a `parameter` node with field name='name'
        name_node = child.child_by_field_name("name")
        if name_node:
            result.append(_text(name_node, src))
        elif child.type == "identifier":
            result.append(_text(child, src))
    return result


def _extract_calls(func_node, src: bytes, func_name_lower: str) -> list[str]:
    """Recursively collect all function/method call names within a node."""
    calls: list[str] = []
    seen: set[str] = set()

    def _walk(node):
        if node.type in _CALL_TYPES:
            name_node = node.child_by_field_name("name")
            if name_node is None:
                # Some grammars put name as first identifier child
                for c in node.children:
                    if c.type == "identifier":
                        name_node = c
                        break
            if name_node:
                name = _text(name_node, src)
                nl = name.lower()
                if nl not in _KEYWORDS_LOWER and nl != func_name_lower and name not in seen:
                    seen.add(name)
                    calls.append(name)
            # Still recurse into arguments
        for child in node.children:
            # Don't recurse into nested function definitions
            if child.type not in _FUNC_TYPES:
                _walk(child)

    # Walk only the body, not the signature
    for child in func_node.children:
        if child.type not in {"identifier", "parameters"} and "keyword" not in child.type:
            _walk(child)

    return calls


def parse_functions(file_path: Path, src: bytes) -> list[dict[str, Any]] | None:
    """Parse BSL source bytes and return function list.

    Returns None if tree-sitter is unavailable (caller should use regex fallback).
    Each dict matches the BSLFunction field set expected by code.py.
    """
    if not _available:
        return None

    try:
        tree = _parser.parse(src)
        root = tree.root_node

        results: list[dict[str, Any]] = []

        def _visit(node):
            if node.type in _FUNC_TYPES:
                name_node = node.child_by_field_name("name")
                if name_node is None:
                    return
                name = _text(name_node, src)
                func_type = "Функция" if "function" in node.type else "Процедура"
                params = _extract_params(node.child_by_field_name("parameters"), src)
                is_export = node.child_by_field_name("export") is not None
                calls = _extract_calls(node, src, name.lower())
                body = _text(node, src)
                line_start = node.start_point[0] + 1
                line_end = node.end_point[0] + 1

                results.append({
                    "name": name,
                    "type": func_type,
                    "params": params,
                    "is_export": is_export,
                    "line_start": line_start,
                    "line_end": line_end,
                    "calls": calls,
                    "body": body,
                })
                return  # don't recurse into nested definitions

            for child in node.children:
                _visit(child)

        _visit(root)

        logger.debug(f"tree-sitter parsed {file_path.name}: {len(results)} functions")
        return results

    except Exception as e:
        logger.warning(f"tree-sitter parse error in {file_path}: {e}")
        return None


# Initialise on import
_init()
