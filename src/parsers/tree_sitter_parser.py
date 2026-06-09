"""Tree-sitter based BSL parser.

Provides precise AST-level parsing as a drop-in replacement for the regex parser.
Falls back gracefully if the compiled grammar library is not available.
"""
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

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

        import warnings
        lib = ctypes.cdll.LoadLibrary(str(_LIB_PATH))
        fn = lib.tree_sitter_bsl
        fn.restype = ctypes.c_void_p
        # Language(int) is deprecated in tree-sitter >= 0.22; capsule API requires
        # a pip-installable binding which alkoleft/tree-sitter-bsl doesn't provide yet.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
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


def _method_name(method_call_node, src: bytes) -> str | None:
    """The method identifier inside a `method_call` node (its first identifier)."""
    for c in method_call_node.named_children:
        if c.type == "identifier":
            return _text(c, src)
    return None


def _access_qualifier(access_node, src: bytes) -> str | None:
    """Rightmost name segment of an `access` (receiver) node.

    `ОбщегоНазначения`            -> 'ОбщегоНазначения'  (single identifier)
    `Справочники.Контрагенты`     -> 'Контрагенты'        (last `property` segment)
    Matches the regex parser's "immediate qualifier before the method" semantics,
    so commonmodule resolution (qualifier -> object_name) lines up across parsers.
    """
    if access_node is None:
        return None
    kids = access_node.named_children
    last = kids[-1] if kids else access_node
    if last.type in ("property", "identifier"):
        return _text(last, src)
    if last.type == "access":
        return _access_qualifier(last, src)
    return _text(last, src) if last is not access_node else None


def _extract_calls(func_node, src: bytes, func_name_lower: str) -> list[tuple[str | None, str]]:
    """Collect all call sites as (qualifier, name) within a function body.

    Grammar shape (alkoleft/tree-sitter-bsl):
      qualified call  -> call_expression( access<qualifier>, method_call<name,args> )
      unqualified call-> bare method_call<name,args>
    The qualifier lives as a SIBLING `access` of `method_call`, not inside it.
    """
    calls: list[tuple[str | None, str]] = []
    seen: set[tuple[str | None, str]] = set()

    def record(qualifier: str | None, name: str) -> None:
        nl = name.lower()
        if nl in _KEYWORDS_LOWER or nl == func_name_lower:
            return
        key = (qualifier, name)
        if key not in seen:
            seen.add(key)
            calls.append(key)

    def _walk_args(method_call_node) -> None:
        for c in method_call_node.named_children:
            if c.type == "arguments":
                for arg in c.named_children:
                    _walk(arg)

    def _walk(node) -> None:
        t = node.type
        if t in _FUNC_TYPES:
            return  # never descend into nested definitions
        if t == "call_expression":
            acc = mc = None
            for c in node.named_children:
                if c.type == "access":
                    acc = c
                elif c.type == "method_call":
                    mc = c
            if mc is not None:
                nm = _method_name(mc, src)
                if nm:
                    record(_access_qualifier(acc, src), nm)
                _walk_args(mc)
            # a receiver expression can itself contain calls, e.g. Получить().Метод()
            if acc is not None:
                _walk(acc)
            return
        if t == "method_call":
            nm = _method_name(node, src)
            if nm:
                record(None, nm)
            _walk_args(node)
            return
        for child in node.named_children:
            _walk(child)

    # Walk only the body, not the signature (identifier/parameters/keywords)
    for child in func_node.children:
        if child.type not in {"identifier", "parameters"} and "keyword" not in child.type.lower():
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
