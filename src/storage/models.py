"""Shared data models for the dual-layer architecture.

Used by both parsers (as output types) and storage (as input/output types).
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Parser output models
# ---------------------------------------------------------------------------


@dataclass
class BSLFunction:
    """A parsed BSL function or procedure with enriched metadata."""

    name: str
    type: str               # "Процедура" | "Функция"
    params: list[str]       # parsed parameter names
    is_export: bool
    line_start: int         # 1-based line number
    line_end: int           # 1-based line number of КонецПроцедуры/КонецФункции
    calls: list[str]        # function names called inside this function
    body: str               # full text of the function body
    module_path: str        # relative file path
    module_type: str        # "ObjectModule" | "ManagerModule" | "FormModule" | "CommonModule" | ...


@dataclass
class Attribute:
    """A metadata object attribute (реквизит)."""

    name: str
    type_ref: str = ""      # e.g. "СправочникСсылка.Контрагенты" or "Строка"
    is_required: bool = False


@dataclass
class TabPart:
    """A tabular section (табличная часть) of a metadata object."""

    name: str
    attributes: list[Attribute] = field(default_factory=list)


@dataclass
class MetadataObject:
    """A parsed 1C metadata object (catalog, document, register, etc.)."""

    name: str
    object_type: str            # "Справочник" | "Документ" | "РегистрНакопления" | ...
    synonym: str = ""
    attributes: list[Attribute] = field(default_factory=list)
    tab_parts: list[TabPart] = field(default_factory=list)
    registers: list[str] = field(default_factory=list)     # register movements (for documents)


# ---------------------------------------------------------------------------
# Storage / search result models
# ---------------------------------------------------------------------------


@dataclass
class FunctionInfo:
    """Function info returned from SQLite search."""

    name: str
    type: str
    params: list[str]
    is_export: bool
    line_start: int
    line_end: int
    module_path: str
    module_type: str
    file_id: int = 0
    symbol_id: int = 0


@dataclass
class FunctionContext:
    """Call graph context for a function."""

    function: FunctionInfo
    calls: list[str]        # names of functions this one calls
    called_by: list[str]    # names of functions that call this one


@dataclass
class MetadataInfo:
    """Metadata search result."""

    name: str
    object_type: str
    synonym: str
    full_name: str           # e.g. "Документ.ЛизинговыйДоговор"
    object_id: int = 0


@dataclass
class ObjectDetails:
    """Full details of a metadata object including attributes and tabular parts."""

    name: str
    object_type: str
    synonym: str
    full_name: str
    attributes: list[Attribute] = field(default_factory=list)
    tab_parts: list[TabPart] = field(default_factory=list)
    registers: list[str] = field(default_factory=list)


@dataclass
class ReferenceInfo:
    """An object that references a given metadata object via an attribute type."""

    referencing_object: str     # full_name of the object that holds the reference
    attribute_name: str         # the attribute whose type contains the reference
    attribute_type: str         # the full type_ref value


@dataclass
class IndexStats:
    """Statistics about the SQLite structural index."""

    files: int = 0
    symbols: int = 0
    objects: int = 0
    attributes: int = 0
