"""Parsers for 1C files."""

from .metadata import MetadataParser
from .code import CodeParser
from .help import HelpParser
from .metadata_xml import MetadataXMLParser

__all__ = ["MetadataParser", "CodeParser", "HelpParser", "MetadataXMLParser"]
