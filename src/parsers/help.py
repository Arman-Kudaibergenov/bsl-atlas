"""Parser for 1C HTML help files.

Converts HTML documentation to Markdown for better indexing.
"""

import logging
import re
from pathlib import Path
from typing import Any

import chardet
from bs4 import BeautifulSoup
from markdownify import markdownify

logger = logging.getLogger(__name__)


class HelpParser:
    """Parser for 1C HTML help documentation files."""

    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding using chardet."""
        with open(file_path, "rb") as f:
            raw_data = f.read(10000)
            result = chardet.detect(raw_data)
            return result.get("encoding", "utf-8") or "utf-8"

    def _read_file(self, file_path: Path) -> str:
        """Read file with automatic encoding detection."""
        encodings = ["utf-8", "utf-8-sig", "cp1251", "windows-1251", "utf-16"]

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

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title from HTML."""
        # Try title tag
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            return title_tag.string.strip()

        # Try h1 tag
        h1_tag = soup.find("h1")
        if h1_tag:
            return h1_tag.get_text(strip=True)

        return "Без названия"

    def _clean_html(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Remove unwanted elements from HTML."""
        # Remove script and style tags
        for tag in soup.find_all(["script", "style", "noscript"]):
            tag.decompose()

        # Remove navigation elements
        for tag in soup.find_all(class_=re.compile(r"nav|menu|sidebar|footer|header")):
            tag.decompose()

        return soup

    def _html_to_markdown(self, html_content: str) -> str:
        """Convert HTML to Markdown."""
        try:
            soup = BeautifulSoup(html_content, "lxml")
            soup = self._clean_html(soup)

            # Get main content
            main_content = soup.find("main") or soup.find("article") or soup.find("body") or soup
            html_str = str(main_content)

            # Convert to markdown
            markdown = markdownify(
                html_str,
                heading_style="ATX",
                bullets="-",
                strip=["a"],  # Remove links but keep text
            )

            # Clean up extra whitespace
            markdown = re.sub(r"\n{3,}", "\n\n", markdown)
            markdown = markdown.strip()

            return markdown
        except Exception as e:
            logger.warning(f"Error converting HTML to Markdown: {e}")
            return BeautifulSoup(html_content, "lxml").get_text(separator="\n", strip=True)

    def _extract_path_from_filename(self, file_path: Path) -> str:
        """Extract 1C object path from help file path.

        Example: Help/Справочники/Контрагенты/index.html
        -> Справочники.Контрагенты.Справка
        """
        parts = file_path.parts

        # Skip common prefixes
        skip_parts = {"Help", "help", "documentation", "docs"}
        relevant_parts = [p for p in parts[:-1] if p not in skip_parts]

        if not relevant_parts:
            return f"Справка.{file_path.stem}"

        # Remove file extension from last part
        return ".".join(relevant_parts) + ".Справка"

    def parse_file(self, file_path: str | Path) -> list[dict[str, Any]]:
        """Parse an HTML help file.

        Args:
            file_path: Path to the HTML file

        Returns:
            List containing parsed help document
        """
        file_path = Path(file_path)
        html_content = self._read_file(file_path)

        if not html_content:
            logger.warning(f"Empty or unreadable file: {file_path}")
            return []

        try:
            soup = BeautifulSoup(html_content, "lxml")
        except Exception as e:
            logger.error(f"Failed to parse HTML {file_path}: {e}")
            return []

        title = self._extract_title(soup)
        markdown_content = self._html_to_markdown(html_content)
        object_path = self._extract_path_from_filename(file_path)

        result = {
            "full_path": object_path,
            "object_type": "Справка",
            "name": title,
            "source_file": str(file_path),
            "content": markdown_content,
            "title": title,
        }

        logger.info(f"Parsed help file {file_path}: {title}")
        return [result]

    def parse_directory(
        self,
        directory: str | Path,
        extensions: tuple[str, ...] = (".html", ".htm"),
    ) -> list[dict[str, Any]]:
        """Parse all HTML help files in a directory recursively.

        Args:
            directory: Root directory to scan
            extensions: File extensions to process

        Returns:
            List of parsed help documents
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

        logger.info(f"Parsed {len(results)} help files from {directory}")
        return results
