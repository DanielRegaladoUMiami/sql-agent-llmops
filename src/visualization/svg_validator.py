"""
SVG validation using lxml for correctness checking.

Validates SVG structure and provides detailed error reporting
for malformed SVG documents.
"""

import logging
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)


class SVGValidator:
    """Validate SVG documents for correctness."""

    def __init__(self) -> None:
        """Initialize SVG validator."""
        self.has_lxml = self._check_lxml()

    def _check_lxml(self) -> bool:
        """Check if lxml is available."""
        try:
            from lxml import etree
            return True
        except ImportError:
            logger.warning("lxml not available for SVG validation")
            return False

    def validate(self, svg_content: str) -> Tuple[bool, Optional[str]]:
        """
        Validate SVG content.

        Args:
            svg_content: SVG document as string

        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if SVG is valid
            - error_message: Error description if invalid, None if valid
        """
        if not svg_content or not svg_content.strip():
            return False, "SVG content is empty"

        if not self.has_lxml:
            # Fallback: basic validation
            return self._basic_validation(svg_content)

        try:
            from lxml import etree

            # Parse SVG
            parser = etree.XMLParser(remove_blank_text=True)
            tree = etree.fromstring(svg_content.encode("utf-8"), parser=parser)

            # Check root element
            if tree.tag != "{http://www.w3.org/2000/svg}svg" and tree.tag != "svg":
                return False, "Root element is not SVG"

            logger.info("SVG validation passed")
            return True, None

        except etree.XMLSyntaxError as e:
            error_msg = f"XML syntax error: {e.msg}"
            logger.warning(f"SVG validation failed: {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"SVG validation error: {str(e)}"
            logger.warning(error_msg)
            return False, error_msg

    def _basic_validation(self, svg_content: str) -> Tuple[bool, Optional[str]]:
        """Fallback basic validation without lxml."""
        content = svg_content.strip()

        # Check for SVG tag
        if not content.startswith("<?xml") and not "<svg" in content:
            return False, "Missing SVG tag"

        # Check for closing tag
        if not content.endswith("</svg>"):
            return False, "Missing closing SVG tag"

        # Check for balanced tags (simple check)
        if content.count("<") != content.count(">"):
            return False, "Unbalanced XML tags"

        return True, None

    def get_info(self, svg_content: str) -> dict:
        """
        Extract basic information from SVG.

        Args:
            svg_content: SVG document as string

        Returns:
            Dictionary with SVG metadata
        """
        if not self.has_lxml:
            return self._get_info_basic(svg_content)

        try:
            from lxml import etree

            tree = etree.fromstring(svg_content.encode("utf-8"))

            info = {
                "width": tree.get("width", "unknown"),
                "height": tree.get("height", "unknown"),
                "viewBox": tree.get("viewBox", "unknown"),
                "element_count": len(tree),
            }

            return info

        except Exception as e:
            logger.warning(f"Error extracting SVG info: {e}")
            return {}

    def _get_info_basic(self, svg_content: str) -> dict:
        """Basic info extraction without lxml."""
        info = {}

        # Extract dimensions
        import re

        width_match = re.search(r'width="([^"]+)"', svg_content)
        height_match = re.search(r'height="([^"]+)"', svg_content)
        viewbox_match = re.search(r'viewBox="([^"]+)"', svg_content)

        if width_match:
            info["width"] = width_match.group(1)
        if height_match:
            info["height"] = height_match.group(1)
        if viewbox_match:
            info["viewBox"] = viewbox_match.group(1)

        info["element_count"] = len(re.findall(r"<\w+[^>]*>", svg_content))

        return info
