"""
Text Processing Module

Provides tools for cleaning, structuring, and enhancing extracted text
before formatting for LLM consumption.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple
from enum import Enum


class ContentType(Enum):
    """Types of content that can be identified in documents."""
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    EQUATION = "equation"
    CODE = "code"
    FIGURE_CAPTION = "figure_caption"
    FOOTNOTE = "footnote"
    PAGE_NUMBER = "page_number"
    HEADER = "header"
    FOOTER = "footer"
    UNKNOWN = "unknown"


@dataclass
class ContentBlock:
    """Represents a classified block of content."""
    content_type: ContentType
    text: str
    level: int = 0  # For headings
    metadata: dict = field(default_factory=dict)


@dataclass
class Section:
    """Represents a document section with heading and content."""
    title: str
    level: int
    content: List[ContentBlock]
    subsections: List['Section'] = field(default_factory=list)
    page_start: int = 0
    page_end: int = 0


@dataclass
class StructuredDocument:
    """A fully processed and structured document."""
    title: str
    source: str
    sections: List[Section]
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "title": self.title,
            "source": self.source,
            "metadata": self.metadata,
            "sections": [self._section_to_dict(s) for s in self.sections]
        }
    
    def _section_to_dict(self, section: Section) -> dict:
        return {
            "title": section.title,
            "level": section.level,
            "content": [
                {"type": b.content_type.value, "text": b.text}
                for b in section.content
            ],
            "subsections": [self._section_to_dict(s) for s in section.subsections]
        }


class TextProcessor:
    """
    Process and structure extracted text content.
    
    Usage:
        processor = TextProcessor()
        processor.add_pattern("equation", r'\$.*?\$')
        structured = processor.process(document)
    """
    
    # Default patterns for content classification
    DEFAULT_PATTERNS = {
        ContentType.HEADING: [
            r'^#{1,6}\s+.+$',  # Markdown headings
            r'^[A-Z][A-Z0-9\s\-\.]+$',  # ALL CAPS headings
            r'^\d+\.[\d\.]*\s+[A-Z].+$',  # Numbered headings (1.2.3 Title)
            r'^(?:Section|Chapter|Part)\s+\d+',  # Section/Chapter headers
        ],
        ContentType.LIST_ITEM: [
            r'^\s*[-•●○◦]\s+.+$',  # Bullet points
            r'^\s*\d+[.)]\s+.+$',  # Numbered lists
            r'^\s*[a-z][.)]\s+.+$',  # Lettered lists
        ],
        ContentType.EQUATION: [
            r'^\s*[A-Za-z]+\s*=\s*.+$',  # Simple equations
            r'\$\$.+?\$\$',  # LaTeX display math
            r'\$.+?\$',  # LaTeX inline math
        ],
        ContentType.TABLE: [
            r'^\s*\|.+\|.+\|',  # Markdown tables
            r'^\s*[-+]+$',  # Table borders
        ],
        ContentType.PAGE_NUMBER: [
            r'^\s*\d+\s*$',  # Standalone numbers
            r'^\s*Page\s+\d+',  # "Page X"
        ],
        ContentType.FOOTNOTE: [
            r'^\s*\[\d+\]',  # [1] style
            r'^\s*\d+\s+[A-Z]',  # Numbered footnotes
        ],
    }
    
    def __init__(self):
        self._patterns: Dict[ContentType, List[re.Pattern]] = {}
        self._custom_classifiers: List[Callable[[str], Optional[ContentType]]] = []
        
        # Compile default patterns
        for content_type, patterns in self.DEFAULT_PATTERNS.items():
            self._patterns[content_type] = [re.compile(p, re.MULTILINE) for p in patterns]
    
    def add_pattern(self, content_type: ContentType, pattern: str):
        """Add a custom pattern for content classification."""
        if content_type not in self._patterns:
            self._patterns[content_type] = []
        self._patterns[content_type].append(re.compile(pattern, re.MULTILINE))
    
    def add_classifier(self, classifier: Callable[[str], Optional[ContentType]]):
        """Add a custom classifier function."""
        self._custom_classifiers.append(classifier)
    
    def classify_line(self, line: str) -> ContentType:
        """Classify a single line of text."""
        # Try custom classifiers first
        for classifier in self._custom_classifiers:
            result = classifier(line)
            if result is not None:
                return result
        
        # Try pattern matching
        for content_type, patterns in self._patterns.items():
            for pattern in patterns:
                if pattern.match(line.strip()):
                    return content_type
        
        # Default to paragraph if line has content
        if line.strip():
            return ContentType.PARAGRAPH
        
        return ContentType.UNKNOWN
    
    def split_into_blocks(self, text: str) -> List[ContentBlock]:
        """Split text into classified content blocks."""
        blocks = []
        current_lines = []
        current_type = None
        
        for line in text.split('\n'):
            line_type = self.classify_line(line)
            
            # Skip unknown/empty lines
            if line_type == ContentType.UNKNOWN:
                if current_lines and current_type:
                    blocks.append(ContentBlock(
                        content_type=current_type,
                        text='\n'.join(current_lines)
                    ))
                    current_lines = []
                    current_type = None
                continue
            
            # Start new block if type changes
            if line_type != current_type and current_lines:
                blocks.append(ContentBlock(
                    content_type=current_type,
                    text='\n'.join(current_lines)
                ))
                current_lines = []
            
            current_lines.append(line)
            current_type = line_type
        
        # Don't forget the last block
        if current_lines and current_type:
            blocks.append(ContentBlock(
                content_type=current_type,
                text='\n'.join(current_lines)
            ))
        
        return blocks
    
    def extract_sections(self, blocks: List[ContentBlock]) -> List[Section]:
        """Extract hierarchical sections from content blocks."""
        sections = []
        current_section = None
        current_content = []
        
        for block in blocks:
            if block.content_type == ContentType.HEADING:
                # Save previous section
                if current_section:
                    current_section.content = current_content
                    sections.append(current_section)
                
                # Determine heading level
                level = self._determine_heading_level(block.text)
                
                current_section = Section(
                    title=block.text.strip(),
                    level=level,
                    content=[]
                )
                current_content = []
            else:
                current_content.append(block)
        
        # Don't forget the last section
        if current_section:
            current_section.content = current_content
            sections.append(current_section)
        elif current_content:
            # Content before first heading
            sections.insert(0, Section(
                title="Introduction",
                level=1,
                content=current_content
            ))
        
        return self._build_hierarchy(sections)
    
    def _determine_heading_level(self, text: str) -> int:
        """Determine the hierarchical level of a heading."""
        text = text.strip()
        
        # Markdown style
        if text.startswith('#'):
            return len(text) - len(text.lstrip('#'))
        
        # Numbered style (1.2.3)
        match = re.match(r'^(\d+\.)+', text)
        if match:
            return match.group().count('.')
        
        # Chapter/Section style
        if re.match(r'^Chapter\s+', text, re.IGNORECASE):
            return 1
        if re.match(r'^Section\s+', text, re.IGNORECASE):
            return 2
        
        # Default based on formatting
        if text.isupper():
            return 1
        
        return 2
    
    def _build_hierarchy(self, sections: List[Section]) -> List[Section]:
        """Build hierarchical section structure."""
        if not sections:
            return []
        
        root_sections = []
        stack: List[Section] = []
        
        for section in sections:
            # Pop sections from stack that are at same or lower level
            while stack and stack[-1].level >= section.level:
                stack.pop()
            
            if stack:
                # Add as subsection
                stack[-1].subsections.append(section)
            else:
                # Add as root section
                root_sections.append(section)
            
            stack.append(section)
        
        return root_sections
    
    def process(self, document) -> StructuredDocument:
        """
        Process a PDFDocument into a StructuredDocument.
        
        Args:
            document: PDFDocument from extractor
            
        Returns:
            StructuredDocument with classified and structured content
        """
        # Combine all pages
        full_text = document.full_text
        
        # Split into blocks
        blocks = self.split_into_blocks(full_text)
        
        # Extract sections
        sections = self.extract_sections(blocks)
        
        return StructuredDocument(
            title=document.title,
            source=document.source_path,
            sections=sections,
            metadata=document.metadata
        )


class StandardsProcessor(TextProcessor):
    """
    Specialized processor for technical standards documents.
    Handles specific patterns common in API, AGA, and similar standards.
    """
    
    STANDARDS_PATTERNS = {
        ContentType.HEADING: [
            r'^\d+\.\d+[\.\d]*\s+[A-Z]',  # 1.2.3 Title format
            r'^[A-Z]\.\d+',  # A.1 Appendix format
            r'^APPENDIX\s+[A-Z]',
            r'^ANNEX\s+[A-Z]',
        ],
        ContentType.EQUATION: [
            r'^Where:?\s*$',  # Equation variable definitions
            r'^\s*[A-Za-z_]+\s*=\s*.+',  # Variable assignments
        ],
        ContentType.TABLE: [
            r'^Table\s+\d+',
            r'^TABLE\s+\d+',
        ],
        ContentType.FIGURE_CAPTION: [
            r'^Figure\s+\d+',
            r'^FIGURE\s+\d+',
            r'^Fig\.\s+\d+',
        ],
    }
    
    def __init__(self):
        super().__init__()
        
        # Add standards-specific patterns
        for content_type, patterns in self.STANDARDS_PATTERNS.items():
            for pattern in patterns:
                self.add_pattern(content_type, pattern)
        
        # Add custom classifier for standards
        self.add_classifier(self._classify_standards_content)
    
    def _classify_standards_content(self, line: str) -> Optional[ContentType]:
        """Custom classifier for standards-specific content."""
        line = line.strip()
        
        # Notes and warnings
        if re.match(r'^NOTE\s*\d*:', line, re.IGNORECASE):
            return ContentType.PARAGRAPH
        if re.match(r'^WARNING:', line, re.IGNORECASE):
            return ContentType.PARAGRAPH
        if re.match(r'^CAUTION:', line, re.IGNORECASE):
            return ContentType.PARAGRAPH
        
        # References
        if re.match(r'^\[\d+\]\s+', line):
            return ContentType.FOOTNOTE
        
        return None
    
    def extract_equations(self, text: str) -> List[Dict]:
        """Extract equations with their variable definitions."""
        equations = []
        
        # Pattern for equation followed by "Where:" definitions
        equation_pattern = re.compile(
            r'([A-Za-z_]+)\s*=\s*([^\n]+)\n+Where:?\n((?:\s*[A-Za-z_]+\s*=.+\n?)+)',
            re.MULTILINE
        )
        
        for match in equation_pattern.finditer(text):
            var_name = match.group(1)
            formula = match.group(2).strip()
            definitions_text = match.group(3)
            
            # Parse variable definitions
            definitions = {}
            for line in definitions_text.strip().split('\n'):
                def_match = re.match(r'\s*([A-Za-z_]+)\s*=\s*(.+)', line)
                if def_match:
                    definitions[def_match.group(1)] = def_match.group(2).strip()
            
            equations.append({
                "variable": var_name,
                "formula": formula,
                "definitions": definitions,
                "raw": match.group(0)
            })
        
        return equations
    
    def extract_tables(self, text: str) -> List[Dict]:
        """Extract table references and their titles."""
        tables = []
        
        table_pattern = re.compile(
            r'Table\s+(\d+[\.\d]*)\s*[-–—]?\s*(.+?)(?=\n\n|\Z)',
            re.IGNORECASE | re.DOTALL
        )
        
        for match in table_pattern.finditer(text):
            tables.append({
                "number": match.group(1),
                "title": match.group(2).strip(),
                "position": match.start()
            })
        
        return tables
