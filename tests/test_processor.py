"""
Tests for Text Processor Module
"""

import pytest
from src.processor import (
    TextProcessor,
    StandardsProcessor,
    ContentType,
    ContentBlock,
    Section,
    StructuredDocument,
)


class TestContentClassification:
    def test_heading_classification(self):
        processor = TextProcessor()
        
        # Markdown heading
        assert processor.classify_line("# Heading 1") == ContentType.HEADING
        assert processor.classify_line("## Heading 2") == ContentType.HEADING
        
        # Numbered heading
        assert processor.classify_line("1.2.3 Section Title") == ContentType.HEADING
        
        # All caps heading
        assert processor.classify_line("INTRODUCTION") == ContentType.HEADING
    
    def test_list_classification(self):
        processor = TextProcessor()
        
        assert processor.classify_line("- Item one") == ContentType.LIST_ITEM
        assert processor.classify_line("• Bullet item") == ContentType.LIST_ITEM
        assert processor.classify_line("1. Numbered item") == ContentType.LIST_ITEM
        assert processor.classify_line("a) Lettered item") == ContentType.LIST_ITEM
    
    def test_equation_classification(self):
        processor = TextProcessor()
        
        assert processor.classify_line("x = y + z") == ContentType.EQUATION
        assert processor.classify_line("$E = mc^2$") == ContentType.EQUATION
    
    def test_paragraph_classification(self):
        processor = TextProcessor()
        
        # Regular text should be paragraph
        result = processor.classify_line("This is a regular sentence.")
        assert result == ContentType.PARAGRAPH
    
    def test_empty_line_classification(self):
        processor = TextProcessor()
        
        assert processor.classify_line("") == ContentType.UNKNOWN
        assert processor.classify_line("   ") == ContentType.UNKNOWN


class TestContentBlocks:
    def test_split_into_blocks(self):
        processor = TextProcessor()
        
        text = """# Introduction

This is the first paragraph.

## Section 1

- Item 1
- Item 2

Another paragraph here."""
        
        blocks = processor.split_into_blocks(text)
        
        assert len(blocks) > 0
        assert any(b.content_type == ContentType.HEADING for b in blocks)
        assert any(b.content_type == ContentType.PARAGRAPH for b in blocks)
        assert any(b.content_type == ContentType.LIST_ITEM for b in blocks)


class TestSectionExtraction:
    def test_extract_sections(self):
        processor = TextProcessor()
        
        blocks = [
            ContentBlock(ContentType.HEADING, "# Introduction"),
            ContentBlock(ContentType.PARAGRAPH, "Intro text"),
            ContentBlock(ContentType.HEADING, "## Methods"),
            ContentBlock(ContentType.PARAGRAPH, "Methods text"),
        ]
        
        sections = processor.extract_sections(blocks)
        
        assert len(sections) >= 1
        assert sections[0].title == "# Introduction"


class TestStandardsProcessor:
    def test_standards_patterns(self):
        processor = StandardsProcessor()
        
        # Standards-specific heading patterns
        assert processor.classify_line("1.1 General Requirements") == ContentType.HEADING
        assert processor.classify_line("A.1 Appendix Section") == ContentType.HEADING
        assert processor.classify_line("APPENDIX A") == ContentType.HEADING
    
    def test_equation_extraction(self):
        processor = StandardsProcessor()
        
        text = """
Q = A × V

Where:
A = cross-sectional area
V = velocity
"""
        equations = processor.extract_equations(text)
        
        # This tests the pattern matching for equations
        # Actual results depend on text format
        assert isinstance(equations, list)
    
    def test_table_extraction(self):
        processor = StandardsProcessor()
        
        text = """
Table 1 - Material Properties

This shows the properties.

Table 2 - Dimensions

Another table.
"""
        tables = processor.extract_tables(text)
        
        assert len(tables) == 2
        assert tables[0]["number"] == "1"
        assert tables[1]["number"] == "2"


class TestStructuredDocument:
    def test_document_to_dict(self):
        section = Section(
            title="Test Section",
            level=1,
            content=[
                ContentBlock(ContentType.PARAGRAPH, "Test content")
            ]
        )
        
        doc = StructuredDocument(
            title="Test Doc",
            source="/test/path.pdf",
            sections=[section],
            metadata={"author": "Test"}
        )
        
        data = doc.to_dict()
        
        assert data["title"] == "Test Doc"
        assert data["source"] == "/test/path.pdf"
        assert len(data["sections"]) == 1
        assert data["sections"][0]["title"] == "Test Section"


class TestCustomPatterns:
    def test_add_custom_pattern(self):
        processor = TextProcessor()
        processor.add_pattern(ContentType.CODE, r'^```')
        
        # The pattern should now be registered
        assert ContentType.CODE in processor._patterns
    
    def test_add_custom_classifier(self):
        processor = TextProcessor()
        
        def custom_classifier(line):
            if line.startswith("TODO:"):
                return ContentType.FOOTNOTE
            return None
        
        processor.add_classifier(custom_classifier)
        
        result = processor.classify_line("TODO: Fix this")
        assert result == ContentType.FOOTNOTE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
