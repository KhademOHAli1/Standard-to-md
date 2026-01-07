"""
Tests for LLM Formatter Module
"""

import pytest
import json
from src.formatter import (
    LLMFormatter,
    MarkdownFormat,
    JSONFormat,
    InstructionFormat,
    create_llm_context,
)
from src.processor import (
    StructuredDocument,
    Section,
    ContentBlock,
    ContentType,
)


@pytest.fixture
def sample_document():
    """Create a sample structured document for testing."""
    sections = [
        Section(
            title="Introduction",
            level=1,
            content=[
                ContentBlock(ContentType.PARAGRAPH, "This is the introduction."),
                ContentBlock(ContentType.LIST_ITEM, "- Point 1\n- Point 2"),
            ],
            subsections=[
                Section(
                    title="Background",
                    level=2,
                    content=[
                        ContentBlock(ContentType.PARAGRAPH, "Background info here.")
                    ],
                    subsections=[]
                )
            ]
        ),
        Section(
            title="Calculations",
            level=1,
            content=[
                ContentBlock(ContentType.EQUATION, "Q = A * V"),
                ContentBlock(ContentType.PARAGRAPH, "Where Q is flow rate."),
            ],
            subsections=[]
        ),
    ]
    
    return StructuredDocument(
        title="Test Standard",
        source="/test/standard.pdf",
        sections=sections,
        metadata={"version": "1.0", "year": "2024"}
    )


class TestMarkdownFormat:
    def test_basic_formatting(self, sample_document):
        formatter = MarkdownFormat()
        output = formatter.format(sample_document)
        
        assert "# Test Standard" in output
        assert "Introduction" in output
        assert "Calculations" in output
    
    def test_with_toc(self, sample_document):
        formatter = MarkdownFormat(include_toc=True)
        output = formatter.format(sample_document)
        
        assert "Table of Contents" in output
        assert "[Introduction]" in output
    
    def test_without_toc(self, sample_document):
        formatter = MarkdownFormat(include_toc=False)
        output = formatter.format(sample_document)
        
        assert "Table of Contents" not in output
    
    def test_with_metadata(self, sample_document):
        formatter = MarkdownFormat(include_metadata=True)
        output = formatter.format(sample_document)
        
        assert "---" in output  # YAML frontmatter
        assert "version: 1.0" in output
    
    def test_equation_formatting(self, sample_document):
        formatter = MarkdownFormat()
        output = formatter.format(sample_document)
        
        assert "Q = A * V" in output
    
    def test_extension(self):
        formatter = MarkdownFormat()
        assert formatter.get_extension() == ".md"


class TestJSONFormat:
    def test_valid_json_output(self, sample_document):
        formatter = JSONFormat()
        output = formatter.format(sample_document)
        
        # Should be valid JSON
        data = json.loads(output)
        
        assert data["title"] == "Test Standard"
        assert len(data["sections"]) == 2
    
    def test_structure(self, sample_document):
        formatter = JSONFormat()
        output = formatter.format(sample_document)
        data = json.loads(output)
        
        # Check structure
        assert "title" in data
        assert "source" in data
        assert "metadata" in data
        assert "sections" in data
        assert "extracted_at" in data
    
    def test_nested_sections(self, sample_document):
        formatter = JSONFormat()
        output = formatter.format(sample_document)
        data = json.loads(output)
        
        # First section should have subsections
        first_section = data["sections"][0]
        assert "subsections" in first_section
        assert len(first_section["subsections"]) == 1
    
    def test_extension(self):
        formatter = JSONFormat()
        assert formatter.get_extension() == ".json"


class TestInstructionFormat:
    def test_detailed_format(self, sample_document):
        formatter = InstructionFormat(instruction_style="detailed")
        output = formatter.format(sample_document)
        
        assert "Reference Guide" in output
        assert "Instructions" in output
    
    def test_concise_format(self, sample_document):
        formatter = InstructionFormat(instruction_style="concise")
        output = formatter.format(sample_document)
        
        assert "Quick Reference" in output
    
    def test_qa_format(self, sample_document):
        formatter = InstructionFormat(instruction_style="qa")
        output = formatter.format(sample_document)
        
        assert "Q&A Format" in output
        assert "Q:" in output
        assert "A:" in output


class TestLLMFormatter:
    def test_invalid_format(self):
        with pytest.raises(ValueError):
            LLMFormatter(format="invalid")
    
    def test_markdown_format(self, sample_document):
        formatter = LLMFormatter(format="markdown")
        output = formatter.format(sample_document)
        
        assert "# Test Standard" in output
    
    def test_json_format(self, sample_document):
        formatter = LLMFormatter(format="json")
        output = formatter.format(sample_document)
        
        # Should be valid JSON
        data = json.loads(output)
        assert data["title"] == "Test Standard"
    
    def test_save_to_file(self, sample_document, tmp_path):
        formatter = LLMFormatter(format="markdown")
        output_path = tmp_path / "test_output.md"
        
        formatter.save(sample_document, str(output_path))
        
        assert output_path.exists()
        content = output_path.read_text()
        assert "Test Standard" in content


class TestLLMContext:
    def test_basic_context(self, sample_document):
        context = create_llm_context(sample_document)
        
        assert "Test Standard" in context
        assert "Introduction" in context
    
    def test_max_tokens_truncation(self, sample_document):
        # Very small limit to force truncation
        context = create_llm_context(sample_document, max_tokens=50)
        
        # Should be truncated
        assert len(context) < 500
    
    def test_focus_sections(self, sample_document):
        context = create_llm_context(
            sample_document,
            focus_sections=["Calculations"]
        )
        
        assert "Calculations" in context


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
