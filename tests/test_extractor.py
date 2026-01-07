"""
Tests for PDF Extractor Module
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.extractor import (
    PDFExtractor,
    DirectTextBackend,
    OCRBackend,
    HybridBackend,
    PageContent,
    PDFDocument,
    create_copyright_filter,
    create_line_filter,
    normalize_whitespace,
)


class TestPageContent:
    def test_page_content_creation(self):
        page = PageContent(page_number=1, text="Hello World")
        assert page.page_number == 1
        assert page.text == "Hello World"
        assert page.images == []
        assert page.tables == []
        assert page.metadata == {}


class TestPDFDocument:
    def test_document_full_text(self):
        pages = [
            PageContent(page_number=1, text="Page one"),
            PageContent(page_number=2, text="Page two"),
            PageContent(page_number=3, text=""),  # Empty page
        ]
        doc = PDFDocument(
            source_path="/test/doc.pdf",
            title="Test Doc",
            pages=pages
        )
        
        assert doc.total_pages == 3
        assert "Page one" in doc.full_text
        assert "Page two" in doc.full_text


class TestFilters:
    def test_copyright_filter(self):
        filter_func = create_copyright_filter()
        text = "Line 1\nCOPYRIGHT 2024\nLine 3\nLicensed by Company\nLine 5"
        result = filter_func(text)
        
        assert "COPYRIGHT" not in result
        assert "Licensed by" not in result
        assert "Line 1" in result
        assert "Line 3" in result
        assert "Line 5" in result
    
    def test_copyright_filter_custom_patterns(self):
        filter_func = create_copyright_filter(["CONFIDENTIAL", "DRAFT"])
        text = "Content\nCONFIDENTIAL DOCUMENT\nMore content\nDRAFT VERSION"
        result = filter_func(text)
        
        assert "CONFIDENTIAL" not in result
        assert "DRAFT" not in result
        assert "Content" in result
    
    def test_line_filter_min_length(self):
        filter_func = create_line_filter(min_length=5)
        text = "OK\nThis is longer\nNo\nYes this works"
        result = filter_func(text)
        
        assert "OK" not in result
        assert "No" not in result
        assert "This is longer" in result
        assert "Yes this works" in result
    
    def test_normalize_whitespace(self):
        text = "Word1   Word2\n\n\n\nParagraph 2"
        result = normalize_whitespace(text)
        
        assert "Word1 Word2" in result
        assert "\n\n\n\n" not in result
        assert "\n\n" in result  # Double newline preserved


class TestPDFExtractor:
    def test_extractor_invalid_backend(self):
        with pytest.raises(ValueError):
            PDFExtractor(backend="invalid_backend")
    
    def test_extractor_with_filters(self):
        def custom_filter(text):
            return text.upper()
        
        # Mock the backend
        with patch.object(DirectTextBackend, 'is_available', return_value=True):
            extractor = PDFExtractor(
                backend="direct",
                filters=[custom_filter]
            )
            
            assert len(extractor._filters) == 1


class TestDirectTextBackend:
    def test_availability_check(self):
        backend = DirectTextBackend()
        # This will depend on whether pymupdf is installed
        # In test environment, it should be available
        assert isinstance(backend.is_available(), bool)


class TestHybridBackend:
    def test_initialization(self):
        backend = HybridBackend(min_chars_per_page=50, ocr_dpi=200)
        assert backend.min_chars == 50
        assert backend.direct_backend is not None


# Integration tests (require actual PDF file)
class TestIntegration:
    @pytest.mark.skip(reason="Requires actual PDF file")
    def test_extract_real_pdf(self):
        extractor = PDFExtractor(backend="direct")
        doc = extractor.extract("test_file.pdf")
        
        assert doc.total_pages > 0
        assert doc.full_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
