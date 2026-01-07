"""
PDF Extraction Module

Provides multiple backends for extracting text from PDF documents:
- Direct text extraction (fast, works for text-based PDFs)
- OCR extraction (slower, works for scanned documents)
- Hybrid extraction (combines both methods)
"""

import io
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, List, Callable

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


@dataclass
class PageContent:
    """Represents extracted content from a single PDF page."""
    page_number: int
    text: str
    images: List[bytes] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass  
class PDFDocument:
    """Represents an extracted PDF document."""
    source_path: str
    title: str
    pages: List[PageContent]
    metadata: dict = field(default_factory=dict)
    
    @property
    def total_pages(self) -> int:
        return len(self.pages)
    
    @property
    def full_text(self) -> str:
        """Get concatenated text from all pages."""
        return "\n\n".join(page.text for page in self.pages if page.text)


class ExtractionBackend(ABC):
    """Abstract base class for PDF extraction backends."""
    
    @abstractmethod
    def extract(self, pdf_path: Path) -> PDFDocument:
        """Extract content from a PDF file."""
        pass
    
    @abstractmethod
    def extract_page(self, pdf_path: Path, page_num: int) -> PageContent:
        """Extract content from a specific page."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available."""
        pass


class DirectTextBackend(ExtractionBackend):
    """
    Direct text extraction using PyMuPDF.
    Fast and efficient for text-based PDFs.
    """
    
    def is_available(self) -> bool:
        return PYMUPDF_AVAILABLE
    
    def extract(self, pdf_path: Path) -> PDFDocument:
        if not self.is_available():
            raise RuntimeError("PyMuPDF is not installed. Run: pip install pymupdf")
        
        doc = fitz.open(str(pdf_path))
        pages = []
        
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                pages.append(PageContent(
                    page_number=page_num + 1,
                    text=text,
                    metadata={
                        "width": page.rect.width,
                        "height": page.rect.height,
                    }
                ))
            
            metadata = dict(doc.metadata) if doc.metadata else {}
            title = metadata.get("title", pdf_path.stem)
            
            return PDFDocument(
                source_path=str(pdf_path),
                title=title,
                pages=pages,
                metadata=metadata
            )
        finally:
            doc.close()
    
    def extract_page(self, pdf_path: Path, page_num: int) -> PageContent:
        if not self.is_available():
            raise RuntimeError("PyMuPDF is not installed. Run: pip install pymupdf")
            
        doc = fitz.open(str(pdf_path))
        try:
            page = doc[page_num]
            return PageContent(
                page_number=page_num + 1,
                text=page.get_text(),
                metadata={
                    "width": page.rect.width,
                    "height": page.rect.height,
                }
            )
        finally:
            doc.close()


class OCRBackend(ExtractionBackend):
    """
    OCR-based extraction using Tesseract.
    Works for scanned documents and images.
    """
    
    def __init__(self, dpi: int = 300, language: str = "eng"):
        self.dpi = dpi
        self.language = language
        self._zoom = dpi / 72  # PDF default is 72 DPI
    
    def is_available(self) -> bool:
        return PYMUPDF_AVAILABLE and OCR_AVAILABLE
    
    def extract(self, pdf_path: Path) -> PDFDocument:
        if not self.is_available():
            missing = []
            if not PYMUPDF_AVAILABLE:
                missing.append("pymupdf")
            if not OCR_AVAILABLE:
                missing.append("pillow pytesseract")
            raise RuntimeError(f"Missing dependencies. Run: pip install {' '.join(missing)}")
        
        doc = fitz.open(str(pdf_path))
        pages = []
        
        try:
            for page_num in range(len(doc)):
                page_content = self._ocr_page(doc, page_num)
                pages.append(page_content)
            
            metadata = dict(doc.metadata) if doc.metadata else {}
            title = metadata.get("title", pdf_path.stem)
            
            return PDFDocument(
                source_path=str(pdf_path),
                title=title,
                pages=pages,
                metadata=metadata
            )
        finally:
            doc.close()
    
    def _ocr_page(self, doc, page_num: int) -> PageContent:
        """Perform OCR on a single page."""
        page = doc[page_num]
        mat = fitz.Matrix(self._zoom, self._zoom)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        
        text = pytesseract.image_to_string(img, lang=self.language)
        
        return PageContent(
            page_number=page_num + 1,
            text=text,
            metadata={
                "extraction_method": "ocr",
                "dpi": self.dpi,
                "language": self.language,
            }
        )
    
    def extract_page(self, pdf_path: Path, page_num: int) -> PageContent:
        if not self.is_available():
            raise RuntimeError("OCR dependencies not available")
            
        doc = fitz.open(str(pdf_path))
        try:
            return self._ocr_page(doc, page_num)
        finally:
            doc.close()


class HybridBackend(ExtractionBackend):
    """
    Hybrid extraction that uses direct text extraction first,
    then falls back to OCR for pages with insufficient text.
    """
    
    def __init__(
        self, 
        min_chars_per_page: int = 100,
        ocr_dpi: int = 300,
        ocr_language: str = "eng"
    ):
        self.min_chars = min_chars_per_page
        self.direct_backend = DirectTextBackend()
        self.ocr_backend = OCRBackend(dpi=ocr_dpi, language=ocr_language)
    
    def is_available(self) -> bool:
        return self.direct_backend.is_available()
    
    def extract(self, pdf_path: Path) -> PDFDocument:
        # First try direct extraction
        doc = self.direct_backend.extract(pdf_path)
        
        # Check each page and use OCR if needed
        if self.ocr_backend.is_available():
            for i, page in enumerate(doc.pages):
                if len(page.text.strip()) < self.min_chars:
                    # Page has little text, try OCR
                    ocr_page = self.ocr_backend.extract_page(pdf_path, i)
                    if len(ocr_page.text.strip()) > len(page.text.strip()):
                        doc.pages[i] = ocr_page
                        doc.pages[i].metadata["extraction_method"] = "ocr_fallback"
        
        return doc
    
    def extract_page(self, pdf_path: Path, page_num: int) -> PageContent:
        page = self.direct_backend.extract_page(pdf_path, page_num)
        
        if len(page.text.strip()) < self.min_chars and self.ocr_backend.is_available():
            ocr_page = self.ocr_backend.extract_page(pdf_path, page_num)
            if len(ocr_page.text.strip()) > len(page.text.strip()):
                return ocr_page
        
        return page


class PDFExtractor:
    """
    Main PDF extraction interface.
    
    Usage:
        extractor = PDFExtractor(backend="hybrid")
        document = extractor.extract("document.pdf")
        
        # With text filters
        extractor = PDFExtractor(
            backend="ocr",
            filters=[remove_headers, remove_footers]
        )
    """
    
    BACKENDS = {
        "direct": DirectTextBackend,
        "ocr": OCRBackend,
        "hybrid": HybridBackend,
    }
    
    def __init__(
        self,
        backend: str = "hybrid",
        filters: Optional[List[Callable[[str], str]]] = None,
        **backend_kwargs
    ):
        if backend not in self.BACKENDS:
            raise ValueError(f"Unknown backend: {backend}. Available: {list(self.BACKENDS.keys())}")
        
        self._backend = self.BACKENDS[backend](**backend_kwargs)
        self._filters = filters or []
        
        if not self._backend.is_available():
            raise RuntimeError(f"Backend '{backend}' is not available. Check dependencies.")
    
    def add_filter(self, filter_func: Callable[[str], str]):
        """Add a text filter to be applied during extraction."""
        self._filters.append(filter_func)
    
    def extract(self, pdf_path: str | Path) -> PDFDocument:
        """Extract content from a PDF file."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        
        document = self._backend.extract(path)
        
        # Apply filters to each page
        for page in document.pages:
            page.text = self._apply_filters(page.text)
        
        return document
    
    def extract_pages(
        self, 
        pdf_path: str | Path, 
        page_numbers: List[int]
    ) -> List[PageContent]:
        """Extract specific pages from a PDF."""
        path = Path(pdf_path)
        pages = []
        
        for page_num in page_numbers:
            page = self._backend.extract_page(path, page_num - 1)  # Convert to 0-indexed
            page.text = self._apply_filters(page.text)
            pages.append(page)
        
        return pages
    
    def _apply_filters(self, text: str) -> str:
        """Apply all registered filters to text."""
        for filter_func in self._filters:
            text = filter_func(text)
        return text
    
    def iter_pages(self, pdf_path: str | Path) -> Iterator[PageContent]:
        """Iterate over pages one at a time (memory efficient)."""
        document = self.extract(pdf_path)
        yield from document.pages


# Common text filters
def create_copyright_filter(patterns: Optional[List[str]] = None) -> Callable[[str], str]:
    """
    Create a filter that removes copyright notices.
    
    Args:
        patterns: List of patterns to filter. If None, uses common patterns.
    """
    if patterns is None:
        patterns = [
            "COPYRIGHT",
            "LICENSED BY",
            "ALL RIGHTS RESERVED",
            "AMERICAN PETROLEUM INSTITUTE",
            "INFORMATION HANDLING SERVICES",
        ]
    
    def filter_func(text: str) -> str:
        lines = text.split('\n')
        filtered = [
            line for line in lines
            if not any(p.upper() in line.upper() for p in patterns)
        ]
        return '\n'.join(filtered)
    
    return filter_func


def create_line_filter(
    min_length: int = 0,
    max_length: Optional[int] = None
) -> Callable[[str], str]:
    """Create a filter that removes lines based on length."""
    def filter_func(text: str) -> str:
        lines = text.split('\n')
        filtered = []
        for line in lines:
            if len(line.strip()) >= min_length:
                if max_length is None or len(line.strip()) <= max_length:
                    filtered.append(line)
        return '\n'.join(filtered)
    
    return filter_func


def normalize_whitespace(text: str) -> str:
    """Normalize excessive whitespace in text."""
    import re
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    # Replace multiple newlines with double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
