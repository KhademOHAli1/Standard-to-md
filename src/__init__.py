"""
PDF to LLM - Transform PDFs into structured data for Large Language Models

A toolkit for extracting, processing, and structuring PDF content
into formats optimized for LLM consumption.
"""

__version__ = "0.1.0"
__author__ = "PDF-to-LLM Contributors"

from .extractor import PDFExtractor
from .processor import TextProcessor
from .formatter import LLMFormatter
from .pipeline import ExtractionPipeline

__all__ = [
    "PDFExtractor",
    "TextProcessor", 
    "LLMFormatter",
    "ExtractionPipeline",
]
