"""
Extraction Pipeline Module

Provides a complete pipeline for processing PDFs into LLM-ready formats.
Combines extraction, processing, and formatting into a single workflow.
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from .extractor import (
    PDFExtractor, 
    PDFDocument,
    create_copyright_filter,
    normalize_whitespace
)
from .processor import TextProcessor, StandardsProcessor, StructuredDocument
from .formatter import LLMFormatter, MarkdownFormat, JSONFormat, InstructionFormat


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the extraction pipeline."""
    
    # Extraction settings
    extraction_backend: str = "hybrid"
    ocr_dpi: int = 300
    ocr_language: str = "eng"
    
    # Processing settings
    processor_type: str = "standard"  # "basic" or "standard"
    
    # Output settings
    output_format: str = "markdown"
    output_dir: str = "./output"
    
    # Filter settings
    remove_copyright: bool = True
    copyright_patterns: List[str] = field(default_factory=lambda: [
        "COPYRIGHT",
        "LICENSED BY",
        "ALL RIGHTS RESERVED",
    ])
    normalize_whitespace: bool = True
    
    # Processing options
    max_workers: int = 4
    
    def to_dict(self) -> dict:
        return {
            "extraction_backend": self.extraction_backend,
            "ocr_dpi": self.ocr_dpi,
            "ocr_language": self.ocr_language,
            "processor_type": self.processor_type,
            "output_format": self.output_format,
            "output_dir": self.output_dir,
            "remove_copyright": self.remove_copyright,
            "normalize_whitespace": self.normalize_whitespace,
            "max_workers": self.max_workers,
        }


@dataclass
class ProcessingResult:
    """Result of processing a single PDF."""
    source_path: str
    output_path: Optional[str]
    success: bool
    error: Optional[str] = None
    pages_processed: int = 0
    document: Optional[StructuredDocument] = None


class ExtractionPipeline:
    """
    Complete pipeline for PDF to LLM-ready format conversion.
    
    Usage:
        # Simple usage
        pipeline = ExtractionPipeline()
        result = pipeline.process_file("document.pdf")
        
        # Batch processing
        pipeline = ExtractionPipeline(config=PipelineConfig(
            output_format="json",
            output_dir="./processed"
        ))
        results = pipeline.process_directory("./pdfs")
        
        # Custom pipeline
        pipeline = ExtractionPipeline()
        pipeline.add_pre_processor(custom_cleaner)
        pipeline.add_post_processor(custom_formatter)
        results = pipeline.process_files(pdf_list)
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # Initialize components
        self._setup_extractor()
        self._setup_processor()
        self._setup_formatter()
        
        # Custom processors
        self._pre_processors: List[Callable[[PDFDocument], PDFDocument]] = []
        self._post_processors: List[Callable[[StructuredDocument], StructuredDocument]] = []
        
        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def _setup_extractor(self):
        """Setup the PDF extractor with configured settings."""
        filters = []
        
        if self.config.remove_copyright:
            filters.append(create_copyright_filter(self.config.copyright_patterns))
        
        if self.config.normalize_whitespace:
            filters.append(normalize_whitespace)
        
        backend_kwargs = {}
        if self.config.extraction_backend == "ocr":
            backend_kwargs = {
                "dpi": self.config.ocr_dpi,
                "language": self.config.ocr_language
            }
        elif self.config.extraction_backend == "hybrid":
            backend_kwargs = {
                "ocr_dpi": self.config.ocr_dpi,
                "ocr_language": self.config.ocr_language
            }
        
        self._extractor = PDFExtractor(
            backend=self.config.extraction_backend,
            filters=filters,
            **backend_kwargs
        )
    
    def _setup_processor(self):
        """Setup the text processor."""
        if self.config.processor_type == "standard":
            self._processor = StandardsProcessor()
        else:
            self._processor = TextProcessor()
    
    def _setup_formatter(self):
        """Setup the output formatter."""
        self._formatter = LLMFormatter(format=self.config.output_format)
    
    def add_pre_processor(self, processor: Callable[[PDFDocument], PDFDocument]):
        """Add a pre-processor that runs after extraction but before processing."""
        self._pre_processors.append(processor)
    
    def add_post_processor(self, processor: Callable[[StructuredDocument], StructuredDocument]):
        """Add a post-processor that runs after processing but before formatting."""
        self._post_processors.append(processor)
    
    def process_file(
        self,
        pdf_path: str,
        output_path: Optional[str] = None
    ) -> ProcessingResult:
        """
        Process a single PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            output_path: Optional custom output path
            
        Returns:
            ProcessingResult with processing details
        """
        pdf_path = Path(pdf_path)
        logger.info(f"Processing: {pdf_path.name}")
        
        try:
            # Step 1: Extract
            logger.debug(f"Extracting text from {pdf_path.name}")
            document = self._extractor.extract(pdf_path)
            
            # Step 2: Pre-processing
            for pre_processor in self._pre_processors:
                document = pre_processor(document)
            
            # Step 3: Process/Structure
            logger.debug(f"Processing content from {pdf_path.name}")
            structured = self._processor.process(document)
            
            # Step 4: Post-processing
            for post_processor in self._post_processors:
                structured = post_processor(structured)
            
            # Step 5: Format and save
            if output_path is None:
                output_name = pdf_path.stem + self._formatter.get_extension()
                output_path = os.path.join(self.config.output_dir, output_name)
            
            logger.debug(f"Saving output to {output_path}")
            self._formatter.save(structured, output_path)
            
            logger.info(f"Successfully processed: {pdf_path.name} -> {output_path}")
            
            return ProcessingResult(
                source_path=str(pdf_path),
                output_path=output_path,
                success=True,
                pages_processed=document.total_pages,
                document=structured
            )
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path.name}: {str(e)}")
            return ProcessingResult(
                source_path=str(pdf_path),
                output_path=None,
                success=False,
                error=str(e)
            )
    
    def process_files(
        self,
        pdf_paths: List[str],
        parallel: bool = True
    ) -> List[ProcessingResult]:
        """
        Process multiple PDF files.
        
        Args:
            pdf_paths: List of paths to PDF files
            parallel: Whether to process in parallel
            
        Returns:
            List of ProcessingResult objects
        """
        results = []
        
        if parallel and len(pdf_paths) > 1:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {
                    executor.submit(self.process_file, path): path
                    for path in pdf_paths
                }
                
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
        else:
            for path in pdf_paths:
                result = self.process_file(path)
                results.append(result)
        
        # Summary
        successful = sum(1 for r in results if r.success)
        logger.info(f"Processed {len(results)} files: {successful} successful, {len(results) - successful} failed")
        
        return results
    
    def process_directory(
        self,
        directory: str,
        recursive: bool = False,
        pattern: str = "*.pdf"
    ) -> List[ProcessingResult]:
        """
        Process all PDFs in a directory.
        
        Args:
            directory: Path to directory containing PDFs
            recursive: Whether to search recursively
            pattern: Glob pattern for PDF files
            
        Returns:
            List of ProcessingResult objects
        """
        directory = Path(directory)
        
        if recursive:
            pdf_files = list(directory.rglob(pattern))
        else:
            pdf_files = list(directory.glob(pattern))
        
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
        
        return self.process_files([str(f) for f in pdf_files])


def create_pipeline(
    output_format: str = "markdown",
    output_dir: str = "./output",
    backend: str = "hybrid",
    remove_copyright: bool = True,
    **kwargs
) -> ExtractionPipeline:
    """
    Factory function to create a configured pipeline.
    
    Args:
        output_format: Output format ("markdown", "json", "instruction")
        output_dir: Directory for output files
        backend: Extraction backend ("direct", "ocr", "hybrid")
        remove_copyright: Whether to remove copyright notices
        **kwargs: Additional config options
        
    Returns:
        Configured ExtractionPipeline
    """
    config = PipelineConfig(
        output_format=output_format,
        output_dir=output_dir,
        extraction_backend=backend,
        remove_copyright=remove_copyright,
        **kwargs
    )
    
    return ExtractionPipeline(config)


# Convenience functions for common use cases

def extract_pdf_to_markdown(
    pdf_path: str,
    output_path: Optional[str] = None
) -> str:
    """
    Quick function to extract a PDF to markdown.
    
    Args:
        pdf_path: Path to PDF file
        output_path: Optional output path
        
    Returns:
        Path to the generated markdown file
    """
    pipeline = create_pipeline(output_format="markdown")
    result = pipeline.process_file(pdf_path, output_path)
    
    if not result.success:
        raise RuntimeError(f"Extraction failed: {result.error}")
    
    return result.output_path


def extract_pdf_to_json(
    pdf_path: str,
    output_path: Optional[str] = None
) -> str:
    """
    Quick function to extract a PDF to JSON.
    
    Args:
        pdf_path: Path to PDF file
        output_path: Optional output path
        
    Returns:
        Path to the generated JSON file
    """
    pipeline = create_pipeline(output_format="json")
    result = pipeline.process_file(pdf_path, output_path)
    
    if not result.success:
        raise RuntimeError(f"Extraction failed: {result.error}")
    
    return result.output_path


def batch_process(
    input_dir: str,
    output_dir: str = "./output",
    output_format: str = "markdown"
) -> Dict[str, Any]:
    """
    Batch process all PDFs in a directory.
    
    Args:
        input_dir: Directory containing PDFs
        output_dir: Output directory
        output_format: Output format
        
    Returns:
        Summary dict with results
    """
    pipeline = create_pipeline(
        output_format=output_format,
        output_dir=output_dir
    )
    
    results = pipeline.process_directory(input_dir)
    
    return {
        "total": len(results),
        "successful": sum(1 for r in results if r.success),
        "failed": sum(1 for r in results if not r.success),
        "results": results
    }
