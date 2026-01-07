#!/usr/bin/env python3
"""
Example: Custom Processing Pipeline

Demonstrates how to extend the pipeline with custom processors.
"""

from src import ExtractionPipeline, PDFExtractor
from src.pipeline import PipelineConfig
from src.processor import TextProcessor, ContentType
from src.extractor import PDFDocument


# Custom pre-processor: runs after extraction, before structuring
def add_extraction_metadata(document: PDFDocument) -> PDFDocument:
    """Add custom metadata to extracted document."""
    document.metadata["extraction_tool"] = "pdf-to-llm"
    document.metadata["custom_field"] = "your-value"
    return document


def clean_specific_patterns(document: PDFDocument) -> PDFDocument:
    """Remove domain-specific patterns from text."""
    patterns_to_remove = [
        "DRAFT - NOT FOR DISTRIBUTION",
        "INTERNAL USE ONLY",
        "[REDACTED]"
    ]
    
    for page in document.pages:
        for pattern in patterns_to_remove:
            page.text = page.text.replace(pattern, "")
    
    return document


# Custom text filter for the extractor
def remove_page_numbers(text: str) -> str:
    """Remove standalone page numbers."""
    import re
    lines = text.split('\n')
    filtered = [l for l in lines if not re.match(r'^\s*\d+\s*$', l.strip())]
    return '\n'.join(filtered)


# Custom content classifier
def classify_warnings(line: str):
    """Classify warning and caution blocks."""
    line_upper = line.strip().upper()
    if line_upper.startswith("WARNING:"):
        return ContentType.PARAGRAPH
    if line_upper.startswith("CAUTION:"):
        return ContentType.PARAGRAPH
    if line_upper.startswith("NOTE:"):
        return ContentType.FOOTNOTE
    return None


# Build custom pipeline
config = PipelineConfig(
    extraction_backend="hybrid",
    output_format="markdown",
    output_dir="./output/custom"
)

pipeline = ExtractionPipeline(config)

# Add custom processors
pipeline.add_pre_processor(add_extraction_metadata)
pipeline.add_pre_processor(clean_specific_patterns)

# For even more control, create custom extractor
extractor = PDFExtractor(
    backend="hybrid",
    filters=[remove_page_numbers]
)

# Create custom processor with additional patterns
processor = TextProcessor()
processor.add_classifier(classify_warnings)
processor.add_pattern(ContentType.EQUATION, r'^Formula \d+:')

print("Custom pipeline configured!")
print("Use pipeline.process_file() or pipeline.process_directory() to run")

# Example usage:
# result = pipeline.process_file("pdfs/your-document.pdf")
