#!/usr/bin/env python3
"""
Example: Basic PDF Extraction

Demonstrates simple PDF to Markdown conversion.
"""

from src import ExtractionPipeline
from src.pipeline import create_pipeline, PipelineConfig

# Example 1: Simple one-liner
# result = extract_pdf_to_markdown("document.pdf")

# Example 2: Using the pipeline factory
pipeline = create_pipeline(
    output_format="markdown",
    output_dir="./output",
    backend="hybrid"
)

# Process a single file
# result = pipeline.process_file("pdfs/your-document.pdf")
# print(f"Output saved to: {result.output_path}")

# Example 3: Full configuration control
config = PipelineConfig(
    extraction_backend="hybrid",      # "direct", "ocr", or "hybrid"
    ocr_dpi=300,                       # Higher = better quality, slower
    ocr_language="eng",                # Tesseract language code
    processor_type="standard",         # Use standards-aware processor
    output_format="markdown",          # "markdown", "json", "instruction"
    output_dir="./output",
    remove_copyright=True,
    copyright_patterns=[
        "COPYRIGHT",
        "LICENSED BY",
        "ALL RIGHTS RESERVED",
        "AMERICAN PETROLEUM INSTITUTE",
    ],
    normalize_whitespace=True,
    max_workers=4
)

pipeline = ExtractionPipeline(config)

# Batch process all PDFs in a directory
# results = pipeline.process_directory("./pdfs")

# Check results
# for result in results:
#     if result.success:
#         print(f"✓ {result.source_path} -> {result.output_path}")
#     else:
#         print(f"✗ {result.source_path}: {result.error}")

print("Example script ready - uncomment lines to run with your PDFs")
