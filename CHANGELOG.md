# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-01-07

### Added
- Initial release
- PDF extraction with multiple backends:
  - Direct text extraction using PyMuPDF
  - OCR extraction using Tesseract
  - Hybrid mode with automatic fallback
- Text processing and content classification:
  - Automatic detection of headings, paragraphs, lists, equations, tables
  - Hierarchical section extraction
  - Standards-aware processor for technical documents
- Output formatting:
  - Markdown format with TOC and metadata
  - JSON format for RAG systems
  - Instruction format for LLM fine-tuning
- Processing pipeline:
  - Single file and batch processing
  - Parallel processing support
  - Custom pre/post processors
- Command-line interface:
  - `extract` command for single files
  - `batch` command for directories
  - `info` command for PDF analysis
- Extensibility:
  - Custom text filters
  - Custom content classifiers
  - Custom extraction backends
- Documentation:
  - README with quick start guide
  - API reference
  - Contributing guidelines
  - Example scripts
