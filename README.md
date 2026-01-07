# PDF to LLM

Transform PDF documents into structured, LLM-ready data formats. Extract text, equations, tables, and hierarchical structure from any PDF for use with Large Language Models.

## 🎯 Purpose

This toolkit converts PDF documents (especially technical standards like API MPMS, AGA, ISO) into structured formats that LLMs can effectively understand and follow. It preserves:

- Document hierarchy and sections
- Equations and formulas
- Tables and lists
- Cross-references

## ✨ Features

- **Multiple Extraction Backends**
  - Direct text extraction (fast, for text-based PDFs)
  - OCR extraction (for scanned documents)
  - Hybrid mode (automatic fallback to OCR when needed)

- **Intelligent Processing**
  - Automatic content classification (headings, equations, tables, etc.)
  - Hierarchical structure extraction
  - Copyright/boilerplate removal
  - Whitespace normalization

- **Flexible Output Formats**
  - **Markdown**: Clean, readable documentation
  - **JSON**: Structured data for RAG systems
  - **Instruction**: LLM fine-tuning format

- **Extensible Pipeline**
  - Custom pre/post processors
  - Pluggable backends
  - Configurable filters

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/KhademOHAli1/Standard-to-md.git
cd Standard-to-md

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For OCR support (optional)
pip install pytesseract pillow
# Also install Tesseract OCR: https://github.com/tesseract-ocr/tesseract
```

## 🚀 Quick Start

### Command Line

```bash
# Extract a single PDF to markdown
python -m src extract document.pdf

# Extract with OCR for scanned documents
python -m src extract scanned.pdf --backend ocr --dpi 300

# Batch process a directory
python -m src batch ./pdfs --output ./output --format json

# Get PDF info
python -m src info document.pdf
```

### Python API

```python
from src import ExtractionPipeline, PDFExtractor
from src.pipeline import create_pipeline

# Simple extraction
pipeline = create_pipeline(output_format="markdown")
result = pipeline.process_file("document.pdf")

# Custom pipeline
from src.pipeline import PipelineConfig

config = PipelineConfig(
    extraction_backend="hybrid",
    output_format="json",
    output_dir="./processed",
    remove_copyright=True
)

pipeline = ExtractionPipeline(config)
results = pipeline.process_directory("./pdfs")

# Low-level control
extractor = PDFExtractor(backend="hybrid")
document = extractor.extract("document.pdf")
print(f"Extracted {document.total_pages} pages")
print(document.full_text[:500])
```

## 📁 Project Structure

```
pdf-to-llm/
├── src/                    # Source code
│   ├── __init__.py         # Package exports
│   ├── __main__.py         # CLI entry point
│   ├── extractor.py        # PDF extraction backends
│   ├── processor.py        # Text processing & structuring
│   ├── formatter.py        # Output formatting for LLMs
│   └── pipeline.py         # Complete processing pipeline
├── pdfs/                   # Input PDF files
├── output/                 # Generated output files
├── config.json             # Default configuration
├── requirements.txt        # Python dependencies
├── setup.py                # Package setup
├── LICENSE                 # MIT License
└── README.md               # This file
```

## 🔧 Configuration

Edit `config.json` or pass options programmatically:

```json
{
    "extraction_backend": "hybrid",
    "ocr_dpi": 300,
    "ocr_language": "eng",
    "processor_type": "standard",
    "output_format": "markdown",
    "output_dir": "./output",
    "remove_copyright": true,
    "copyright_patterns": [
        "COPYRIGHT",
        "LICENSED BY",
        "ALL RIGHTS RESERVED"
    ],
    "normalize_whitespace": true,
    "max_workers": 4
}
```

## 📊 Output Formats

### Markdown
Clean, readable markdown with proper heading hierarchy:

```markdown
# Document Title

## Table of Contents
- [Section 1](#section-1)
- [Section 2](#section-2)

## Section 1

Content with preserved structure...

### 1.1 Subsection

**Formula:**
```
Q = A × V
```
```

### JSON
Structured data for programmatic access:

```json
{
    "title": "Document Title",
    "sections": [
        {
            "title": "Section 1",
            "level": 1,
            "content": [
                {"type": "paragraph", "text": "..."},
                {"type": "equation", "text": "Q = A × V"}
            ],
            "subsections": [...]
        }
    ]
}
```

### Instruction Format
Optimized for LLM training/prompting:

```markdown
# Document Title - Reference Guide

## Instructions

When applying this standard, follow these guidelines:

### 1. Flow Calculation

**Formula:**
```
Q = A × V
```

**Requirements:**
- Measure area in square meters
- Velocity in meters per second
```

## 🔌 Extending

### Custom Filters

```python
from src import PDFExtractor

def remove_watermarks(text):
    return text.replace("DRAFT", "").replace("CONFIDENTIAL", "")

extractor = PDFExtractor(
    backend="hybrid",
    filters=[remove_watermarks]
)
```

### Custom Processors

```python
from src import ExtractionPipeline

def add_metadata(document):
    document.metadata["processed_by"] = "my-system"
    return document

pipeline = ExtractionPipeline()
pipeline.add_pre_processor(add_metadata)
```

### Custom Content Classifiers

```python
from src.processor import TextProcessor, ContentType

processor = TextProcessor()
processor.add_pattern(ContentType.EQUATION, r'^Equation \d+:')

def classify_notes(line):
    if line.startswith("NOTE:"):
        return ContentType.FOOTNOTE
    return None

processor.add_classifier(classify_notes)
```

## 🛠️ Use Cases

1. **RAG Systems**: Convert standards documents to searchable chunks
2. **Fine-tuning**: Create instruction datasets from technical documents
3. **Documentation**: Generate clean markdown from PDF manuals
4. **Compliance**: Extract requirements and rules for automated checking
5. **Analysis**: Structure documents for NLP analysis

## 📋 Requirements

- Python 3.9+
- PyMuPDF (required)
- Pillow + pytesseract (optional, for OCR)
- Tesseract OCR binary (optional, for OCR)

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF parsing
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for OCR capabilities
- API, AGA, and other standards organizations for the original documents
