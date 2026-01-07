# API Reference

## src.extractor

### Classes

#### `PDFExtractor`
Main PDF extraction interface.

```python
PDFExtractor(
    backend: str = "hybrid",      # "direct", "ocr", or "hybrid"
    filters: List[Callable] = None,
    **backend_kwargs
)
```

**Methods:**
- `extract(pdf_path: str | Path) -> PDFDocument` - Extract entire PDF
- `extract_pages(pdf_path, page_numbers: List[int]) -> List[PageContent]` - Extract specific pages
- `iter_pages(pdf_path) -> Iterator[PageContent]` - Memory-efficient page iteration
- `add_filter(filter_func: Callable[[str], str])` - Add text filter

#### `PDFDocument`
Represents an extracted PDF.

```python
@dataclass
class PDFDocument:
    source_path: str
    title: str
    pages: List[PageContent]
    metadata: dict
    
    @property
    def total_pages(self) -> int
    
    @property
    def full_text(self) -> str
```

#### `PageContent`
Represents a single page.

```python
@dataclass
class PageContent:
    page_number: int
    text: str
    images: List[bytes] = []
    tables: List[str] = []
    metadata: dict = {}
```

### Functions

#### `create_copyright_filter`
```python
create_copyright_filter(patterns: List[str] = None) -> Callable[[str], str]
```
Creates a filter that removes lines containing copyright patterns.

#### `create_line_filter`
```python
create_line_filter(min_length: int = 0, max_length: int = None) -> Callable[[str], str]
```
Creates a filter based on line length.

#### `normalize_whitespace`
```python
normalize_whitespace(text: str) -> str
```
Normalizes excessive whitespace.

---

## src.processor

### Classes

#### `TextProcessor`
Base text processor for content classification.

```python
TextProcessor()
```

**Methods:**
- `classify_line(line: str) -> ContentType` - Classify a single line
- `split_into_blocks(text: str) -> List[ContentBlock]` - Split text into classified blocks
- `extract_sections(blocks: List[ContentBlock]) -> List[Section]` - Extract hierarchical sections
- `process(document: PDFDocument) -> StructuredDocument` - Full processing pipeline
- `add_pattern(content_type: ContentType, pattern: str)` - Add classification pattern
- `add_classifier(classifier: Callable)` - Add custom classifier function

#### `StandardsProcessor`
Extended processor for technical standards documents. Inherits from `TextProcessor`.

```python
StandardsProcessor()
```

**Additional Methods:**
- `extract_equations(text: str) -> List[Dict]` - Extract equations with variable definitions
- `extract_tables(text: str) -> List[Dict]` - Extract table references

#### `ContentType`
Enum of content types.

```python
class ContentType(Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    EQUATION = "equation"
    CODE = "code"
    FIGURE_CAPTION = "figure_caption"
    FOOTNOTE = "footnote"
    PAGE_NUMBER = "page_number"
    HEADER = "header"
    FOOTER = "footer"
    UNKNOWN = "unknown"
```

#### `StructuredDocument`
Fully processed document.

```python
@dataclass
class StructuredDocument:
    title: str
    source: str
    sections: List[Section]
    metadata: dict
    
    def to_dict(self) -> dict
```

---

## src.formatter

### Classes

#### `LLMFormatter`
Main formatter interface.

```python
LLMFormatter(
    format: str = "markdown",  # "markdown", "json", "instruction"
    **format_kwargs
)
```

**Methods:**
- `format(document: StructuredDocument) -> str` - Format document
- `save(document: StructuredDocument, output_path: str)` - Format and save
- `get_extension() -> str` - Get file extension

#### `MarkdownFormat`
```python
MarkdownFormat(
    include_toc: bool = True,
    include_metadata: bool = True,
    max_heading_depth: int = 4
)
```

#### `JSONFormat`
```python
JSONFormat(
    indent: int = 2,
    include_raw_text: bool = True
)
```

#### `InstructionFormat`
```python
InstructionFormat(
    instruction_style: str = "detailed",  # "detailed", "concise", "qa"
    include_examples: bool = True
)
```

### Functions

#### `create_llm_context`
```python
create_llm_context(
    document: StructuredDocument,
    max_tokens: int = 4000,
    focus_sections: List[str] = None
) -> str
```
Creates condensed context string for LLM prompts.

---

## src.pipeline

### Classes

#### `ExtractionPipeline`
Complete processing pipeline.

```python
ExtractionPipeline(config: PipelineConfig = None)
```

**Methods:**
- `process_file(pdf_path: str, output_path: str = None) -> ProcessingResult`
- `process_files(pdf_paths: List[str], parallel: bool = True) -> List[ProcessingResult]`
- `process_directory(directory: str, recursive: bool = False) -> List[ProcessingResult]`
- `add_pre_processor(processor: Callable)` - Add pre-processing step
- `add_post_processor(processor: Callable)` - Add post-processing step

#### `PipelineConfig`
```python
@dataclass
class PipelineConfig:
    extraction_backend: str = "hybrid"
    ocr_dpi: int = 300
    ocr_language: str = "eng"
    processor_type: str = "standard"
    output_format: str = "markdown"
    output_dir: str = "./output"
    remove_copyright: bool = True
    copyright_patterns: List[str] = ["COPYRIGHT", ...]
    normalize_whitespace: bool = True
    max_workers: int = 4
```

#### `ProcessingResult`
```python
@dataclass
class ProcessingResult:
    source_path: str
    output_path: Optional[str]
    success: bool
    error: Optional[str] = None
    pages_processed: int = 0
    document: Optional[StructuredDocument] = None
```

### Functions

#### `create_pipeline`
```python
create_pipeline(
    output_format: str = "markdown",
    output_dir: str = "./output",
    backend: str = "hybrid",
    remove_copyright: bool = True,
    **kwargs
) -> ExtractionPipeline
```

#### `extract_pdf_to_markdown`
```python
extract_pdf_to_markdown(pdf_path: str, output_path: str = None) -> str
```

#### `extract_pdf_to_json`
```python
extract_pdf_to_json(pdf_path: str, output_path: str = None) -> str
```

#### `batch_process`
```python
batch_process(
    input_dir: str,
    output_dir: str = "./output",
    output_format: str = "markdown"
) -> Dict[str, Any]
```
