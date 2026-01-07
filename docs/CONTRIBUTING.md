# Contributing to PDF to LLM

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- (Optional) Tesseract OCR for OCR features

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/KhademOHAli1/Standard-to-md.git
cd pdf-to-llm

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,ocr]"
```

### Project Structure

```
pdf-to-llm/
├── src/                    # Main source code
│   ├── __init__.py         # Public API
│   ├── __main__.py         # CLI
│   ├── extractor.py        # PDF extraction
│   ├── processor.py        # Text processing
│   ├── formatter.py        # Output formatting
│   └── pipeline.py         # Processing pipeline
├── tests/                  # Test suite
├── examples/               # Usage examples
├── docs/                   # Documentation
├── pyproject.toml          # Project configuration
└── README.md
```

## Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **mypy** for type checking

### Before Submitting

```bash
# Format code
black src/ tests/
isort src/ tests/

# Type check
mypy src/

# Run tests
pytest
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_extractor.py

# Run specific test
pytest tests/test_extractor.py::TestFilters::test_copyright_filter
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Use pytest fixtures for common setup
- Aim for high coverage of new features

Example test:

```python
import pytest
from src.extractor import PDFExtractor

class TestMyFeature:
    def test_basic_functionality(self):
        # Arrange
        extractor = PDFExtractor(backend="direct")
        
        # Act
        result = extractor.some_method()
        
        # Assert
        assert result is not None
    
    @pytest.fixture
    def sample_document(self):
        # Setup code
        return create_sample_document()
    
    def test_with_fixture(self, sample_document):
        assert sample_document.title == "Test"
```

## Pull Request Process

1. **Fork** the repository
2. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/my-feature
   ```
3. **Make your changes** with clear, focused commits
4. **Add tests** for new functionality
5. **Update documentation** if needed
6. **Run the test suite** to ensure everything passes
7. **Push** your branch and create a Pull Request

### PR Guidelines

- Keep PRs focused on a single feature or fix
- Write a clear description of the changes
- Reference any related issues
- Ensure CI checks pass

## Adding New Features

### Adding a New Extraction Backend

1. Create a new class inheriting from `ExtractionBackend`:

```python
# In src/extractor.py

class MyCustomBackend(ExtractionBackend):
    def is_available(self) -> bool:
        # Check if dependencies are available
        return True
    
    def extract(self, pdf_path: Path) -> PDFDocument:
        # Implementation
        pass
    
    def extract_page(self, pdf_path: Path, page_num: int) -> PageContent:
        # Implementation
        pass
```

2. Register in `PDFExtractor.BACKENDS`:

```python
class PDFExtractor:
    BACKENDS = {
        "direct": DirectTextBackend,
        "ocr": OCRBackend,
        "hybrid": HybridBackend,
        "custom": MyCustomBackend,  # Add here
    }
```

3. Add tests in `tests/test_extractor.py`

### Adding a New Output Format

1. Create a new class inheriting from `OutputFormat`:

```python
# In src/formatter.py

class MyFormat(OutputFormat):
    def format(self, document: StructuredDocument) -> str:
        # Implementation
        pass
    
    def get_extension(self) -> str:
        return ".myext"
```

2. Register in `LLMFormatter.FORMATS`:

```python
class LLMFormatter:
    FORMATS = {
        "markdown": MarkdownFormat,
        "json": JSONFormat,
        "instruction": InstructionFormat,
        "myformat": MyFormat,  # Add here
    }
```

3. Add tests in `tests/test_formatter.py`

### Adding Content Classifiers

```python
from src.processor import TextProcessor, ContentType

processor = TextProcessor()

# Add pattern-based classifier
processor.add_pattern(ContentType.EQUATION, r'^Equation \d+:')

# Add function-based classifier
def my_classifier(line: str) -> Optional[ContentType]:
    if line.startswith("CUSTOM:"):
        return ContentType.PARAGRAPH
    return None

processor.add_classifier(my_classifier)
```

## Documentation

- Update README.md for user-facing changes
- Update docs/API.md for API changes
- Add docstrings to new functions and classes
- Include examples for new features

### Docstring Style

```python
def my_function(param1: str, param2: int = 10) -> bool:
    """
    Brief description of what the function does.
    
    Args:
        param1: Description of param1
        param2: Description of param2 (default: 10)
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is empty
        
    Example:
        >>> my_function("test", 5)
        True
    """
    pass
```

## Issue Reporting

When reporting issues, please include:

1. **Description** of the problem
2. **Steps to reproduce**
3. **Expected behavior**
4. **Actual behavior**
5. **Environment info** (Python version, OS, etc.)
6. **Sample PDF** if relevant (or description of PDF type)

## Questions?

Feel free to open an issue for questions or discussions about potential features.
