#!/usr/bin/env python3
"""
PDF to LLM - Command Line Interface

Extract, process, and format PDFs for LLM consumption.
"""

import argparse
import sys
import os
from pathlib import Path

from src import ExtractionPipeline
from src.pipeline import PipelineConfig, create_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Transform PDFs into structured data for LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract a single PDF to markdown
  python -m pdf_to_llm extract document.pdf
  
  # Extract with OCR for scanned documents
  python -m pdf_to_llm extract document.pdf --backend ocr --dpi 300
  
  # Batch process a directory
  python -m pdf_to_llm batch ./pdfs --output ./output --format json
  
  # Extract to instruction format for fine-tuning
  python -m pdf_to_llm extract document.pdf --format instruction
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Extract command
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract a single PDF file"
    )
    extract_parser.add_argument(
        "input",
        help="Path to the PDF file"
    )
    extract_parser.add_argument(
        "-o", "--output",
        help="Output file path (default: auto-generated)"
    )
    extract_parser.add_argument(
        "-f", "--format",
        choices=["markdown", "json", "instruction"],
        default="markdown",
        help="Output format (default: markdown)"
    )
    extract_parser.add_argument(
        "-b", "--backend",
        choices=["direct", "ocr", "hybrid"],
        default="hybrid",
        help="Extraction backend (default: hybrid)"
    )
    extract_parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for OCR extraction (default: 300)"
    )
    extract_parser.add_argument(
        "--lang",
        default="eng",
        help="Language for OCR (default: eng)"
    )
    extract_parser.add_argument(
        "--keep-copyright",
        action="store_true",
        help="Keep copyright notices in output"
    )
    extract_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    # Batch command
    batch_parser = subparsers.add_parser(
        "batch",
        help="Process all PDFs in a directory"
    )
    batch_parser.add_argument(
        "input_dir",
        help="Directory containing PDF files"
    )
    batch_parser.add_argument(
        "-o", "--output",
        default="./output",
        help="Output directory (default: ./output)"
    )
    batch_parser.add_argument(
        "-f", "--format",
        choices=["markdown", "json", "instruction"],
        default="markdown",
        help="Output format (default: markdown)"
    )
    batch_parser.add_argument(
        "-b", "--backend",
        choices=["direct", "ocr", "hybrid"],
        default="hybrid",
        help="Extraction backend (default: hybrid)"
    )
    batch_parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Search for PDFs recursively"
    )
    batch_parser.add_argument(
        "-w", "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    batch_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show information about a PDF"
    )
    info_parser.add_argument(
        "input",
        help="Path to the PDF file"
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    # Set up logging
    import logging
    level = logging.DEBUG if getattr(args, 'verbose', False) else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if args.command == "extract":
        return cmd_extract(args)
    elif args.command == "batch":
        return cmd_batch(args)
    elif args.command == "info":
        return cmd_info(args)
    
    return 0


def cmd_extract(args):
    """Handle the extract command."""
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return 1
    
    if not input_path.suffix.lower() == '.pdf':
        print(f"Warning: File may not be a PDF: {input_path}")
    
    # Create pipeline
    config = PipelineConfig(
        extraction_backend=args.backend,
        ocr_dpi=args.dpi,
        ocr_language=args.lang,
        output_format=args.format,
        remove_copyright=not args.keep_copyright,
    )
    
    pipeline = ExtractionPipeline(config)
    
    # Process file
    result = pipeline.process_file(str(input_path), args.output)
    
    if result.success:
        print(f"✓ Successfully extracted: {result.output_path}")
        print(f"  Pages processed: {result.pages_processed}")
        return 0
    else:
        print(f"✗ Extraction failed: {result.error}")
        return 1


def cmd_batch(args):
    """Handle the batch command."""
    input_dir = Path(args.input_dir)
    
    if not input_dir.exists():
        print(f"Error: Directory not found: {input_dir}")
        return 1
    
    if not input_dir.is_dir():
        print(f"Error: Not a directory: {input_dir}")
        return 1
    
    # Create pipeline
    config = PipelineConfig(
        extraction_backend=args.backend,
        output_format=args.format,
        output_dir=args.output,
        max_workers=args.workers,
    )
    
    pipeline = ExtractionPipeline(config)
    
    # Process directory
    results = pipeline.process_directory(
        str(input_dir),
        recursive=args.recursive
    )
    
    # Print summary
    successful = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)
    
    print(f"\n{'='*50}")
    print(f"Processing Complete")
    print(f"{'='*50}")
    print(f"Total files: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print(f"\nFailed files:")
        for r in results:
            if not r.success:
                print(f"  - {r.source_path}: {r.error}")
    
    return 0 if failed == 0 else 1


def cmd_info(args):
    """Handle the info command."""
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return 1
    
    try:
        import fitz
        doc = fitz.open(str(input_path))
        
        print(f"File: {input_path.name}")
        print(f"{'='*50}")
        print(f"Pages: {len(doc)}")
        
        if doc.metadata:
            print(f"\nMetadata:")
            for key, value in doc.metadata.items():
                if value:
                    print(f"  {key}: {value}")
        
        # Sample text detection
        text_pages = 0
        for i in range(min(5, len(doc))):
            text = doc[i].get_text().strip()
            if len(text) > 100:
                text_pages += 1
        
        if text_pages > 0:
            print(f"\nText extraction: Direct extraction recommended")
        else:
            print(f"\nText extraction: OCR may be needed")
        
        doc.close()
        return 0
        
    except ImportError:
        print("Error: PyMuPDF not installed. Run: pip install pymupdf")
        return 1
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
