#!/usr/bin/env python3
"""
Example: Generate LLM Training Data

Demonstrates how to create instruction-following datasets from PDFs.
"""

import json
from pathlib import Path

from src import ExtractionPipeline
from src.pipeline import PipelineConfig
from src.processor import StandardsProcessor
from src.formatter import create_llm_context


def create_training_examples(structured_doc):
    """Generate Q&A training examples from a structured document."""
    examples = []
    
    for section in structured_doc.sections:
        # Create questions from section titles
        question = section_to_question(section.title)
        
        # Create answer from section content
        answer_parts = []
        for block in section.content:
            if block.text.strip():
                answer_parts.append(block.text.strip())
        
        if answer_parts and question:
            examples.append({
                "instruction": question,
                "input": "",
                "output": "\n".join(answer_parts[:5])  # Limit length
            })
        
        # Recursively process subsections
        for subsection in section.subsections:
            sub_examples = create_subsection_examples(subsection)
            examples.extend(sub_examples)
    
    return examples


def section_to_question(title: str) -> str:
    """Convert section title to a question."""
    title = title.strip()
    
    # Remove numbering
    import re
    title = re.sub(r'^\d+[\.\d]*\s*', '', title)
    
    # Generate appropriate question
    title_lower = title.lower()
    
    if "definition" in title_lower:
        return f"What is the definition of {title.replace('Definition', '').strip()}?"
    elif "calculation" in title_lower or "formula" in title_lower:
        return f"How do you perform the calculation for {title}?"
    elif "requirement" in title_lower:
        return f"What are the requirements for {title}?"
    elif "procedure" in title_lower:
        return f"What is the procedure for {title}?"
    elif "specification" in title_lower:
        return f"What are the specifications for {title}?"
    else:
        return f"According to the standard, what does {title} specify?"


def create_subsection_examples(section, parent_context=""):
    """Create examples from subsections with parent context."""
    examples = []
    
    context = f"{parent_context} > {section.title}" if parent_context else section.title
    
    answer_parts = []
    for block in section.content:
        if block.text.strip():
            answer_parts.append(block.text.strip())
    
    if answer_parts:
        examples.append({
            "instruction": f"Explain {section.title}",
            "input": f"Context: {context}",
            "output": "\n".join(answer_parts[:3])
        })
    
    for subsection in section.subsections:
        examples.extend(create_subsection_examples(subsection, context))
    
    return examples


def generate_training_dataset(pdf_dir: str, output_file: str):
    """Generate a complete training dataset from a directory of PDFs."""
    
    # Configure pipeline for JSON output (easier to process)
    config = PipelineConfig(
        extraction_backend="hybrid",
        output_format="json",
        output_dir="./temp_json",
        processor_type="standard"
    )
    
    pipeline = ExtractionPipeline(config)
    
    # Process all PDFs
    results = pipeline.process_directory(pdf_dir)
    
    # Collect all training examples
    all_examples = []
    
    for result in results:
        if result.success and result.document:
            examples = create_training_examples(result.document)
            
            # Add source attribution
            for ex in examples:
                ex["source"] = Path(result.source_path).name
            
            all_examples.extend(examples)
    
    # Save training dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_examples, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {len(all_examples)} training examples")
    print(f"Saved to: {output_file}")
    
    return all_examples


# Example usage
if __name__ == "__main__":
    print("Training Data Generator")
    print("=" * 50)
    print()
    print("Usage:")
    print("  examples = generate_training_dataset('./pdfs', 'training_data.json')")
    print()
    print("This will:")
    print("  1. Extract all PDFs in the pdfs/ directory")
    print("  2. Convert sections to Q&A format")
    print("  3. Save as JSON for fine-tuning")
