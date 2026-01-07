"""
LLM Formatting Module

Transforms processed documents into formats optimized for LLM consumption,
including structured markdown, JSON, and instruction-following formats.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from datetime import datetime

from .processor import StructuredDocument, Section, ContentBlock, ContentType


class OutputFormat(ABC):
    """Abstract base class for output formatters."""
    
    @abstractmethod
    def format(self, document: StructuredDocument) -> str:
        """Format the document into the target format."""
        pass
    
    @abstractmethod
    def get_extension(self) -> str:
        """Get the file extension for this format."""
        pass


class MarkdownFormat(OutputFormat):
    """
    Format documents as clean, structured Markdown.
    Optimized for LLM readability and context.
    """
    
    def __init__(
        self,
        include_toc: bool = True,
        include_metadata: bool = True,
        max_heading_depth: int = 4
    ):
        self.include_toc = include_toc
        self.include_metadata = include_metadata
        self.max_heading_depth = max_heading_depth
    
    def format(self, document: StructuredDocument) -> str:
        parts = []
        
        # Title
        parts.append(f"# {document.title}\n")
        
        # Metadata block
        if self.include_metadata and document.metadata:
            parts.append(self._format_metadata(document.metadata))
        
        # Table of contents
        if self.include_toc:
            toc = self._generate_toc(document.sections)
            if toc:
                parts.append("## Table of Contents\n")
                parts.append(toc)
                parts.append("\n---\n")
        
        # Content
        for section in document.sections:
            parts.append(self._format_section(section, level=2))
        
        return '\n'.join(parts)
    
    def _format_metadata(self, metadata: dict) -> str:
        """Format metadata as YAML frontmatter."""
        lines = ["---"]
        for key, value in metadata.items():
            if value:
                lines.append(f"{key}: {value}")
        lines.append("---\n")
        return '\n'.join(lines)
    
    def _generate_toc(self, sections: List[Section], level: int = 0) -> str:
        """Generate table of contents."""
        lines = []
        indent = "  " * level
        
        for section in sections:
            # Create anchor link
            anchor = section.title.lower()
            anchor = anchor.replace(' ', '-')
            anchor = ''.join(c for c in anchor if c.isalnum() or c == '-')
            
            lines.append(f"{indent}- [{section.title}](#{anchor})")
            
            if section.subsections:
                lines.append(self._generate_toc(section.subsections, level + 1))
        
        return '\n'.join(lines)
    
    def _format_section(self, section: Section, level: int) -> str:
        """Format a section and its subsections."""
        parts = []
        
        # Heading (cap at max depth)
        heading_level = min(level, self.max_heading_depth)
        heading_prefix = '#' * heading_level
        parts.append(f"\n{heading_prefix} {section.title}\n")
        
        # Content blocks
        for block in section.content:
            formatted = self._format_block(block)
            if formatted:
                parts.append(formatted)
        
        # Subsections
        for subsection in section.subsections:
            parts.append(self._format_section(subsection, level + 1))
        
        return '\n'.join(parts)
    
    def _format_block(self, block: ContentBlock) -> str:
        """Format a content block based on its type."""
        text = block.text.strip()
        if not text:
            return ""
        
        if block.content_type == ContentType.LIST_ITEM:
            # Ensure proper list formatting
            lines = text.split('\n')
            formatted_lines = []
            for line in lines:
                line = line.strip()
                if not line.startswith(('-', '*', '•')):
                    if not line[0].isdigit():
                        line = f"- {line}"
                formatted_lines.append(line)
            return '\n'.join(formatted_lines) + '\n'
        
        elif block.content_type == ContentType.EQUATION:
            # Format as code block or math block
            if '$' in text:
                return f"\n{text}\n"
            return f"\n```\n{text}\n```\n"
        
        elif block.content_type == ContentType.TABLE:
            return f"\n{text}\n"
        
        elif block.content_type == ContentType.CODE:
            return f"\n```\n{text}\n```\n"
        
        elif block.content_type == ContentType.FIGURE_CAPTION:
            return f"\n*{text}*\n"
        
        elif block.content_type == ContentType.FOOTNOTE:
            return f"\n> {text}\n"
        
        else:
            # Paragraph
            return f"\n{text}\n"
    
    def get_extension(self) -> str:
        return ".md"


class JSONFormat(OutputFormat):
    """
    Format documents as structured JSON.
    Useful for programmatic access and RAG systems.
    """
    
    def __init__(self, indent: int = 2, include_raw_text: bool = True):
        self.indent = indent
        self.include_raw_text = include_raw_text
    
    def format(self, document: StructuredDocument) -> str:
        data = {
            "title": document.title,
            "source": document.source,
            "metadata": document.metadata,
            "extracted_at": datetime.now().isoformat(),
            "sections": [self._section_to_dict(s) for s in document.sections]
        }
        
        return json.dumps(data, indent=self.indent, ensure_ascii=False)
    
    def _section_to_dict(self, section: Section) -> dict:
        content_items = []
        for block in section.content:
            item = {
                "type": block.content_type.value,
                "text": block.text
            }
            if block.metadata:
                item["metadata"] = block.metadata
            content_items.append(item)
        
        return {
            "title": section.title,
            "level": section.level,
            "content": content_items,
            "subsections": [self._section_to_dict(s) for s in section.subsections]
        }
    
    def get_extension(self) -> str:
        return ".json"


class InstructionFormat(OutputFormat):
    """
    Format documents as instruction-following prompts for LLMs.
    Creates structured instructions that LLMs can follow.
    """
    
    def __init__(
        self,
        instruction_style: str = "detailed",  # "detailed", "concise", "qa"
        include_examples: bool = True
    ):
        self.instruction_style = instruction_style
        self.include_examples = include_examples
    
    def format(self, document: StructuredDocument) -> str:
        if self.instruction_style == "detailed":
            return self._format_detailed(document)
        elif self.instruction_style == "concise":
            return self._format_concise(document)
        elif self.instruction_style == "qa":
            return self._format_qa(document)
        else:
            return self._format_detailed(document)
    
    def _format_detailed(self, document: StructuredDocument) -> str:
        """Create detailed instruction format."""
        parts = []
        
        parts.append(f"# {document.title} - Reference Guide\n")
        parts.append("## Overview\n")
        parts.append(f"This document provides detailed instructions and specifications from **{document.title}**.\n")
        parts.append("Use this reference to ensure compliance with the standard's requirements.\n")
        
        parts.append("## Instructions\n")
        parts.append("When applying this standard, follow these guidelines:\n")
        
        for i, section in enumerate(document.sections, 1):
            parts.append(self._format_instruction_section(section, i))
        
        parts.append("\n## Key Points to Remember\n")
        parts.append("- Always refer to the specific section numbers when implementing requirements\n")
        parts.append("- Equations and formulas must be applied exactly as specified\n")
        parts.append("- When in doubt, refer to the original standard document\n")
        
        return '\n'.join(parts)
    
    def _format_instruction_section(self, section: Section, num: int) -> str:
        """Format a section as an instruction block."""
        parts = []
        
        parts.append(f"\n### {num}. {section.title}\n")
        
        # Extract key points from content
        for block in section.content:
            if block.content_type == ContentType.EQUATION:
                parts.append(f"\n**Formula:**\n```\n{block.text}\n```\n")
            elif block.content_type == ContentType.LIST_ITEM:
                parts.append(f"\n**Requirements:**\n{block.text}\n")
            elif block.content_type == ContentType.PARAGRAPH:
                # Summarize long paragraphs
                text = block.text
                if len(text) > 500:
                    text = text[:500] + "..."
                parts.append(f"\n{text}\n")
        
        # Process subsections
        for j, subsection in enumerate(section.subsections, 1):
            sub_num = f"{num}.{j}"
            parts.append(f"\n#### {sub_num} {subsection.title}\n")
            for block in subsection.content:
                if block.text.strip():
                    parts.append(f"{block.text}\n")
        
        return '\n'.join(parts)
    
    def _format_concise(self, document: StructuredDocument) -> str:
        """Create concise reference format."""
        parts = []
        
        parts.append(f"# {document.title} - Quick Reference\n")
        
        for section in document.sections:
            parts.append(f"\n## {section.title}\n")
            
            # Only include equations and key points
            for block in section.content:
                if block.content_type in [ContentType.EQUATION, ContentType.LIST_ITEM]:
                    parts.append(f"{block.text}\n")
        
        return '\n'.join(parts)
    
    def _format_qa(self, document: StructuredDocument) -> str:
        """Create Q&A format for fine-tuning."""
        parts = []
        
        parts.append(f"# {document.title} - Q&A Format\n")
        parts.append("Use these Q&A pairs for training or reference.\n")
        
        for section in document.sections:
            # Generate questions from section titles
            question = self._title_to_question(section.title)
            
            # Compile answer from content
            answer_parts = []
            for block in section.content:
                if block.text.strip():
                    answer_parts.append(block.text.strip())
            
            if answer_parts:
                answer = '\n'.join(answer_parts[:3])  # Limit answer length
                
                parts.append(f"\n**Q: {question}**\n")
                parts.append(f"A: {answer}\n")
        
        return '\n'.join(parts)
    
    def _title_to_question(self, title: str) -> str:
        """Convert a section title to a question."""
        title = title.strip()
        
        # Remove numbering
        title = ' '.join(title.split()[1:]) if title[0].isdigit() else title
        
        # Generate question
        lower_title = title.lower()
        if "definition" in lower_title:
            return f"What is the definition of {title.replace('Definition', '').strip()}?"
        elif "calculation" in lower_title or "formula" in lower_title:
            return f"How do you calculate {title.replace('Calculation', '').replace('Formula', '').strip()}?"
        elif "requirement" in lower_title:
            return f"What are the requirements for {title.replace('Requirements', '').strip()}?"
        elif "procedure" in lower_title:
            return f"What is the procedure for {title.replace('Procedure', '').strip()}?"
        else:
            return f"What does the standard say about {title}?"
    
    def get_extension(self) -> str:
        return ".md"


class LLMFormatter:
    """
    Main formatter interface for converting documents to LLM-ready formats.
    
    Usage:
        formatter = LLMFormatter(format="markdown")
        output = formatter.format(structured_document)
        formatter.save(structured_document, "output.md")
    """
    
    FORMATS = {
        "markdown": MarkdownFormat,
        "json": JSONFormat,
        "instruction": InstructionFormat,
    }
    
    def __init__(
        self,
        format: str = "markdown",
        **format_kwargs
    ):
        if format not in self.FORMATS:
            raise ValueError(f"Unknown format: {format}. Available: {list(self.FORMATS.keys())}")
        
        self._formatter = self.FORMATS[format](**format_kwargs)
    
    def format(self, document: StructuredDocument) -> str:
        """Format the document."""
        return self._formatter.format(document)
    
    def save(self, document: StructuredDocument, output_path: str):
        """Format and save the document to a file."""
        content = self.format(document)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def get_extension(self) -> str:
        """Get the recommended file extension."""
        return self._formatter.get_extension()


def create_llm_context(
    document: StructuredDocument,
    max_tokens: int = 4000,
    focus_sections: Optional[List[str]] = None
) -> str:
    """
    Create a condensed context string suitable for LLM prompts.
    
    Args:
        document: The structured document
        max_tokens: Approximate maximum tokens (using char estimate)
        focus_sections: Optional list of section titles to prioritize
        
    Returns:
        A condensed string representation of the document
    """
    max_chars = max_tokens * 4  # Rough estimate
    
    parts = [f"Document: {document.title}\n"]
    
    sections_to_process = document.sections
    if focus_sections:
        sections_to_process = [
            s for s in document.sections
            if any(f.lower() in s.title.lower() for f in focus_sections)
        ]
    
    current_length = len(parts[0])
    
    for section in sections_to_process:
        section_text = f"\n## {section.title}\n"
        for block in section.content:
            section_text += f"{block.text}\n"
        
        if current_length + len(section_text) > max_chars:
            # Truncate and add indicator
            remaining = max_chars - current_length - 50
            if remaining > 100:
                section_text = section_text[:remaining] + "\n[Content truncated...]"
                parts.append(section_text)
            break
        
        parts.append(section_text)
        current_length += len(section_text)
    
    return ''.join(parts)
