"""
Semantic Chunker - Intelligent chunking with table preservation
Implements semantic chunking that preserves table structures and adds enriched metadata
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from loguru import logger
import pandas as pd
from io import StringIO


class SemanticChunker:
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 225):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Patterns for detecting different content types
        self.table_patterns = [
            # Pipe-separated tables
            r'^\s*\|.*\|.*\|\s*$',
            # Tab-separated tables (multiple tabs)
            r'^\s*[^\t\n]+\t+[^\t\n]+\t+[^\t\n]+.*$',
            # Space-separated columns (3+ columns with consistent spacing)
            r'^\s*\S+\s{2,}\S+\s{2,}\S+.*$',
            # Excel-like table headers
            r'^\s*[A-Za-z][A-Za-z0-9\s]*\s+[A-Za-z][A-Za-z0-9\s]*\s+[A-Za-z][A-Za-z0-9\s]*.*$'
        ]
        
        # Patterns for technical codes
        self.code_patterns = [
            r'\b[A-Z]{2,3}\d{2,4}\b',  # AC01, Z001, SAP123, etc.
            r'\b[A-Z]+_[A-Z0-9_]+\b',  # SAP_MODULE_CODE, etc.
            r'\b\d{4,6}[A-Z]{1,3}\b',  # 1234AB, 567890C, etc.
            r'\b[A-Z]{1,3}-\d{3,6}\b', # A-1234, AB-567890, etc.
        ]
        
        # Content type indicators
        self.content_type_indicators = {
            'table': ['tabla', 'columna', 'fila', 'datos', 'registro', 'campo'],
            'procedure': ['procedimiento', 'proceso', 'paso', 'instrucción', 'método'],
            'code': ['código', 'función', 'variable', 'parámetro', 'configuración'],
            'diagram': ['diagrama', 'esquema', 'flujo', 'gráfico', 'imagen'],
            'reference': ['referencia', 'manual', 'documentación', 'guía']
        }

    def chunk_with_table_preservation(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Main chunking method that preserves table structures
        """
        if not text or len(text.strip()) == 0:
            return []
        
        # Split text into sections
        sections = self._split_into_sections(text)
        
        chunks = []
        current_position = 0
        
        for section in sections:
            section_chunks = self._process_section(section, current_position, metadata or {})
            chunks.extend(section_chunks)
            current_position += len(section['content'])
        
        logger.info(f"Created {len(chunks)} semantic chunks with table preservation")
        return chunks

    def _split_into_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into logical sections, identifying tables and other content types
        """
        lines = text.split('\n')
        sections = []
        current_section = {'content': '', 'type': 'text', 'lines': []}
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check if this line starts a table
            if self._is_table_line(line):
                # Save current section if it has content
                if current_section['content'].strip():
                    sections.append(current_section)
                
                # Process table section
                table_section, lines_consumed = self._extract_table_section(lines, i)
                sections.append(table_section)
                
                # Reset current section
                current_section = {'content': '', 'type': 'text', 'lines': []}
                i += lines_consumed
            else:
                # Add line to current section
                current_section['content'] += line + '\n'
                current_section['lines'].append(line)
                i += 1
        
        # Add final section if it has content
        if current_section['content'].strip():
            sections.append(current_section)
        
        return sections

    def _is_table_line(self, line: str) -> bool:
        """
        Check if a line appears to be part of a table
        """
        if not line.strip():
            return False
        
        for pattern in self.table_patterns:
            if re.match(pattern, line):
                return True
        
        return False

    def _extract_table_section(self, lines: List[str], start_idx: int) -> Tuple[Dict[str, Any], int]:
        """
        Extract a complete table section starting from start_idx
        """
        table_lines = []
        i = start_idx
        
        # Look ahead to find the end of the table
        while i < len(lines):
            line = lines[i]
            
            # Empty line might be part of table formatting
            if not line.strip():
                # Check if next non-empty line is still table
                next_table_line = False
                for j in range(i + 1, min(i + 3, len(lines))):
                    if j < len(lines) and lines[j].strip():
                        if self._is_table_line(lines[j]):
                            next_table_line = True
                        break
                
                if next_table_line:
                    table_lines.append(line)
                    i += 1
                else:
                    break
            elif self._is_table_line(line):
                table_lines.append(line)
                i += 1
            else:
                break
        
        # Include context lines before and after table if available
        context_before = []
        context_after = []
        
        # Get 1-2 lines before table for context
        for j in range(max(0, start_idx - 2), start_idx):
            if j >= 0 and lines[j].strip():
                context_before.append(lines[j])
        
        # Get 1-2 lines after table for context
        for j in range(i, min(i + 2, len(lines))):
            if j < len(lines) and lines[j].strip():
                context_after.append(lines[j])
        
        # Build complete table content with context
        full_content = '\n'.join(context_before + table_lines + context_after)
        
        table_section = {
            'content': full_content,
            'type': 'table',
            'lines': context_before + table_lines + context_after,
            'table_lines': table_lines,
            'context_before': context_before,
            'context_after': context_after
        }
        
        return table_section, i - start_idx

    def _process_section(self, section: Dict[str, Any], position: int, base_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a section into chunks with appropriate metadata
        """
        content = section['content']
        section_type = section['type']
        
        if section_type == 'table':
            # Tables are kept as single chunks (unless extremely large)
            return self._create_table_chunk(section, position, base_metadata)
        else:
            # Regular text chunking with semantic boundaries
            return self._create_text_chunks(content, position, base_metadata)

    def _create_table_chunk(self, table_section: Dict[str, Any], position: int, base_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create a chunk for a table, preserving its complete structure
        """
        content = table_section['content']
        table_lines = table_section.get('table_lines', [])
        
        # Analyze table structure
        table_analysis = self._analyze_table_structure(table_lines)
        
        # Detect technical codes in table
        codes_found = self._extract_technical_codes(content)
        
        # Enhanced metadata for table
        enhanced_metadata = {
            **base_metadata,
            'content_type': 'table',
            'contains_codes': len(codes_found) > 0,
            'technical_codes': codes_found,
            'table_headers': table_analysis.get('headers', []),
            'table_rows_count': table_analysis.get('row_count', 0),
            'table_columns_count': table_analysis.get('column_count', 0),
            'table_format': table_analysis.get('format', 'unknown'),
            'has_structured_data': True,
            'chunk_type': 'table_complete',
            'preservation_method': 'table_intact'
        }
        
        chunk = {
            'text': content,
            'metadata': enhanced_metadata,
            'chunk_index': 0,
            'position': position,
            'length': len(content),
            'chunk_type': 'table'
        }
        
        logger.info(f"Created table chunk with {table_analysis.get('row_count', 0)} rows and {len(codes_found)} technical codes")
        
        return [chunk]

    def _analyze_table_structure(self, table_lines: List[str]) -> Dict[str, Any]:
        """
        Analyze the structure of a table to extract metadata
        """
        if not table_lines:
            return {}
        
        analysis = {
            'row_count': len(table_lines),
            'column_count': 0,
            'headers': [],
            'format': 'unknown'
        }
        
        # Try to detect table format and extract headers
        first_line = table_lines[0] if table_lines else ""
        
        if '|' in first_line:
            # Pipe-separated table
            analysis['format'] = 'pipe_separated'
            headers = [col.strip() for col in first_line.split('|') if col.strip()]
            analysis['headers'] = headers
            analysis['column_count'] = len(headers)
        elif '\t' in first_line:
            # Tab-separated table
            analysis['format'] = 'tab_separated'
            headers = [col.strip() for col in first_line.split('\t') if col.strip()]
            analysis['headers'] = headers
            analysis['column_count'] = len(headers)
        else:
            # Space-separated or other format
            analysis['format'] = 'space_separated'
            # Try to detect columns by consistent spacing
            words = first_line.split()
            analysis['column_count'] = len(words)
            analysis['headers'] = words[:5]  # Take first 5 as potential headers
        
        return analysis

    def _create_text_chunks(self, content: str, position: int, base_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create chunks for regular text content with semantic boundaries
        """
        if len(content) <= self.chunk_size:
            # Content fits in single chunk
            enhanced_metadata = self._enhance_metadata(content, base_metadata)
            return [{
                'text': content,
                'metadata': enhanced_metadata,
                'chunk_index': 0,
                'position': position,
                'length': len(content),
                'chunk_type': 'text'
            }]
        
        # Split into multiple chunks with semantic boundaries
        chunks = []
        sentences = self._split_into_sentences(content)
        
        current_chunk = ""
        current_position = position
        chunk_index = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Create chunk with current content
                enhanced_metadata = self._enhance_metadata(current_chunk, base_metadata)
                enhanced_metadata['chunk_index'] = chunk_index
                
                chunks.append({
                    'text': current_chunk.strip(),
                    'metadata': enhanced_metadata,
                    'chunk_index': chunk_index,
                    'position': current_position,
                    'length': len(current_chunk),
                    'chunk_type': 'text'
                })
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                current_chunk = overlap_text + sentence
                current_position += len(current_chunk) - len(overlap_text)
                chunk_index += 1
            else:
                current_chunk += sentence
        
        # Add final chunk if there's remaining content
        if current_chunk.strip():
            enhanced_metadata = self._enhance_metadata(current_chunk, base_metadata)
            enhanced_metadata['chunk_index'] = chunk_index
            
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': enhanced_metadata,
                'chunk_index': chunk_index,
                'position': current_position,
                'length': len(current_chunk),
                'chunk_type': 'text'
            })
        
        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences, preserving semantic boundaries
        """
        # Split by sentence endings, but keep the punctuation
        sentences = re.split(r'([.!?]+\s+)', text)
        
        # Recombine sentences with their punctuation
        result = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]
            if sentence.strip():
                result.append(sentence)
        
        # If no sentence boundaries found, split by paragraphs
        if len(result) <= 1:
            paragraphs = text.split('\n\n')
            result = [p + '\n\n' for p in paragraphs if p.strip()]
        
        return result

    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """
        Get the last overlap_size characters from text, trying to break at word boundaries
        """
        if len(text) <= overlap_size:
            return text
        
        overlap_text = text[-overlap_size:]
        
        # Try to break at word boundary
        space_idx = overlap_text.find(' ')
        if space_idx > 0:
            overlap_text = overlap_text[space_idx + 1:]
        
        return overlap_text

    def _enhance_metadata(self, content: str, base_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance metadata with content analysis
        """
        enhanced = {**base_metadata}
        
        # Detect technical codes
        codes_found = self._extract_technical_codes(content)
        enhanced['contains_codes'] = len(codes_found) > 0
        enhanced['technical_codes'] = codes_found
        
        # Detect content type
        content_type = self._detect_content_type(content)
        enhanced['content_type'] = content_type
        
        # Detect module/system references
        module = self._detect_module(content)
        if module:
            enhanced['module'] = module
        
        # Add semantic indicators
        enhanced['has_structured_data'] = self._has_structured_data(content)
        enhanced['chunk_method'] = 'semantic'
        
        return enhanced

    def _extract_technical_codes(self, text: str) -> List[str]:
        """
        Extract technical codes from text using predefined patterns
        """
        codes = set()
        
        for pattern in self.code_patterns:
            matches = re.findall(pattern, text)
            codes.update(matches)
        
        return list(codes)

    def _detect_content_type(self, content: str) -> str:
        """
        Detect the type of content based on keywords and patterns
        """
        content_lower = content.lower()
        
        # Count indicators for each type
        type_scores = {}
        for content_type, indicators in self.content_type_indicators.items():
            score = sum(1 for indicator in indicators if indicator in content_lower)
            if score > 0:
                type_scores[content_type] = score
        
        # Return type with highest score, default to 'text'
        if type_scores:
            return max(type_scores.items(), key=lambda x: x[1])[0]
        
        return 'text'

    def _detect_module(self, content: str) -> Optional[str]:
        """
        Detect which module/system the content refers to
        """
        content_upper = content.upper()
        
        # Module indicators
        modules = {
            'DARWIN': ['DARWIN', 'SISTEMA DARWIN'],
            'SAP': ['SAP', 'SISTEMA SAP', 'ERP']
        }
        
        for module, indicators in modules.items():
            if any(indicator in content_upper for indicator in indicators):
                return module
        
        return None

    def _has_structured_data(self, content: str) -> bool:
        """
        Check if content contains structured data patterns
        """
        # Check for list patterns, numbered items, etc.
        patterns = [
            r'^\s*\d+\.\s+',  # Numbered lists
            r'^\s*[-*]\s+',   # Bullet points
            r'^\s*[A-Za-z]\)\s+',  # Lettered lists
            r':\s*$',         # Key-value patterns
        ]
        
        lines = content.split('\n')
        structured_lines = 0
        
        for line in lines:
            for pattern in patterns:
                if re.match(pattern, line):
                    structured_lines += 1
                    break
        
        # If more than 20% of lines are structured, consider it structured data
        return structured_lines > len(lines) * 0.2
