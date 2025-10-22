"""
Structured Response Parser
Handles parsing and validation of structured JSON responses from LLM
"""

import json
import re
import logging
from typing import Dict, Any, Optional, Tuple
from src.generation.structured_response_schema import (
    StructuredResponse, ConfidenceAssessment, SourceReference, 
    ResponseMetadata, ResponseType, ConfidenceLevel,
    get_confidence_level, get_confidence_emoji
)

logger = logging.getLogger(__name__)


class StructuredResponseParser:
    """Parser for structured JSON responses from LLM"""
    
    def __init__(self):
        """Initialize the parser"""
        self.fallback_enabled = True
    
    def parse_response(self, raw_response: str, metadata: Dict[str, Any] = None) -> Tuple[Dict[str, Any], bool]:
        """
        Parse LLM response and extract structured information.
        
        Args:
            raw_response: Raw response text from LLM
            metadata: Additional metadata to include
            
        Returns:
            Tuple of (parsed_data, is_structured)
            - parsed_data: Dictionary with structured response data
            - is_structured: True if response was in JSON format, False if fallback was used
        """
        metadata = metadata or {}
        
        # Try to parse as structured JSON response
        structured_data = self._try_parse_json_response(raw_response)
        
        if structured_data:
            logger.info("Successfully parsed structured JSON response")
            # Validate and enhance the structured data
            enhanced_data = self._enhance_structured_data(structured_data, metadata)
            return enhanced_data, True
        
        # Fallback to legacy parsing
        logger.info("JSON parsing failed, falling back to legacy parsing")
        fallback_data = self._fallback_parse(raw_response, metadata)
        return fallback_data, False
    
    def _try_parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Try to extract and parse JSON from the response.
        
        Args:
            response: Raw response text
            
        Returns:
            Parsed JSON data or None if parsing fails
        """
        # Preprocess: Remove any text before the first opening brace
        preprocessed_response = self._preprocess_response(response)
        
        try:
            # Clean up potential formatting issues (newlines, tabs, etc.)
            cleaned_json = self._clean_json_formatting(preprocessed_response.strip())
            
            # First, try to parse the cleaned response as JSON
            parsed = json.loads(cleaned_json)
            logger.debug(f"Successfully parsed JSON with keys: {list(parsed.keys())}")
            # Validate that it looks like our structured response
            if self._validate_json_structure(parsed):
                logger.debug("JSON structure validation passed")
                return parsed
            else:
                logger.warning(f"JSON structure validation failed. Missing required fields. Available keys: {list(parsed.keys())}")
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parsing failed: {e}")
            logger.debug(f"Error position: {getattr(e, 'pos', 'unknown')}")
            logger.debug(f"Preprocessed response (first 200 chars): {preprocessed_response[:200]}")
            logger.debug(f"Preprocessed response (last 200 chars): ...{preprocessed_response[-200:]}")
        
        # Try to find JSON within code blocks or extract the largest JSON object
        json_patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
            r'(\{[^{}]*"response_type"[^{}]*\})',
            r'(\{.*?\})'  # More specific pattern to avoid greedy matching
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            for match in matches:
                try:
                    # Also preprocess matches from patterns
                    preprocessed_match = self._preprocess_response(match)
                    parsed = json.loads(preprocessed_match.strip())
                    logger.debug(f"Pattern match parsed JSON with keys: {list(parsed.keys())}")
                    # Validate that it looks like our structured response
                    if self._validate_json_structure(parsed):
                        logger.debug("Pattern match JSON structure validation passed")
                        return parsed
                    else:
                        logger.debug(f"Pattern match JSON structure validation failed. Available keys: {list(parsed.keys())}")
                except json.JSONDecodeError as e:
                    logger.debug(f"Pattern match JSON parsing failed: {e}")
                    continue
        
        # Try to extract JSON by finding balanced braces
        try:
            start_idx = preprocessed_response.find('{')
            if start_idx != -1:
                brace_count = 0
                end_idx = start_idx
                for i, char in enumerate(preprocessed_response[start_idx:], start_idx):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                
                if brace_count == 0:  # Found balanced JSON
                    json_str = preprocessed_response[start_idx:end_idx]
                    parsed = json.loads(json_str)
                    logger.debug(f"Balanced brace extraction parsed JSON with keys: {list(parsed.keys())}")
                    if self._validate_json_structure(parsed):
                        logger.debug("Balanced brace JSON structure validation passed")
                        return parsed
                    else:
                        logger.debug(f"Balanced brace JSON structure validation failed. Available keys: {list(parsed.keys())}")
        except (json.JSONDecodeError, Exception) as e:
            logger.debug(f"Balanced brace extraction failed: {e}")
        
        logger.warning("All JSON parsing attempts failed")
        return None
    
    def _preprocess_response(self, response: str) -> str:
        """
        Preprocess response to remove any text before the first opening brace.
        
        This handles cases where Claude Haiku 3 adds introductory text before the JSON,
        such as: "De acuerdo a la información proporcionada... { ... }"
        
        Args:
            response: Raw response text
            
        Returns:
            Cleaned response starting from the first opening brace
        """
        # Find the first opening brace
        first_brace_index = response.find('{')
        
        if first_brace_index == -1:
            # No opening brace found, return original response
            return response
        
        # Return everything from the first brace onwards
        cleaned_response = response[first_brace_index:].strip()
        
        # Additional cleaning: remove any trailing text after the last closing brace
        last_brace_index = cleaned_response.rfind('}')
        if last_brace_index != -1:
            cleaned_response = cleaned_response[:last_brace_index + 1].strip()
        
        # Log the preprocessing if text was removed
        if first_brace_index > 0:
            removed_text = response[:first_brace_index].strip()
            if removed_text:
                logger.info(f"Preprocessed response: removed {len(removed_text)} characters of introductory text")
                logger.debug(f"Removed text: {removed_text[:100]}...")
        
        # Log the final cleaned response for debugging
        logger.debug(f"Final cleaned response length: {len(cleaned_response)} characters")
        logger.debug(f"Cleaned response starts with: {cleaned_response[:50]}...")
        logger.debug(f"Cleaned response ends with: ...{cleaned_response[-50:]}")
        
        return cleaned_response
    
    def _clean_json_formatting(self, json_str: str) -> str:
        """
        Enhanced JSON string cleaner that handles all common LLM JSON issues:
        1. Unescaped quotes in string values (like "/consumos" endpoints)
        2. Unescaped newlines and control characters
        3. Incomplete JSON structures (missing closing braces)
        4. Trailing commas
        
        Args:
            json_str: Raw JSON string that may have formatting issues
            
        Returns:
            Cleaned JSON string that should be parseable
        """
        try:
            # First attempt: try to parse as-is
            json.loads(json_str)
            return json_str  # If it parses, return as-is
        except json.JSONDecodeError:
            pass
        
        logger.debug("Attempting enhanced JSON cleaning")
        
        # Apply the ultimate JSON fix
        try:
            fixed = self._ultimate_json_fix(json_str)
            json.loads(fixed)  # Validate the fix
            logger.debug("Successfully cleaned JSON using ultimate fix method")
            return fixed
        except (json.JSONDecodeError, Exception) as e:
            logger.debug(f"Ultimate fix method failed: {e}")
        
        # Fallback to legacy methods
        return self._legacy_json_cleaning(json_str)
    
    def _ultimate_json_fix(self, json_str: str) -> str:
        """
        Ultimate JSON fixer that handles all common LLM JSON issues.
        """
        # Step 1: Fix typographic quotes first (most common issue)
        fixed = self._fix_typographic_quotes(json_str)
        
        # Step 2: Fix structural issues
        fixed = self._fix_structural_issues(fixed)
        
        # Step 3: Fix unescaped quotes in string values (MAIN ISSUE)
        fixed = self._fix_unescaped_quotes_smart(fixed)
        
        # Step 4: Fix control characters
        fixed = self._fix_control_characters(fixed)
        
        # Step 5: Final cleanup
        fixed = self._final_json_cleanup(fixed)
        
        return fixed
    
    def _fix_structural_issues(self, json_str: str) -> str:
        """Fix structural JSON issues like incomplete braces and trailing commas."""
        # Remove trailing whitespace and newlines
        fixed = json_str.rstrip()
        
        # Fix trailing comma at the end
        if fixed.endswith(','):
            fixed = fixed[:-1]
        
        # Count braces to determine if JSON is incomplete
        open_braces = fixed.count('{')
        close_braces = fixed.count('}')
        
        if open_braces > close_braces:
            # Add missing closing braces
            missing_braces = open_braces - close_braces
            fixed += '}' * missing_braces
            logger.debug(f"Added {missing_braces} missing closing braces")
        
        # Handle incomplete arrays
        open_brackets = fixed.count('[')
        close_brackets = fixed.count(']')
        
        if open_brackets > close_brackets:
            missing_brackets = open_brackets - close_brackets
            fixed += ']' * missing_brackets
            logger.debug(f"Added {missing_brackets} missing closing brackets")
        
        return fixed
    
    def _fix_typographic_quotes(self, json_str: str) -> str:
        """Fix typographic quotes (curly quotes) that LLMs sometimes generate."""
        quote_mapping = {
            '"': '"',  # LEFT DOUBLE QUOTATION MARK
            '"': '"',  # RIGHT DOUBLE QUOTATION MARK
            ''': "'",  # LEFT SINGLE QUOTATION MARK
            ''': "'",  # RIGHT SINGLE QUOTATION MARK
            '„': '"',  # DOUBLE LOW-9 QUOTATION MARK
            '‚': "'",  # SINGLE LOW-9 QUOTATION MARK
            '«': '"',  # LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
            '»': '"',  # RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
            '‹': "'",  # SINGLE LEFT-POINTING ANGLE QUOTATION MARK
            '›': "'",  # SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
        }
        
        fixed_str = json_str
        replacements_made = 0
        
        for typographic, straight in quote_mapping.items():
            if typographic in fixed_str:
                count = fixed_str.count(typographic)
                fixed_str = fixed_str.replace(typographic, straight)
                replacements_made += count
                logger.debug(f"Replaced {count} instances of '{typographic}' with '{straight}'")
        
        if replacements_made > 0:
            logger.info(f"Fixed {replacements_made} typographic quote characters")
        
        return fixed_str

    def _fix_unescaped_quotes_smart(self, json_str: str) -> str:
        """
        Smart fix for unescaped quotes within JSON string values.
        Uses a state machine approach to properly identify string boundaries.
        """
        result = []
        i = 0
        in_string = False
        escape_next = False
        string_start_pos = -1
        
        while i < len(json_str):
            char = json_str[i]
            
            if escape_next:
                # This character is escaped, add as-is
                result.append(char)
                escape_next = False
            elif char == '\\':
                # This is an escape character
                result.append(char)
                escape_next = True
            elif char == '"':
                if not in_string:
                    # Starting a new string
                    in_string = True
                    string_start_pos = i
                    result.append(char)
                    logger.debug(f"Starting string at position {i}")
                else:
                    # We're in a string, check if this quote should end the string
                    if self._is_string_terminator_smart(json_str, i):
                        # This quote ends the string
                        in_string = False
                        result.append(char)
                        logger.debug(f"Ending string at position {i} (started at {string_start_pos})")
                    else:
                        # This quote is inside the string, escape it
                        result.append('\\"')
                        logger.debug(f"Escaped quote at position {i} (inside string started at {string_start_pos})")
            else:
                result.append(char)
            
            i += 1
        
        return ''.join(result)

    def _fix_unescaped_quotes_in_strings(self, json_str: str) -> str:
        """Legacy method - kept for compatibility."""
        return self._fix_unescaped_quotes_smart(json_str)
    
    def _is_string_terminator_smart(self, json_str: str, quote_pos: int) -> bool:
        """
        Smart determination if a quote at the given position terminates a string.
        Uses better heuristics to identify JSON structure.
        """
        # Look at what comes after the quote (ignoring whitespace)
        remaining = json_str[quote_pos + 1:].lstrip()
        
        if not remaining:
            return True  # End of string
        
        # Characters that typically follow a string value in JSON
        terminators = [',', '}', ']', ':']
        
        first_char = remaining[0]
        
        # If followed by a terminator, this is likely the end of the string
        if first_char in terminators:
            return True
        
        # Special case: if followed by another quote immediately, it's likely not a terminator
        # unless there's clear JSON structure after
        if first_char == '"':
            # Look further ahead to see if there's JSON structure
            next_remaining = remaining[1:].lstrip()
            if next_remaining and next_remaining[0] in terminators:
                return False  # This is likely a quote within the string
        
        # Additional heuristic: if followed by text that looks like it continues the sentence,
        # it's probably not a terminator
        if first_char.isalpha() or first_char in '.,;:!?':
            return False
        
        # If followed by whitespace and then a terminator, it's likely the end
        if remaining.lstrip() and remaining.lstrip()[0] in terminators:
            return True
        
        # Default to not terminating (safer to escape)
        return False

    def _is_string_terminator(self, json_str: str, quote_pos: int) -> bool:
        """Legacy method - kept for compatibility."""
        return self._is_string_terminator_smart(json_str, quote_pos)
    
    def _fix_control_characters(self, json_str: str) -> str:
        """Enhanced fix for unescaped control characters in string values."""
        
        # First apply enhanced control character fixing
        fixed = self._fix_control_characters_enhanced(json_str)
        
        # Then apply pattern-based fixing for string values
        def fix_string_content(match):
            key = match.group(1)
            value = match.group(2)
            
            # Fix common control characters
            value = value.replace('\n', '\\n')
            value = value.replace('\r', '\\r')
            value = value.replace('\t', '\\t')
            value = value.replace('\b', '\\b')
            value = value.replace('\f', '\\f')
            
            return f'"{key}": "{value}"'
        
        # Pattern to match key-value pairs with string values
        pattern = r'"([^"]+)":\s*"([^"]*(?:\\.[^"]*)*)"'
        
        try:
            fixed = re.sub(pattern, fix_string_content, fixed, flags=re.DOTALL)
            return fixed
        except Exception as e:
            logger.debug(f"Control character fix failed: {e}")
            return fixed
    
    def _fix_control_characters_enhanced(self, json_str: str) -> str:
        """Enhanced control character fixing with comprehensive edge case handling."""
        
        # Fix all control characters, including rare ones that can cause JSON parsing issues
        control_chars = {
            '\x00': '\\u0000', '\x01': '\\u0001', '\x02': '\\u0002',
            '\x03': '\\u0003', '\x04': '\\u0004', '\x05': '\\u0005',
            '\x06': '\\u0006', '\x07': '\\u0007', '\x08': '\\b',
            '\x09': '\\t', '\x0a': '\\n', '\x0b': '\\u000b',
            '\x0c': '\\f', '\x0d': '\\r', '\x0e': '\\u000e',
            '\x0f': '\\u000f', '\x10': '\\u0010', '\x11': '\\u0011',
            '\x12': '\\u0012', '\x13': '\\u0013', '\x14': '\\u0014',
            '\x15': '\\u0015', '\x16': '\\u0016', '\x17': '\\u0017',
            '\x18': '\\u0018', '\x19': '\\u0019', '\x1a': '\\u001a',
            '\x1b': '\\u001b', '\x1c': '\\u001c', '\x1d': '\\u001d',
            '\x1e': '\\u001e', '\x1f': '\\u001f'
        }
        
        # Apply control character fixes
        for char, replacement in control_chars.items():
            if char in json_str:
                json_str = json_str.replace(char, replacement)
                logger.debug(f"Fixed control character: {repr(char)} -> {replacement}")
        
        return json_str
    
    def _final_json_cleanup(self, json_str: str) -> str:
        """Final cleanup and validation."""
        # Remove any duplicate escaping that might have been introduced
        fixed = json_str.replace('\\\\n', '\\n')
        fixed = fixed.replace('\\\\t', '\\t')
        fixed = fixed.replace('\\\\r', '\\r')
        
        # Fix any double-escaped quotes that aren't needed
        fixed = re.sub(r'\\\\(")', r'\1', fixed)
        
        return fixed
    
    def _legacy_json_cleaning(self, json_str: str) -> str:
        """Legacy JSON cleaning methods as fallback."""
        # Method 1: Try to fix unescaped newlines and tabs within string values
        try:
            def escape_string_content(match):
                content = match.group(1)
                content = content.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')
                return f'"{content}"'
            
            string_pattern = r'"([^"\\]*(\\.[^"\\]*)*)"'
            cleaned = re.sub(string_pattern, escape_string_content, json_str)
            
            json.loads(cleaned)
            logger.debug("Successfully cleaned JSON using legacy string escaping method")
            return cleaned
            
        except (json.JSONDecodeError, Exception) as e:
            logger.debug(f"Legacy string escaping method failed: {e}")
        
        # Method 2: More aggressive cleaning
        try:
            cleaned = json_str.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')
            json.loads(cleaned)
            logger.debug("Successfully cleaned JSON using legacy character replacement method")
            return cleaned
            
        except (json.JSONDecodeError, Exception) as e:
            logger.debug(f"Legacy character replacement method failed: {e}")
        
        # Method 3: Whitespace normalization
        try:
            cleaned = re.sub(r'\s+', ' ', json_str.strip())
            cleaned = cleaned.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
            
            json.loads(cleaned)
            logger.debug("Successfully cleaned JSON using legacy whitespace normalization")
            return cleaned
            
        except (json.JSONDecodeError, Exception) as e:
            logger.debug(f"Legacy whitespace normalization method failed: {e}")
        
        # If all methods fail, return the original string
        logger.warning("All JSON cleaning methods failed, returning original string")
        return json_str
    
    def _validate_json_structure(self, data: Dict[str, Any]) -> bool:
        """
        Validate that the JSON has the expected structure.
        
        Args:
            data: Parsed JSON data
            
        Returns:
            True if structure is valid
        """
        # More flexible validation - only require 'answer' field as minimum
        # Other fields can be optional and will be filled with defaults
        if not isinstance(data, dict):
            return False
        
        # At minimum, we need an 'answer' field or 'response_type' field
        has_answer = 'answer' in data and isinstance(data['answer'], str) and data['answer'].strip()
        has_response_type = 'response_type' in data
        
        # Accept if we have either a meaningful answer or response_type
        return has_answer or has_response_type
    
    def _enhance_structured_data(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance and validate structured data.
        
        Args:
            data: Parsed structured data
            metadata: Additional metadata
            
        Returns:
            Enhanced structured data
        """
        # Ensure all required fields are present with defaults
        enhanced = {
            'response_type': data.get('response_type', 'document_based'),
            'answer': data.get('answer', ''),
            'confidence': self._process_confidence_data(data.get('confidence', {})),
            'sources': self._process_sources_data(data.get('sources', [])),
            'key_points': data.get('key_points', []),
            'metadata': self._merge_metadata(data.get('metadata', {}), metadata),
            'follow_up_questions': data.get('follow_up_questions', []),
            'related_topics': data.get('related_topics', []),
            'warnings': data.get('warnings', [])
        }
        
        return enhanced
    
    def _process_confidence_data(self, confidence_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and validate confidence data.
        
        Args:
            confidence_data: Raw confidence data
            
        Returns:
            Processed confidence data
        """
        score = confidence_data.get('score', 0.7)
        
        # Ensure score is in valid range
        if not isinstance(score, (int, float)) or score < 0 or score > 1:
            score = 0.7
        
        level = confidence_data.get('level', get_confidence_level(score).value)
        rationale = confidence_data.get('rationale', 'Nivel de confianza estimado basado en la información disponible.')
        
        # Process factors
        factors = confidence_data.get('factors', {})
        processed_factors = self._process_confidence_factors(factors, score)
        
        return {
            'score': score,
            'level': level,
            'rationale': rationale,
            'factors': processed_factors
        }
    
    def _process_confidence_factors(self, factors: Dict[str, Any], total_score: float) -> Dict[str, Any]:
        """
        Process confidence factors and ensure they're valid.
        
        Args:
            factors: Raw factors data
            total_score: Total confidence score
            
        Returns:
            Processed factors
        """
        default_factors = {
            'information_quality': {
                'score': int(total_score * 25),
                'explanation': 'Calidad de la información recuperada'
            },
            'query_coverage': {
                'score': int(total_score * 25),
                'explanation': 'Cobertura de la consulta'
            },
            'source_consistency': {
                'score': int(total_score * 20),
                'explanation': 'Consistencia entre fuentes'
            },
            'specificity': {
                'score': int(total_score * 20),
                'explanation': 'Especificidad y detalle de la respuesta'
            }
        }
        
        # Use provided factors or defaults
        processed = {}
        for factor_name, default_data in default_factors.items():
            if factor_name in factors:
                factor_data = factors[factor_name]
                processed[factor_name] = {
                    'score': factor_data.get('score', default_data['score']),
                    'explanation': factor_data.get('explanation', default_data['explanation'])
                }
            else:
                processed[factor_name] = default_data
        
        return processed
    
    def _process_sources_data(self, sources_data: list) -> list:
        """
        Process and validate sources data.
        
        Args:
            sources_data: Raw sources data
            
        Returns:
            Processed sources list
        """
        processed_sources = []
        
        for source in sources_data:
            if isinstance(source, dict):
                processed_source = {
                    'id': source.get('id', ''),
                    'title': source.get('title', 'Documento sin título'),
                    'relevance_score': source.get('relevance_score', 0.0),
                    'excerpt': source.get('excerpt', ''),
                    'metadata': source.get('metadata', {})
                }
                processed_sources.append(processed_source)
        
        return processed_sources
    
    def _merge_metadata(self, response_metadata: Dict[str, Any], additional_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge response metadata with additional metadata.
        
        Args:
            response_metadata: Metadata from structured response
            additional_metadata: Additional metadata to merge
            
        Returns:
            Merged metadata
        """
        merged = {
            'model': additional_metadata.get('model', 'claude-3-haiku'),
            'tokens_used': additional_metadata.get('usage', {'input': 0, 'output': 0}),
            'processing_time': additional_metadata.get('processing_time', 0.0),
            'search_strategy': additional_metadata.get('search_strategy', 'hybrid_search'),
            'documents_retrieved': additional_metadata.get('documents_retrieved', 0),
            'cache_metrics': additional_metadata.get('cache_metrics')
        }
        
        # Override with response metadata if provided
        merged.update(response_metadata)
        
        return merged
    
    def _fallback_parse(self, response: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback parsing for non-structured responses.
        
        Args:
            response: Raw response text
            metadata: Additional metadata
            
        Returns:
            Structured data created from legacy parsing
        """
        logger.info("Using fallback parsing for legacy response format")
        
        # Extract confidence using legacy method
        confidence_data = self._extract_legacy_confidence(response)
        
        # Extract sources from citations in text
        sources = self._extract_legacy_sources(response, metadata.get('sources', []))
        
        # Extract key points (simple heuristic)
        key_points = self._extract_key_points(response)
        
        # Create structured response
        fallback_data = {
            'response_type': 'document_based',
            'answer': response,
            'confidence': confidence_data,
            'sources': sources,
            'key_points': key_points,
            'metadata': self._merge_metadata({}, metadata),
            'follow_up_questions': [],
            'related_topics': [],
            'warnings': ['Esta respuesta fue procesada con el formato legacy. Para mejor estructuración, actualice el sistema.']
        }
        
        return fallback_data
    
    def _extract_legacy_confidence(self, response: str) -> Dict[str, Any]:
        """
        Extract confidence using legacy regex patterns.
        
        Args:
            response: Response text
            
        Returns:
            Confidence data
        """
        # Look for CONFIDENCE: XX% pattern
        pattern = r'CONFIDENCE:\s*(\d+)%\s*(.*?)(?=\n\n|\Z)'
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        
        if match:
            confidence_percent = int(match.group(1))
            rationale = match.group(2).strip() if match.group(2) else ""
            
            if 0 <= confidence_percent <= 100:
                score = confidence_percent / 100.0
                level = get_confidence_level(score).value
                
                return self._process_confidence_data({
                    'score': score,
                    'level': level,
                    'rationale': rationale or f'Confianza del {confidence_percent}% basada en la información disponible.'
                })
        
        # Default confidence
        return self._process_confidence_data({
            'score': 0.7,
            'level': 'medium',
            'rationale': 'Nivel de confianza estimado (no se encontró evaluación explícita).'
        })
    
    def _extract_legacy_sources(self, response: str, original_sources: list) -> list:
        """
        Extract sources from citation markers in response.
        
        Args:
            response: Response text
            original_sources: Original sources from search
            
        Returns:
            List of source references
        """
        # Find citation markers [N]
        citations = re.findall(r'\[(\d+)\]', response)
        cited_sources = []
        
        for citation in set(citations):
            try:
                index = int(citation) - 1
                if 0 <= index < len(original_sources):
                    source = original_sources[index]
                    
                    # Use better relevance score - prioritize actual search scores
                    relevance_score = source.get('rrf_score', source.get('score', 0.0))
                    
                    # If score is still very low, use a more realistic score based on citation usage
                    if relevance_score < 0.1:
                        # Sources that are cited are inherently more relevant
                        # Use a score between 0.6-0.9 based on citation frequency
                        citation_count = response.count(f'[{citation}]')
                        relevance_score = min(0.6 + (citation_count * 0.1), 0.9)
                    
                    cited_sources.append({
                        'id': f'[{citation}]',
                        'title': source.get('source_file', source.get('source', source.get('title', 'Documento sin título'))),
                        'relevance_score': relevance_score,
                        'excerpt': source.get('text', '')[:200] + '...' if source.get('text') else '',
                        'metadata': source.get('metadata', {})
                    })
            except (ValueError, IndexError):
                continue
        
        return cited_sources
    
    def _extract_key_points(self, response: str) -> list:
        """
        Extract key points from response using simple heuristics.
        
        Args:
            response: Response text
            
        Returns:
            List of key points
        """
        # Simple heuristic: look for sentences with key indicators
        sentences = re.split(r'[.!?]+', response)
        key_points = []
        
        key_indicators = [
            'principalmente', 'específicamente', 'importante', 'clave',
            'fundamental', 'esencial', 'destacar', 'señalar'
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and len(sentence) < 150:
                if any(indicator in sentence.lower() for indicator in key_indicators):
                    # Clean up the sentence
                    clean_sentence = re.sub(r'\[\d+\]', '', sentence).strip()
                    if clean_sentence and clean_sentence not in key_points:
                        key_points.append(clean_sentence)
        
        # Limit to top 5 key points
        return key_points[:5]
    
    def format_for_display(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format parsed data for display in the chat interface.
        
        Args:
            parsed_data: Parsed structured data
            
        Returns:
            Data formatted for display
        """
        confidence = parsed_data.get('confidence', {})
        confidence_score = confidence.get('score', 0.7)
        confidence_level = confidence.get('level', 'medium')
        
        # Map confidence level to emoji and display text
        level_mapping = {
            'very_high': {'emoji': '🟢', 'text': 'muy alta'},
            'high': {'emoji': '🟢', 'text': 'alta'},
            'medium': {'emoji': '🟡', 'text': 'media'},
            'low': {'emoji': '🟠', 'text': 'baja'},
            'very_low': {'emoji': '🔴', 'text': 'muy baja'}
        }
        
        level_info = level_mapping.get(confidence_level, level_mapping['medium'])
        
        return {
            'answer': parsed_data.get('answer', ''),
            'confidence_score': confidence_score,
            'confidence_level': level_info['text'],
            'confidence_emoji': level_info['emoji'],
            'confidence_rationale': confidence.get('rationale', ''),
            'sources': parsed_data.get('sources', []),
            'key_points': parsed_data.get('key_points', []),
            'follow_up_questions': parsed_data.get('follow_up_questions', []),
            'related_topics': parsed_data.get('related_topics', []),
            'warnings': parsed_data.get('warnings', []),
            'metadata': parsed_data.get('metadata', {}),
            'response_type': parsed_data.get('response_type', 'document_based')
        }
