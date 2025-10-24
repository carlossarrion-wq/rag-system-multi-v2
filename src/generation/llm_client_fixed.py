"""
LLM Client for Claude 4 via AWS Bedrock - EC2 Complete Version with Fixed Image Support
Handles answer generation with confidence scoring and full multimodal functionality.
"""

import json
from typing import List, Dict, Any, Optional
import boto3
from botocore.exceptions import ClientError
import yaml
import logging
import re
import os
from datetime import datetime
from src.utils.connection_manager import ConnectionManager
from src.generation.structured_response_parser import StructuredResponseParser
from src.generation.image_validator import ImageValidator
from src.generation.image_summary_retriever import ImageSummaryRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMClient:
    """
    LLM client for Claude 4 via AWS Bedrock - Complete EC2 Version with Fixed Image Support.

    Features:
    - Claude 4 Sonnet integration via Bedrock
    - Confidence scoring extraction
    - RAG-optimized prompting with multi-application support
    - Streaming support with confidence
    - FIXED: Multimodal support (images) in both streaming and non-streaming methods
    - Full compatibility with EC2 infrastructure
    """

    def __init__(self, config_path: str = "config/multi_app_config.yaml"):
        """
        Initialize LLM Client.

        Args:
            config_path: Path to AWS configuration file
        """
        self.config = self._load_config(config_path)
        self.connection_manager = ConnectionManager(config_path)
        self.bedrock_client = self.connection_manager.get_bedrock_client()

        # Model configuration
        self.model_id = self.config['bedrock']['llm_model']
        self.max_tokens = self.config['bedrock'].get('max_tokens', 4096)  # Output tokens
        self.max_input_tokens = self.config['bedrock'].get('max_input_tokens', 180000)  # Input tokens
        self.temperature = self.config['bedrock'].get('temperature', 0.7)
        self.top_p = self.config['bedrock'].get('top_p', 0.9)

        # Prompt caching configuration
        self.prompt_caching_config = self.config['bedrock'].get('prompt_caching', {})
        self.caching_enabled = self.prompt_caching_config.get('enabled', False)
        self.default_ttl = self.prompt_caching_config.get('default_ttl', '5m')
        self.cache_system_prompt = self.prompt_caching_config.get('cache_system_prompt', True)
        self.cache_context_documents = self.prompt_caching_config.get('cache_context_documents', True)
        self.min_tokens_to_cache = self.prompt_caching_config.get('min_tokens_to_cache', 2048)
        self.track_metrics = self.prompt_caching_config.get('track_metrics', True)

        # Initialize structured response parser
        self.response_parser = StructuredResponseParser()
        
        # Initialize image validator
        self.image_validator = ImageValidator()
        
        # Initialize image summary retriever (will be set by multi-app system)
        self.image_summary_retriever = None
        
        # Initialize document enhancer for context reduction (will be set by multi-app system)
        self.document_enhancer = None

        logger.info(f"LLMClient initialized with model: {self.model_id}")
        if self.caching_enabled:
            logger.info(f"Prompt caching enabled with TTL: {self.default_ttl}")

    def _dump_llm_request_and_response(
        self, 
        query: str, 
        raw_answer: str, 
        full_request_body: Dict[str, Any] = None,
        request_info: Dict[str, Any] = None,
        context_data: Dict[str, Any] = None
    ) -> None:
        """
        Dump complete LLM request and response to file for comprehensive analysis.
        ENHANCED: Now includes ALL content sent to LLM including document chunks.
        
        Args:
            query: User query
            raw_answer: Raw LLM response
            full_request_body: Complete request body sent to Bedrock
            request_info: Additional request information
            context_data: Context and document data
        """
        try:
            # Create logs directory if it doesn't exist - try both local and EC2 paths
            logs_dirs = ["logs", "/home/ec2-user/rag-system-multi/logs"]
            logs_dir = None
            
            for dir_path in logs_dirs:
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    # Test write access
                    test_file = os.path.join(dir_path, "test_write.tmp")
                    with open(test_file, 'w') as f:
                        f.write("test")
                    os.remove(test_file)
                    logs_dir = dir_path
                    break
                except:
                    continue
            
            if not logs_dir:
                logger.error("Could not create or access logs directory")
                return
            
            # Generate timestamp and filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            filename = f"llm_complete_dump_{timestamp}.txt"
            filepath = os.path.join(logs_dir, filename)
            
            # Extract system prompt and user content from request body
            system_prompt = ""
            user_content = ""
            images_info = []
            
            if full_request_body:
                system_prompt = full_request_body.get('system', 'No system prompt')
                messages = full_request_body.get('messages', [])
                
                for message in messages:
                    if message.get('role') == 'user':
                        content = message.get('content', '')
                        
                        # Handle multimodal content (list of content blocks)
                        if isinstance(content, list):
                            for block in content:
                                if block.get('type') == 'text':
                                    user_content = block.get('text', '')
                                elif block.get('type') == 'image':
                                    source = block.get('source', {})
                                    images_info.append({
                                        'media_type': source.get('media_type', 'unknown'),
                                        'data_length': len(source.get('data', '')),
                                        'type': source.get('type', 'unknown')
                                    })
                        else:
                            # Simple text content
                            user_content = content
            
            # Prepare comprehensive dump content
            dump_content = f"""
{'='*80}
=== COMPLETE LLM REQUEST AND RESPONSE DUMP ===
{'='*80}
Timestamp: {datetime.now().isoformat()}
Model: {self.model_id}
Log File: {filename}

{'='*80}
=== USER QUERY ===
{'='*80}
{query}

{'='*80}
=== REQUEST METADATA ===
{'='*80}
{json.dumps(request_info or {}, indent=2, ensure_ascii=False)}

{'='*80}
=== CONTEXT DATA ===
{'='*80}
"""
            
            # Add context data if available
            if context_data:
                dump_content += f"""
Sources Count: {context_data.get('sources_count', 0)}
Images Count: {context_data.get('images_count', 0)}
Context Length: {context_data.get('context_length', 0)} characters
Total Documents: {context_data.get('total_documents', 0)}

Document Sources:
"""
                sources = context_data.get('sources', [])
                for i, source in enumerate(sources, 1):
                    dump_content += f"""
--- SOURCE {i} ---
ID: {source.get('id', 'N/A')}
Document Title: {source.get('document_title', 'N/A')}
File Name: {source.get('file_name', 'N/A')}
Doc ID: {source.get('doc_id', 'N/A')}
Chunk ID: {source.get('chunk_id', 'N/A')}
Score: {source.get('score', 'N/A')}
Content Length: {len(source.get('text', ''))} characters
Content Preview: {source.get('text', '')[:200]}{'...' if len(source.get('text', '')) > 200 else ''}
Metadata: {json.dumps(source.get('metadata', {}), indent=2, ensure_ascii=False)}
"""
            else:
                dump_content += "No context data available\n"
            
            dump_content += f"""
{'='*80}
=== SYSTEM PROMPT SENT TO LLM ===
{'='*80}
{system_prompt}

{'='*80}
=== COMPLETE USER CONTENT SENT TO LLM ===
{'='*80}
{user_content}
"""
            
            # Add image information if present
            if images_info:
                dump_content += f"""
{'='*80}
=== IMAGES SENT TO LLM ===
{'='*80}
Total Images: {len(images_info)}

"""
                for i, img_info in enumerate(images_info, 1):
                    dump_content += f"""Image {i}:
  - Media Type: {img_info['media_type']}
  - Data Length: {img_info['data_length']} bytes
  - Source Type: {img_info['type']}

"""
            
            dump_content += f"""
{'='*80}
=== COMPLETE REQUEST BODY (JSON) ===
{'='*80}
{json.dumps(full_request_body or {}, indent=2, ensure_ascii=False)}

{'='*80}
=== RAW LLM RESPONSE ===
{'='*80}
{raw_answer}

{'='*80}
=== RESPONSE ANALYSIS ===
{'='*80}
Response Length: {len(raw_answer)} characters
Response Lines: {len(raw_answer.splitlines())} lines
Starts with JSON: {raw_answer.strip().startswith('{')}
Ends with JSON: {raw_answer.strip().endswith('}')}
Contains JSON: {'{' in raw_answer and '}' in raw_answer}

{'='*80}
=== TOKEN ANALYSIS ===
{'='*80}
"""
            
            # Add token analysis
            if request_info:
                usage = request_info.get('usage', {})
                dump_content += f"""Input Tokens: {usage.get('input_tokens', 'N/A')}
Output Tokens: {usage.get('output_tokens', 'N/A')}
Total Tokens: {usage.get('input_tokens', 0) + usage.get('output_tokens', 0)}

System Prompt Length: {request_info.get('system_prompt_length', 'N/A')} chars
Context Length: {request_info.get('context_length', 'N/A')} chars
Sources Count: {request_info.get('sources_count', 'N/A')}
Images Count: {request_info.get('images_count', 'N/A')}

Cache Metrics: {json.dumps(request_info.get('cache_metrics', {}), indent=2)}
"""
            
            dump_content += f"""
{'='*80}
=== END COMPLETE DUMP ===
{'='*80}
"""
            
            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(dump_content)
            
            logger.info(f"Complete LLM request/response dumped to: {filepath}")
            logger.info(f"Dump includes: System prompt, User content, {len(images_info)} images, Full request body, Response")
            
        except Exception as e:
            logger.error(f"Failed to dump complete LLM request/response: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    def _dump_llm_response(self, query: str, raw_answer: str, request_info: Dict[str, Any] = None) -> None:
        """
        Legacy method for backward compatibility.
        Now calls the enhanced _dump_llm_request_and_response method.
        """
        self._dump_llm_request_and_response(
            query=query,
            raw_answer=raw_answer,
            request_info=request_info
        )

    def _load_config(self, config_path: str) -> Dict:
        """Load AWS configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _extract_confidence_score_and_rationale(self, answer: str) -> Dict[str, Any]:
        """
        Extract confidence score and rationale from LLM answer.

        Args:
            answer: The LLM's answer text

        Returns:
            Dictionary with confidence_score (0.0-1.0) and rationale text
        """
        
        # Look for pattern: CONFIDENCE: XX% followed by rationale
        # More flexible pattern that captures everything after CONFIDENCE: XX%
        pattern = r'CONFIDENCE:\s*(\d+)%\s*(.*?)(?=\n\n|\Z)'
        match = re.search(pattern, answer, re.IGNORECASE | re.DOTALL)

        if match:
            confidence_percent = int(match.group(1))
            rationale = match.group(2).strip() if match.group(2) else ""
            
            
            # Validate range
            if 0 <= confidence_percent <= 100:
                confidence_decimal = confidence_percent / 100.0
                return {
                    'confidence_score': confidence_decimal,
                    'confidence_rationale': rationale
                }
            else:
                return {'confidence_score': None, 'confidence_rationale': ''}
        else:
            # Try alternative patterns
            alt_patterns = [
                r'CONFIDENCE:\s*(\d+)%',
                r'confidence:\s*(\d+)%',
                r'Confianza:\s*(\d+)%'
            ]
            for alt_pattern in alt_patterns:
                alt_match = re.search(alt_pattern, answer, re.IGNORECASE)
                if alt_match:
                    confidence_percent = int(alt_match.group(1))
                    if 0 <= confidence_percent <= 100:
                        return {
                            'confidence_score': confidence_percent / 100.0,
                            'confidence_rationale': ''
                        }
            
            return {'confidence_score': None, 'confidence_rationale': ''}

    def _extract_confidence_score(self, answer: str) -> Optional[float]:
        """
        Extract confidence score from LLM answer (backward compatibility).

        Args:
            answer: The LLM's answer text

        Returns:
            Confidence score (0.0-1.0) or None if not found
        """
        result = self._extract_confidence_score_and_rationale(answer)
        return result['confidence_score']

    def _build_rag_prompt(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Build RAG prompt with query and context.

        Args:
            query: User query
            context: Retrieved context
            system_prompt: Optional system prompt

        Returns:
            Formatted prompt
        """
        if system_prompt is None:
            system_prompt = """Eres un asistente de IA útil que responde preguntas basándose en el contexto proporcionado.
Tu tarea es proporcionar respuestas precisas y detalladas usando ÚNICAMENTE la información del contexto.
Si el contexto no contiene suficiente información para responder la pregunta, dilo claramente.
Siempre cita qué partes del contexto usaste para formular tu respuesta."""

        prompt = f"""{system_prompt}

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""

        return prompt

    def _build_json_structured_prompt(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        memory_context: Optional[str] = None,
        sources: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Build a clean JSON-structured prompt with optimized architecture.
        IMPROVED: Eliminates redundancy by using simplified structure with direct system prompt injection.
        
        Args:
            query: User query
            context: Retrieved context
            system_prompt: System prompt instructions (Darwin/SAP specific)
            memory_context: Conversation memory
            sources: Source documents metadata
            
        Returns:
            JSON-structured prompt with optimized architecture
        """
        # Clean, non-redundant structure - system prompt goes directly into tu_tarea
        structured_input = {
            "tu_tarea": system_prompt or "Analiza los documentos proporcionados y responde en formato JSON",
            "conversation_context": {
                "previous_interactions": memory_context or "No previous context",
                "current_session": "active"
            },
            "document_sources": [
                {
                    "source_id": source.get('id', f'[{i+1}]'),
                    "document_title": source.get('title', source.get('file_name', source.get('source_file', 'Documento sin título'))),
                    "file_name": source.get('file_name', source.get('title', source.get('source_file', 'N/A'))),
                    "source_file": source.get('source_file', source.get('title', source.get('file_name', 'N/A'))),
                    "content": source.get('text', ''),
                    "metadata": source.get('metadata', {}),
                    "relevance_score": source.get('score', 0.0)
                }
                for i, source in enumerate(sources or [])
            ],
            "user_query": {
                "question": query,
                "type": "document_analysis",
                "requires_citations": True
            },
            "response_requirements": {
                "format": "json",
                "include_sources": True,
                "include_confidence": True,
                "cite_documents": True,
                "language": "spanish"
            }
        }
        
        # Create the optimized JSON-structured prompt
        json_prompt = f"""DATOS_DE_ENTRADA:
{json.dumps(structured_input, indent=2, ensure_ascii=False)}

TAREA: Analiza ÚNICAMENTE los documentos proporcionados en "document_sources" y responde en formato JSON siguiendo exactamente esta estructura.

IMPORTANTE: Solo puedes usar información de los documentos listados en "document_sources". NO inventes ni uses información de fuentes externas.

{{
  "response_type": "document_based",
  "answer": "Tu respuesta detallada con citas [N] usando SOLO los documentos proporcionados",
  "confidence": {{
    "score": 0.85,
    "level": "high",
    "rationale": "Explicación del nivel de confianza basada en los documentos disponibles"
  }},
  "sources": [
    {{
      "id": "[1]",
      "title": "Título del documento (debe coincidir con document_sources)",
      "relevance_score": 0.90,
      "excerpt": "Fragmento relevante del documento proporcionado"
    }}
  ],
  "key_points": ["Punto clave 1", "Punto clave 2"],
  "related_topics": ["Tema relacionado 1", "Tema relacionado 2"]
}}

RESPUESTA:"""

        return json_prompt

    def _get_system_prompt(self) -> str:
        """
        Get system prompt. This method can be overridden by multi-app system.
        Default fallback prompt for standalone usage.
        """
        return """Eres un asistente de IA útil que responde preguntas basándose en el contexto proporcionado.
Tu tarea es proporcionar respuestas precisas y detalladas usando ÚNICAMENTE la información del contexto.
Si el contexto no contiene suficiente información para responder la pregunta, dilo claramente.
Siempre cita qué partes del contexto usaste para formular tu respuesta.

Al final de tu respuesta, incluye una evaluación de confianza en el formato:
CONFIANZA: XX%

Donde XX es un número entre 0 y 100 que representa tu confianza en la respuesta."""

    def _extract_images_from_results(self, expanded_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract and validate image data from search results.
        MODIFIED: Now returns empty list to prevent raw image sending during queries.
        Images are processed during ingestion to generate detailed text descriptions.
        
        Args:
            expanded_results: Search results with metadata
            
        Returns:
            Empty list - raw images are never sent during queries
        """
        # Count images for logging purposes
        image_count = 0
        for result in expanded_results:
            metadata = result.get('metadata', {})
            if metadata.get('has_image') and metadata.get('image_base64'):
                image_count += 1
        
        if image_count > 0:
            logger.info(f"Found {image_count} images in search results - using detailed descriptions generated during ingestion instead of raw images")
        
        # Return empty list - detailed descriptions from ingestion are already in document text
        return []

    def _build_multimodal_content(self, prompt: str, images_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Build multimodal message content with text and images.
        ENHANCED: Uses validated image data with proper media types.
        
        Args:
            prompt: Text prompt
            images_data: List of validated image data
            
        Returns:
            List of content blocks for Claude
        """
        message_content = []
        
        # Add images first if present
        if images_data:
            for img_data in images_data:
                # Use validated media_type from image validator
                media_type = img_data.get('media_type', 'image/png')
                
                # Ensure base64 data is clean
                base64_data = img_data.get('image_base64', '')
                if not base64_data:
                    logger.warning(f"Empty base64 data for image {img_data.get('image_id', 'unknown')}")
                    continue
                
                try:
                    message_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64_data
                        }
                    })
                    
                    size_info = f" ({img_data.get('size_bytes', 0)} bytes)" if img_data.get('size_bytes') else ""
                    logger.info(f"Added validated image {img_data.get('image_id', 'unknown')} to message with media_type: {media_type}{size_info}")
                    
                except Exception as e:
                    logger.error(f"Error adding image {img_data.get('image_id', 'unknown')} to message: {e}")
                    continue
        
        # Add text prompt
        message_content.append({
            "type": "text",
            "text": prompt
        })
        
        return message_content

    def _extract_cache_metrics(self, usage_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract cache metrics from usage data.
        
        Args:
            usage_data: Usage data from Claude response
            
        Returns:
            Cache metrics dictionary
        """
        if not self.caching_enabled or not usage_data:
            return {}
        
        cache_metrics = {}
        
        # Extract cache-related metrics
        cache_creation_tokens = usage_data.get('cache_creation_input_tokens', 0)
        cache_read_tokens = usage_data.get('cache_read_input_tokens', 0)
        regular_input_tokens = usage_data.get('input_tokens', 0)
        
        if cache_creation_tokens > 0 or cache_read_tokens > 0:
            cache_metrics = {
                'cache_creation_tokens': cache_creation_tokens,
                'cache_read_tokens': cache_read_tokens,
                'regular_input_tokens': regular_input_tokens,
                'cache_hit': cache_read_tokens > 0,
                'cache_miss': cache_creation_tokens > 0,
                'cache_efficiency': cache_read_tokens / (cache_read_tokens + regular_input_tokens) if (cache_read_tokens + regular_input_tokens) > 0 else 0
            }
        
        return cache_metrics

    def _log_cache_metrics(self, cache_metrics: Dict[str, Any]) -> None:
        """
        Log cache performance metrics.
        
        Args:
            cache_metrics: Cache metrics to log
        """
        if not cache_metrics:
            return
        
        cache_status = "HIT" if cache_metrics.get('cache_hit') else "MISS"
        efficiency = cache_metrics.get('cache_efficiency', 0) * 100
        
        logger.info(f"Cache {cache_status}: "
                   f"Read={cache_metrics.get('cache_read_tokens', 0)} tokens, "
                   f"Created={cache_metrics.get('cache_creation_tokens', 0)} tokens, "
                   f"Efficiency={efficiency:.1f}%")

    def _create_cache_control(self, ttl: Optional[str] = None) -> Dict[str, Any]:
        """
        Create cache control configuration.
        
        Args:
            ttl: Time to live ("5m" or "1h"), uses default if None
            
        Returns:
            Cache control dictionary
        """
        if not self.caching_enabled:
            return {}
        
        ttl = ttl or self.default_ttl
        return {
            "type": "ephemeral",
            "ttl": ttl
        }

    def _should_cache_content(self, content: str) -> bool:
        """
        Determine if content should be cached based on token count estimation.
        
        Args:
            content: Content to evaluate
            
        Returns:
            True if content should be cached
        """
        if not self.caching_enabled:
            return False
        
        # Rough estimation: 1 token ≈ 4 characters for English text
        estimated_tokens = len(content) // 4
        return estimated_tokens >= self.min_tokens_to_cache

    def _build_system_message_with_cache(self, system_prompt: str) -> Any:
        """
        Build system message with cache control if enabled.
        
        Args:
            system_prompt: System prompt text
            
        Returns:
            String for simple system prompt or list of blocks for cached system prompt
        """
        # For now, disable caching for system messages to avoid API validation errors
        # AWS Bedrock's cache_control implementation may have specific requirements
        # that are not fully compatible with the current format
        logger.info("Using simple system prompt format (caching temporarily disabled)")
        return system_prompt

    def generate_with_citations(
        self,
        query: str,
        expanded_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate answer with source citations and confidence scoring.
        FIXED: Now properly handles images in non-streaming mode.

        Args:
            query: User query
            expanded_results: Results with expanded context

        Returns:
            Dictionary with answer, confidence_score, and metadata
        """
        # Format context with source markers and collect images
        context_parts = []
        sources = []
        images_data = self._extract_images_from_results(expanded_results)

        for i, result in enumerate(expanded_results, 1):
            source_id = f"[{i}]"
            context_parts.append(f"{source_id} {result['text']}")

            sources.append({
                'id': source_id,
                'doc_id': result.get('doc_id'),
                'chunk_id': result.get('chunk_id'),
                'score': result.get('rrf_score', result.get('score')),
                'metadata': result.get('metadata', {}),
                # FIXED: Include title and filename fields from expanded_results
                'title': result.get('title', ''),
                'file_name': result.get('file_name', ''),
                'source_file': result.get('source_file', ''),
                'text': result.get('text', '')
            })

        context = "\n\n".join(context_parts)
        
        # Reduce context size when images are present to avoid token limit
        if images_data and hasattr(self, 'document_enhancer') and hasattr(self.document_enhancer, 'reduce_context_for_images'):
            original_length = len(context)
            context = self.document_enhancer.reduce_context_for_images(context, max_tokens=100000)
            logger.info(f"Context reduced from {original_length} to {len(context)} characters for image processing")
        logger.info(f"Formatted context with {len(sources)} sources, {len(images_data)} images")

        # If images are present, add note to context
        if images_data:
            context += f"\n\n[NOTA: Se han incluido {len(images_data)} imagen(es) visual(es) para análisis]"

        # Get system prompt (will be overridden by multi-app system)
        system_prompt = self._get_system_prompt()

        try:
            # NEW: Use JSON-structured approach to encourage consistent JSON responses
            logger.info("Using JSON-structured prompt approach for consistent JSON responses")
            
            # Prepare sources data for JSON structure
            sources_for_json = []
            for i, result in enumerate(expanded_results, 1):
                # Extract document title and filename from expanded_results
                document_title = (
                    result.get('title') or 
                    result.get('source_file') or 
                    result.get('file_name') or 
                    'Documento sin título'
                )
                
                file_name = (
                    result.get('source_file') or 
                    result.get('file_name') or 
                    result.get('title') or 
                    'N/A'
                )
                
                sources_for_json.append({
                    'id': f'[{i}]',
                    'text': result['text'],
                    'metadata': result.get('metadata', {}),
                    'score': result.get('rrf_score', result.get('score', 0.0)),
                    # FIXED: Properly extract and include document names
                    'title': document_title,
                    'file_name': file_name,
                    'source_file': result.get('source_file', document_title)
                })
            
            # Build JSON-structured prompt - FIXED: Explicitly disable memory context to prevent contamination
            json_structured_prompt = self._build_json_structured_prompt(
                query=query,
                context=context,
                system_prompt=system_prompt,
                memory_context=None,  # Explicitly disabled to prevent source contamination
                sources=sources_for_json
            )
            
            # Use minimal system prompt for JSON approach
            json_system_prompt = "Eres un asistente de análisis documental. Analiza la entrada JSON proporcionada y responde con un objeto JSON válido siguiendo la estructura especificada."
            system_messages = self._build_system_message_with_cache(json_system_prompt)

            # Build message content (text + optional images)
            if images_data:
                user_message_content = self._build_multimodal_content(json_structured_prompt, images_data)
            else:
                user_message_content = json_structured_prompt

            # Prepare Claude request with system messages and prompt caching
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "system": system_messages,
                "messages": [
                    {
                        "role": "user",
                        "content": user_message_content
                    }
                ]
            }

            # Call Bedrock with enhanced error handling
            logger.info(f"Generating answer with {self.model_id}... (Images: {len(images_data)})")
            
            # If images are present and we get validation errors, try fallback without images
            try:
                response = self.bedrock_client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(request_body),
                    contentType='application/json',
                    accept='application/json'
                )
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                error_message = e.response.get('Error', {}).get('Message', str(e))
                
                # Check if it's an image processing error and we have images
                if 'Could not process image' in error_message and images_data:
                    logger.warning(f"Image processing failed: {error_message}. Using image summaries instead...")
                    
                    # Log image validation stats for debugging
                    validation_stats = self.image_validator.get_stats()
                    logger.info(f"Image validation stats: {validation_stats}")
                    
                    # Get image summaries instead of discarding images
                    image_summaries_text = ""
                    if self.image_summary_retriever:
                        try:
                            # Extract metadata for image summary retrieval
                            images_metadata = []
                            for img_data in images_data:
                                metadata = {
                                    'source_file': img_data.get('source_id', '').replace('[', '').replace(']', ''),
                                    'chunk_id': img_data.get('image_id', ''),
                                    'image_id': img_data.get('image_id', ''),
                                    'image_context': img_data.get('image_context', '')
                                }
                                images_metadata.append(metadata)
                            
                            # Retrieve image summaries
                            summaries = self.image_summary_retriever.get_multiple_image_summaries(images_metadata)
                            image_summaries_text = self.image_summary_retriever.format_summaries_for_context(summaries)
                            logger.info(f"Retrieved {len(summaries)} image summaries as fallback")
                            
                        except Exception as summary_error:
                            logger.warning(f"Failed to retrieve image summaries: {summary_error}")
                            image_summaries_text = "\n\n=== INFORMACIÓN DE IMÁGENES ===\nSe detectaron imágenes en los documentos pero no se pudieron procesar directamente. Las imágenes contienen diagramas y contenido visual relevante para la consulta."
                    else:
                        image_summaries_text = "\n\n=== INFORMACIÓN DE IMÁGENES ===\nSe detectaron imágenes en los documentos pero no se pudieron procesar directamente. Las imágenes contienen diagramas y contenido visual relevante para la consulta."
                    
                    # Add image summaries to the context
                    enhanced_prompt = json_structured_prompt + image_summaries_text
                    
                    # Retry with image summaries instead of raw images
                    fallback_request_body = {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": self.max_tokens,
                        "temperature": self.temperature,
                        "system": system_messages,
                        "messages": [
                            {
                                "role": "user",
                                "content": enhanced_prompt  # Text with image summaries
                            }
                        ]
                    }
                    
                    logger.info("Retrying request with image summaries instead of raw images...")
                    
                    # Apply text-only context reduction if document enhancer is available
                    if hasattr(self, 'document_enhancer') and hasattr(self.document_enhancer, 'reduce_context_for_text_only'):
                        try:
                            original_length = len(enhanced_prompt)
                            reduced_prompt = self.document_enhancer.reduce_context_for_text_only(enhanced_prompt)
                            fallback_request_body["messages"][0]["content"] = reduced_prompt
                            logger.info(f"Context reduced for image summaries fallback: {original_length} -> {len(reduced_prompt)} chars")
                        except Exception as e:
                            logger.warning(f"Text-only context reduction failed: {e}")
                    
                    response = self.bedrock_client.invoke_model(
                        modelId=self.model_id,
                        body=json.dumps(fallback_request_body),
                        contentType='application/json',
                        accept='application/json'
                    )
                    
                    # Update images_data to reflect that summaries were used instead
                    images_data = []
                    logger.info("Successfully generated response using image summaries")
                else:
                    # Re-raise if it's not an image processing error
                    raise

            # Parse response
            response_body = json.loads(response['body'].read())

            # Extract answer from Claude response
            raw_answer = response_body['content'][0]['text']
            usage = {
                'input_tokens': response_body.get('usage', {}).get('input_tokens', 0),
                'output_tokens': response_body.get('usage', {}).get('output_tokens', 0)
            }
            stop_reason = response_body.get('stop_reason', 'unknown')

            # Extract cache metrics if available and tracking is enabled
            cache_metrics = self._extract_cache_metrics(response_body.get('usage', {}))

            # FIXED: Simplified and corrected source mapping
            # The problem was that sources and expanded_results have the same order (1:1 mapping)
            enhanced_sources = []
            logger.debug("Processing sources for metadata mapping")
            
            for i, source in enumerate(sources):
                # Direct mapping - sources[i] corresponds to expanded_results[i]
                expanded_result = expanded_results[i] if i < len(expanded_results) else None
                
                if expanded_result:
                    # Extract title and filename directly from expanded_result (which has the correct data)
                    document_title = (
                        expanded_result.get('title') or 
                        expanded_result.get('source_file') or 
                        expanded_result.get('file_name') or 
                        'Documento sin título'
                    )
                    
                    file_name = (
                        expanded_result.get('source_file') or 
                        expanded_result.get('file_name') or 
                        expanded_result.get('title') or 
                        'N/A'
                    )

                    
                    enhanced_source = {
                        'id': source.get('id', f'[{i+1}]'),
                        'doc_id': expanded_result.get('doc_id', 'N/A'),
                        'chunk_id': expanded_result.get('chunk_id', 'N/A'),
                        'score': source.get('score', expanded_result.get('score', 0.0)),
                        'metadata': expanded_result.get('metadata', {}),
                        'text': expanded_result.get('text', ''),
                        'document_title': document_title,
                        'file_name': file_name
                    }
                else:
                    # Fallback if no expanded_result
                    enhanced_source = {
                        'id': source.get('id', f'[{i+1}]'),
                        'doc_id': source.get('doc_id', 'N/A'),
                        'chunk_id': source.get('chunk_id', 'N/A'),
                        'score': source.get('score', 0.0),
                        'metadata': source.get('metadata', {}),
                        'text': source.get('text', ''),
                        'document_title': source.get('title', 'Documento sin título'),
                        'file_name': source.get('file_name', 'N/A')
                    }
                
                enhanced_sources.append(enhanced_source)
            
            logger.debug(f"Enhanced {len(enhanced_sources)} sources with proper filename mapping")
            
            context_data = {
                'sources_count': len(sources),
                'images_count': len(images_data),
                'context_length': len(context),
                'total_documents': len(expanded_results),
                'sources': enhanced_sources
            }

            # Dump complete LLM request and response for analysis
            request_info = {
                'model': self.model_id,
                'max_tokens': self.max_tokens,
                'temperature': self.temperature,
                'system_prompt_length': len(system_messages) if isinstance(system_messages, str) else len(str(system_messages)),
                'context_length': len(context),
                'sources_count': len(sources),
                'images_count': len(images_data),
                'usage': usage,
                'stop_reason': stop_reason,
                'cache_metrics': cache_metrics
            }
            
            self._dump_llm_request_and_response(
                query=query,
                raw_answer=raw_answer,
                full_request_body=request_body,
                request_info=request_info,
                context_data=context_data
            )

            # Prepare metadata for structured parsing
            parsing_metadata = {
                'model': self.model_id,
                'usage': usage,
                'processing_time': 0.0,  # Will be calculated by caller
                'search_strategy': 'hybrid_search',
                'documents_retrieved': len(sources),
                'cache_metrics': cache_metrics,
                'sources': sources,
                'stop_reason': stop_reason,
                'images_processed': len(images_data)
            }

            # Parse response using structured parser
            parsed_data, is_structured = self.response_parser.parse_response(raw_answer, parsing_metadata)
            
            # Format for display
            display_data = self.response_parser.format_for_display(parsed_data)

            # Prepare result with both structured and legacy format for compatibility
            result = {
                # Legacy format fields for backward compatibility
                'answer': display_data['answer'],
                'confidence_score': display_data['confidence_score'],
                'confidence_rationale': display_data['confidence_rationale'],
                'model': self.model_id,
                'usage': usage,
                'stop_reason': stop_reason,
                'sources': sources,
                'num_sources': len(sources),
                'images_processed': len(images_data),
                'cache_metrics': cache_metrics,
                
                # New structured format fields
                'structured_response': parsed_data,
                'display_data': display_data,
                'is_structured': is_structured,
                'key_points': display_data['key_points'],
                'follow_up_questions': display_data['follow_up_questions'],
                'related_topics': display_data['related_topics'],
                'warnings': display_data['warnings']
            }

            confidence_pct = display_data['confidence_score'] * 100
            logger.info(f"Answer generated successfully. Confidence: {confidence_pct:.0f}% ({display_data['confidence_level']}), "
                       f"Structured: {is_structured}, Images processed: {len(images_data)}")
            if cache_metrics and self.track_metrics:
                self._log_cache_metrics(cache_metrics)

            return result

        except ClientError as e:
            logger.error(f"Bedrock API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise


    def get_stats(self) -> Dict[str, Any]:
        """Get LLM client statistics."""
        return {
            'model': self.model_id,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p
        }

    def generate(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """Generate answer using Claude via AWS Bedrock."""
        prompt = self._build_rag_prompt(query, context, system_prompt)
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature

        try:
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}]
            }

            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType='application/json',
                accept='application/json'
            )

            response_body = json.loads(response['body'].read())
            answer = response_body['content'][0]['text']
            confidence_score = self._extract_confidence_score(answer)

            return {
                'answer': answer,
                'model': self.model_id,
                'usage': response_body.get('usage', {}),
                'confidence_score': confidence_score
            }

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise
