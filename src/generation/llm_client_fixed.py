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
    - RAG-optimized prompting with GADEA specialization
    - Streaming support with confidence
    - FIXED: Multimodal support (images) in both streaming and non-streaming methods
    - Full compatibility with EC2 infrastructure
    """

    def __init__(self, config_path: str = "config/aws_config_production.yaml"):
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

    def _dump_llm_response(self, query: str, raw_answer: str, request_info: Dict[str, Any] = None) -> None:
        """
        Dump LLM response to file for analysis.
        
        Args:
            query: User query
            raw_answer: Raw LLM response
            request_info: Additional request information
        """
        try:
            # Create logs directory if it doesn't exist
            logs_dir = "/home/ec2-user/rag-system-multi/logs"
            os.makedirs(logs_dir, exist_ok=True)
            
            # Generate timestamp and filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            filename = f"llm_response_{timestamp}.txt"
            filepath = os.path.join(logs_dir, filename)
            
            # Prepare dump content
            dump_content = f"""
=== LLM RESPONSE DUMP ===
Timestamp: {datetime.now().isoformat()}
Model: {self.model_id}
Query: {query}

=== REQUEST INFO ===
{json.dumps(request_info or {}, indent=2)}

=== RAW LLM RESPONSE ===
{raw_answer}

=== RESPONSE ANALYSIS ===
Length: {len(raw_answer)} characters
Lines: {len(raw_answer.splitlines())} lines
Starts with JSON: {raw_answer.strip().startswith('{')}
Ends with JSON: {raw_answer.strip().endswith('}')}
Contains JSON: {'{' in raw_answer and '}' in raw_answer}

=== END DUMP ===
"""
            
            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(dump_content)
            
            logger.info(f"LLM response dumped to: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to dump LLM response: {e}")

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
        # Debug: Log the end of the answer to see what we're working with
        logger.info(f"Extracting confidence from answer ending: ...{answer[-200:]}")
        
        # Look for pattern: CONFIDENCE: XX% followed by rationale
        # More flexible pattern that captures everything after CONFIDENCE: XX%
        pattern = r'CONFIDENCE:\s*(\d+)%\s*(.*?)(?=\n\n|\Z)'
        match = re.search(pattern, answer, re.IGNORECASE | re.DOTALL)

        if match:
            confidence_percent = int(match.group(1))
            rationale = match.group(2).strip() if match.group(2) else ""
            
            # Debug: Log what we extracted
            logger.info(f"Regex matched - Confidence: {confidence_percent}%, Rationale length: {len(rationale)}")
            if rationale:
                logger.info(f"Rationale preview: {rationale[:100]}...")
            else:
                logger.warning("No rationale text found after CONFIDENCE: XX%")
            
            # Validate range
            if 0 <= confidence_percent <= 100:
                confidence_decimal = confidence_percent / 100.0
                logger.info(f"Extracted confidence score: {confidence_percent}% with rationale")
                return {
                    'confidence_score': confidence_decimal,
                    'confidence_rationale': rationale
                }
            else:
                logger.warning(f"Confidence score out of range: {confidence_percent}%")
                return {'confidence_score': None, 'confidence_rationale': ''}
        else:
            logger.warning("No confidence score found in answer")
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
                    logger.info(f"Found confidence with alternative pattern: {confidence_percent}%")
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
            system_prompt = """You are a helpful AI assistant that answers questions based on the provided context.
Your task is to provide accurate, detailed answers using ONLY the information from the context.
If the context doesn't contain enough information to answer the question, say so clearly.
Always cite which parts of the context you used to formulate your answer."""

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
        Build a completely JSON-structured prompt to encourage consistent JSON responses.
        
        Args:
            query: User query
            context: Retrieved context
            system_prompt: System prompt instructions
            memory_context: Conversation memory
            sources: Source documents metadata
            
        Returns:
            JSON-structured prompt
        """
        # Structure everything as JSON to create a pattern for the LLM
        structured_input = {
            "system_instructions": {
                "role": "document_assistant",
                "task": "analyze_documents_and_respond_in_json",
                "instructions": system_prompt or "Analyze the provided documents and respond in JSON format",
                "output_format": "json_only",
                "language": "spanish"
            },
            "conversation_context": {
                "previous_interactions": memory_context or "No previous context",
                "current_session": "active"
            },
            "document_sources": [
                {
                    "source_id": source.get('id', f'[{i+1}]'),
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
        
        # Create the JSON-structured prompt
        json_prompt = f"""INPUT_DATA:
{json.dumps(structured_input, indent=2, ensure_ascii=False)}

TASK: Analyze the INPUT_DATA and provide a comprehensive response in JSON format following this exact structure:

{{
  "response_type": "document_based",
  "answer": "Your detailed answer with citations [N]",
  "confidence": {{
    "score": 0.85,
    "level": "high",
    "rationale": "Explanation of confidence level"
  }},
  "sources": [
    {{
      "id": "[1]",
      "title": "Document title",
      "relevance_score": 0.90,
      "excerpt": "Relevant excerpt from document"
    }}
  ],
  "key_points": ["Key point 1", "Key point 2"],
  "related_topics": ["Topic 1", "Topic 2"]
}}

RESPONSE:"""

        return json_prompt

    def _get_system_prompt(self) -> str:
        """
        Get system prompt. This method can be overridden by multi-app system.
        Default fallback prompt for standalone usage.
        """
        return """You are a helpful AI assistant that answers questions based on the provided context.
Your task is to provide accurate, detailed answers using ONLY the information from the context.
If the context doesn't contain enough information to answer the question, say so clearly.
Always cite which parts of the context you used to formulate your answer.

At the end of your response, include a confidence assessment in the format:
CONFIDENCE: XX%

Where XX is a number between 0 and 100 representing your confidence in the answer."""

    def _extract_images_from_results(self, expanded_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract and validate image data from search results.
        ENHANCED: With increased input token limit (180K), we can process more images.
        
        Args:
            expanded_results: Search results with metadata
            
        Returns:
            List of validated image data dictionaries
        """
        images_data = []
        max_images = 5  # Increased limit with higher token capacity
        total_size_limit = 8 * 1024 * 1024  # 8MB total limit (increased)
        current_total_size = 0
        
        for i, result in enumerate(expanded_results, 1):
            # Stop if we've reached the image limit
            if len(images_data) >= max_images:
                logger.info(f"Reached maximum image limit ({max_images}), skipping remaining images")
                break
                
            metadata = result.get('metadata', {})
            if metadata.get('has_image') and metadata.get('image_base64'):
                source_id = f"[{i}]"
                raw_image_data = {
                    'source_id': source_id,
                    'image_base64': metadata['image_base64'],
                    'image_id': metadata.get('image_id', f'img_{i}'),
                    'image_context': metadata.get('image_context', ''),
                    'image_format': metadata.get('image_format', 'PNG')
                }
                
                # Validar y corregir imagen
                is_valid, validated_data, error_msg = self.image_validator.validate_and_fix_image(raw_image_data)
                
                if is_valid:
                    image_size = validated_data.get('size_bytes', 0)
                    
                    # Check if adding this image would exceed total size limit
                    if current_total_size + image_size > total_size_limit:
                        logger.warning(f"Adding image {source_id} would exceed total size limit ({total_size_limit} bytes), skipping")
                        continue
                    
                    images_data.append(validated_data)
                    current_total_size += image_size
                    logger.info(f"Found and validated image in source {source_id}: {validated_data.get('image_id')} ({image_size} bytes)")
                else:
                    logger.warning(f"Image validation failed for source {source_id}: {error_msg}")
        
        if images_data:
            logger.info(f"Selected {len(images_data)} images with total size: {current_total_size} bytes (max input tokens: {self.max_input_tokens})")
        
        return images_data

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
                'metadata': result.get('metadata', {})
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
                sources_for_json.append({
                    'id': f'[{i}]',
                    'text': result['text'],
                    'metadata': result.get('metadata', {}),
                    'score': result.get('rrf_score', result.get('score', 0.0))
                })
            
            # Build JSON-structured prompt
            json_structured_prompt = self._build_json_structured_prompt(
                query=query,
                context=context,
                system_prompt=system_prompt,
                memory_context=None,  # TODO: Add memory context if available
                sources=sources_for_json
            )
            
            # Use minimal system prompt for JSON approach
            json_system_prompt = "You are a document analysis assistant. Analyze the provided JSON input and respond with a valid JSON object following the specified structure."
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

            # Dump LLM response for analysis
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
            self._dump_llm_response(query, raw_answer, request_info)

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

    def generate_with_citations_streaming(
        self,
        query: str,
        expanded_results: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ):
        """
        Generate answer with source citations using streaming.
        Already has proper image support.

        Args:
            query: User query
            expanded_results: Results with expanded context
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Yields:
            Streaming response chunks
        """
        try:
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
                    'metadata': result.get('metadata', {})
                })

            context = "\n\n".join(context_parts)
            logger.info(f"Formatted context with {len(sources)} sources, {len(images_data)} images")

            # If images are present, add note to context
            if images_data:
                context += f"\n\n[NOTA: Se han incluido {len(images_data)} imagen(es) visual(es) para análisis]"

            # Get system prompt
            system_prompt = self._get_system_prompt()

            # Build prompt
            prompt = self._build_rag_prompt(query, context, system_prompt)

            # Use provided parameters or defaults
            max_tokens = max_tokens or self.max_tokens
            temperature = temperature or self.temperature

            # Build message content (text + optional images)
            if images_data:
                message_content = self._build_multimodal_content(prompt, images_data)
            else:
                message_content = prompt

            # Prepare Claude streaming request
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": message_content
                    }
                ]
            }

            logger.info(f"Calling Bedrock streaming API with model: {self.model_id} (Images: {len(images_data)})")

            # Call Bedrock with streaming
            response = self.bedrock_client.invoke_model_with_response_stream(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType='application/json',
                accept='application/json'
            )

            # Track full response for confidence extraction
            full_text = []
            usage_data = {}
            stop_reason = None
            chunks_received = 0

            # Stream response
            stream = response.get('body')
            if not stream:
                logger.error("No stream body in Bedrock response")
                raise RuntimeError("No stream body received from Bedrock")

            logger.info("Processing streaming events...")

            for event in stream:
                chunk = event.get('chunk')
                if chunk:
                    try:
                        chunk_data = json.loads(chunk.get('bytes').decode())

                        # Handle Claude streaming format
                        if chunk_data['type'] == 'content_block_delta':
                            delta = chunk_data.get('delta', {})
                            if delta.get('type') == 'text_delta':
                                text = delta.get('text', '')
                                full_text.append(text)
                                chunks_received += 1

                                # Yield text chunk
                                yield {
                                    'type': 'chunk',
                                    'text': text
                                }

                        elif chunk_data['type'] == 'message_delta':
                            # Capture usage and stop reason
                            usage_data = chunk_data.get('usage', {})
                            stop_reason = chunk_data.get('delta', {}).get('stop_reason')

                        elif chunk_data['type'] == 'message_stop':
                            logger.info(f"Streaming completed successfully. Chunks received: {chunks_received}")

                            # Extract confidence score and rationale from full text
                            full_response = ''.join(full_text)
                            confidence_data = self._extract_confidence_score_and_rationale(full_response)
                            confidence_score = confidence_data['confidence_score']
                            confidence_rationale = confidence_data['confidence_rationale']

                            # Yield completion metadata with confidence score and rationale
                            yield {
                                'type': 'complete',
                                'metadata': {
                                    'model': self.model_id,
                                    'usage': {
                                        'input_tokens': usage_data.get('input_tokens', 0),
                                        'output_tokens': usage_data.get('output_tokens', 0)
                                    },
                                    'stop_reason': stop_reason or 'end_turn',
                                    'full_text': full_response,
                                    'chunks_received': chunks_received,
                                    'confidence_score': confidence_score,
                                    'confidence_rationale': confidence_rationale,
                                    'sources': sources,
                                    'num_sources': len(sources),
                                    'images_processed': len(images_data)
                                }
                            }
                            break

                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse chunk JSON: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing chunk: {e}")
                        continue

                # Handle error events
                elif event.get('error'):
                    error_data = event.get('error')
                    logger.error(f"Stream error event: {error_data}")
                    raise RuntimeError(f"Stream error: {error_data}")

            logger.info(f"Stream processing completed. Total chunks: {chunks_received}, Images processed: {len(images_data)}")

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            logger.error(f"Bedrock API error [{error_code}]: {error_message}")
            raise RuntimeError(f"Bedrock API error: {error_message}")
        except Exception as e:
            logger.error(f"Error in streaming generation: {e}", exc_info=True)
            raise RuntimeError(f"Streaming generation failed: {str(e)}")

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
