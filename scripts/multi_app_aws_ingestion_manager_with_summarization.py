#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMPORTANT: This script requires Python 3.6+ due to type annotations and f-strings.
If 'python' points to Python 2.x, use 'python3' instead.
"""
"""
Multi-Application AWS Ingestion Manager with Document Summarization
Enhanced version that includes AI-powered document summarization during ingestion

FUNCIONALIDADES:
- scan: Indexa archivos con generación automática de resúmenes
- clean: Limpia chunks huérfanos por aplicación
- remove: Elimina archivos por ruta de OpenSearch y tracking por aplicación
- remove-by-id: Elimina archivo por ID de OpenSearch y tracking por aplicación
- remove-pattern: Elimina archivos por patrón por aplicación
- list: Lista estado completo por aplicación (OpenSearch + Tracking + S3)
- list-apps: Lista aplicaciones disponibles
- update-inventory: Actualiza el inventario de documentos en S3
"""

import sys
import json
import hashlib
import boto3
import yaml
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Set, Tuple
from tabulate import tabulate
import tempfile
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.multi_app_config_manager import MultiAppConfigManager
from src.utils.connection_manager import ConnectionManager
from src.utils.stop_words_manager import get_stop_words_manager
from src.ingestion.document_loader import DocumentLoader
from src.indexing.multi_app_opensearch_indexer import MultiAppOpenSearchIndexer
from loguru import logger

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class DocumentSummarizer:
    """
    Document summarization component integrated with the ingestion pipeline.
    """
    
    def __init__(self, app_name: str, config_manager: MultiAppConfigManager):
        """
        Initialize document summarizer.
        
        Args:
            app_name: Application name
            config_manager: Multi-app configuration manager
        """
        self.app_name = app_name
        self.config_manager = config_manager
        self.app_config = config_manager.get_application_config(app_name)
        
        # Initialize Bedrock client for summarization
        legacy_config = config_manager.create_legacy_config(app_name)
        temp_config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(legacy_config, temp_config_file, default_flow_style=False)
        temp_config_file.close()
        
        self.conn_manager = ConnectionManager(config_path=temp_config_file.name)
        self.bedrock_client = self.conn_manager.get_bedrock_client()
        self.s3_client = self.conn_manager.get_s3_client()
        
        # Clean up temp file
        os.unlink(temp_config_file.name)
        
        # Get application-specific settings
        self.s3_bucket = self.app_config['services']['s3']['bucket']
        self.bedrock_config = self.config_manager.get_bedrock_config()
        self.system_prompt = self.config_manager.get_system_prompt(app_name)
        
        logger.info(f"[{app_name}] DocumentSummarizer initialized")
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        return int(len(text.split()) * 1.3)  # Rough approximation
    
    def extract_key_terms(self, text: str, max_terms: int = 10) -> List[str]:
        """Extract key terms from text using enhanced stop words filtering."""
        logger.info(f"[{self.app_name}] 🔍 TRACE: Starting key terms extraction for text ({len(text)} chars)")
        
        try:
            logger.info(f"[{self.app_name}] 🔍 TRACE: Attempting to use StopWordsManager...")
            
            # Use the new stop words manager for better filtering
            stop_words_manager = get_stop_words_manager()
            logger.info(f"[{self.app_name}] 🔍 TRACE: StopWordsManager initialized successfully")
            
            key_terms = stop_words_manager.extract_key_terms(
                text=text,
                languages=['english', 'spanish'],
                application=self.app_name,
                max_terms=max_terms
            )
            
            logger.info(f"[{self.app_name}] ✅ TRACE: Successfully extracted {len(key_terms)} key terms using StopWordsManager")
            logger.info(f"[{self.app_name}] 🔍 TRACE: Key terms from StopWordsManager: {key_terms}")
            return key_terms
            
        except Exception as e:
            logger.warning(f"[{self.app_name}] ❌ TRACE: Error using StopWordsManager, falling back to basic method: {e}")
            logger.warning(f"[{self.app_name}] 🔍 TRACE: Exception details: {type(e).__name__}: {str(e)}")
            
            # Fallback to basic method if stop words manager fails
            import re
            from collections import Counter
            
            logger.info(f"[{self.app_name}] 🔍 TRACE: Using FALLBACK method for key terms extraction")
            
            # Simple tokenization and filtering
            words = re.findall(r'\b[a-zA-ZáéíóúÁÉÍÓÚñÑ]{3,}\b', text.lower())
            logger.info(f"[{self.app_name}] 🔍 TRACE: Fallback tokenized {len(words)} words")
            
            # Basic stop words (fallback)
            basic_stop_words = {
                'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                'que', 'de', 'la', 'el', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da',
                'su', 'por', 'son', 'con', 'una', 'las', 'los', 'del', 'al', 'como', 'pero',
                'sus', 'fue', 'ser', 'han', 'más', 'para', 'está', 'hasta', 'todo', 'este',
                'esta', 'estos', 'estas', 'ese', 'esa', 'esos', 'esas', 'aquel', 'aquella'
            }
            
            filtered_words = [word for word in words if word not in basic_stop_words and len(word) > 3]
            logger.info(f"[{self.app_name}] 🔍 TRACE: Fallback filtered to {len(filtered_words)} words after removing {len(basic_stop_words)} stop words")
            
            # Get most common terms
            counter = Counter(filtered_words)
            fallback_terms = [term for term, count in counter.most_common(max_terms)]
            logger.info(f"[{self.app_name}] ⚠️ TRACE: Fallback method extracted {len(fallback_terms)} key terms: {fallback_terms}")
            
            return fallback_terms
    
    def generate_summary(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate AI summary for a document.
        
        Args:
            document: Document dictionary with content and metadata
            
        Returns:
            Dictionary with summary and metadata
        """
        try:
            content = document.get('content', '')
            file_name = document.get('file_name', 'unknown')
            file_extension = document.get('file_extension', '')
            
            if not content:
                logger.warning(f"[{self.app_name}] No content to summarize for {file_name}")
                return self._create_empty_summary(file_name, "No content available")
            
            # Estimate tokens
            estimated_tokens = self.estimate_tokens(content)
            max_context_tokens = 8000  # Conservative limit for Claude
            
            # Handle large documents with hierarchical summarization
            if estimated_tokens > max_context_tokens:
                logger.info(f"[{self.app_name}] Large document detected ({estimated_tokens} tokens), using hierarchical summarization")
                return self._generate_hierarchical_summary(document)
            
            # Generate summary for regular-sized documents
            return self._generate_single_summary(document)
            
        except Exception as e:
            logger.error(f"[{self.app_name}] Error generating summary for {file_name}: {e}")
            return self._create_empty_summary(file_name, f"Error: {str(e)}")
    
    def _generate_single_summary(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary for a single document."""
        content = document.get('content', '')
        file_name = document.get('file_name', 'unknown')
        file_extension = document.get('file_extension', '')
        
        # Create application-specific prompt
        prompt = f"""
{self.system_prompt}

TAREA ESPECÍFICA: GENERACIÓN DE RESUMEN DOCUMENTAL

Analiza el siguiente documento del sistema {self.app_name.upper()} y genera un resumen estructurado:

DOCUMENTO: {file_name}
TIPO: {file_extension}

CONTENIDO:
{content}

INSTRUCCIONES:
1. Genera un resumen conciso pero completo (150-300 palabras)
2. Identifica los temas principales y conceptos clave
3. Extrae términos técnicos relevantes para {self.app_name.upper()}
4. Mantén el contexto específico de la aplicación
5. Usa un lenguaje claro y profesional

FORMATO DE RESPUESTA:
RESUMEN: [Tu resumen aquí]

TEMAS PRINCIPALES: [Lista de 3-5 temas principales]

TÉRMINOS CLAVE: [Lista de 5-10 términos técnicos relevantes]

RELEVANCIA: [Puntuación de relevancia del 1-10 para el sistema {self.app_name.upper()}]
"""
        
        try:
            # Call Bedrock Claude
            response = self.bedrock_client.invoke_model(
                modelId=self.bedrock_config['llm_model'],
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": self.bedrock_config.get('max_tokens', 1000),
                    "temperature": self.bedrock_config.get('temperature', 0.3),
                    "top_p": self.bedrock_config.get('top_p', 0.9),
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }),
                contentType='application/json',
                accept='application/json'
            )
            
            response_body = json.loads(response['body'].read())
            ai_response = response_body['content'][0]['text']
            
            # Parse the structured response
            summary_data = self._parse_ai_response(ai_response, file_name)
            
            # Add metadata
            summary_data.update({
                'generated_at': datetime.now().isoformat(),
                'model_used': self.bedrock_config['llm_model'],
                'application_context': self.app_name,
                'token_count': self.estimate_tokens(summary_data['summary']),
                'source_token_count': self.estimate_tokens(content),
                'summarization_method': 'single_pass'
            })
            
            logger.info(f"[{self.app_name}] Generated summary for {file_name} ({len(summary_data['summary'])} chars)")
            return summary_data
            
        except Exception as e:
            logger.error(f"[{self.app_name}] Error calling Bedrock for {file_name}: {e}")
            return self._create_empty_summary(file_name, f"Bedrock error: {str(e)}")
    
    def _generate_hierarchical_summary(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Generate hierarchical summary for large documents."""
        content = document.get('content', '')
        file_name = document.get('file_name', 'unknown')
        
        # Split content into chunks
        chunk_size = 6000  # Conservative chunk size
        chunks = []
        
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            chunks.append(chunk)
        
        logger.info(f"[{self.app_name}] Processing {len(chunks)} chunks for hierarchical summary")
        
        # Generate summaries for each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            try:
                chunk_doc = {
                    **document,
                    'content': chunk,
                    'file_name': f"{file_name}_chunk_{i+1}"
                }
                chunk_summary = self._generate_single_summary(chunk_doc)
                chunk_summaries.append(chunk_summary['summary'])
            except Exception as e:
                logger.warning(f"[{self.app_name}] Error summarizing chunk {i+1}: {e}")
                continue
        
        if not chunk_summaries:
            return self._create_empty_summary(file_name, "Failed to generate chunk summaries")
        
        # Combine chunk summaries into final summary
        combined_content = "\n\n".join(chunk_summaries)
        
        final_doc = {
            **document,
            'content': combined_content,
            'file_name': f"{file_name}_combined"
        }
        
        final_summary = self._generate_single_summary(final_doc)
        final_summary['summarization_method'] = 'hierarchical'
        final_summary['chunk_count'] = len(chunks)
        
        logger.info(f"[{self.app_name}] Generated hierarchical summary for {file_name}")
        return final_summary
    
    def _parse_ai_response(self, ai_response: str, file_name: str) -> Dict[str, Any]:
        """Parse structured AI response into summary data."""
        logger.info(f"[{self.app_name}] 🔍 TRACE: Starting _parse_ai_response for {file_name}")
        logger.info(f"[{self.app_name}] 🔍 TRACE: AI response length: {len(ai_response)} chars")
        
        try:
            # Extract sections using regex
            logger.info(f"[{self.app_name}] 🔍 TRACE: Extracting sections with regex...")
            summary_match = re.search(r'RESUMEN:\s*(.*?)(?=\n\n|\nTEMAS|$)', ai_response, re.DOTALL)
            topics_match = re.search(r'TEMAS PRINCIPALES:\s*(.*?)(?=\n\n|\nTÉRMINOS|$)', ai_response, re.DOTALL)
            terms_match = re.search(r'TÉRMINOS CLAVE:\s*(.*?)(?=\n\n|\nRELEVANCIA|$)', ai_response, re.DOTALL)
            relevance_match = re.search(r'RELEVANCIA:\s*(\d+)', ai_response)
            
            logger.info(f"[{self.app_name}] 🔍 TRACE: Regex matches - summary: {bool(summary_match)}, topics: {bool(topics_match)}, terms: {bool(terms_match)}, relevance: {bool(relevance_match)}")
            
            summary = summary_match.group(1).strip() if summary_match else ai_response[:500]
            logger.info(f"[{self.app_name}] 🔍 TRACE: Extracted summary length: {len(summary)} chars")
            
            # Parse topics and clean them with stop words filtering
            topics = []
            logger.info(f"[{self.app_name}] 🔍 TRACE: Processing topics...")
            if topics_match:
                topics_text = topics_match.group(1).strip()
                logger.info(f"[{self.app_name}] 🔍 TRACE: Raw topics text: {topics_text[:200]}...")
                raw_topics = [topic.strip('- ').strip() for topic in topics_text.split('\n') if topic.strip()]
                
                # Clean topics by removing numbers and applying stop words filtering
                logger.info(f"[{self.app_name}] 🔍 TRACE: Processing {len(raw_topics)} raw topics, removing numbers and applying stop words...")
                for topic in raw_topics:
                    clean_topic = re.sub(r'^\d+\.\s*', '', topic.strip())
                    if clean_topic:
                        # Apply stop words filtering to each topic
                        try:
                            logger.info(f"[{self.app_name}] 🔍 TRACE: Filtering topic: '{clean_topic}'")
                            filtered_topic_terms = self.extract_key_terms(clean_topic, max_terms=5)
                            if filtered_topic_terms:
                                # Reconstruct topic from filtered terms
                                filtered_topic = ' '.join(filtered_topic_terms)
                                topics.append(filtered_topic)
                                logger.info(f"[{self.app_name}] ✅ TRACE: Topic filtered: '{clean_topic}' -> '{filtered_topic}'")
                            else:
                                # Keep original if filtering returns empty
                                topics.append(clean_topic)
                                logger.info(f"[{self.app_name}] ⚠️ TRACE: Topic kept original (empty after filtering): '{clean_topic}'")
                        except Exception as e:
                            logger.warning(f"[{self.app_name}] ❌ TRACE: Error filtering topic '{clean_topic}': {e}")
                            topics.append(clean_topic)  # Fallback to original
                            
                logger.info(f"[{self.app_name}] ✅ TRACE: Final filtered topics: {topics}")
            else:
                logger.info(f"[{self.app_name}] ❌ TRACE: No topics found in AI response")
            
            # Parse terms
            terms = []
            logger.info(f"[{self.app_name}] 🔍 TRACE: Processing terms...")
            if terms_match:
                terms_text = terms_match.group(1).strip()
                logger.info(f"[{self.app_name}] 🔍 TRACE: Raw terms text: {terms_text[:200]}...")
                terms = [term.strip('- ').strip() for term in terms_text.split('\n') if term.strip()]
                logger.info(f"[{self.app_name}] 🔍 TRACE: Extracted {len(terms)} raw terms: {terms}")
            else:
                logger.info(f"[{self.app_name}] ❌ TRACE: No terms found in AI response")
            
            # Parse relevance
            relevance = float(relevance_match.group(1)) if relevance_match else 5.0
            
            # ALWAYS process key terms through StopWordsManager for proper filtering
            logger.info(f"[{self.app_name}] 🔍 TRACE: CRITICAL POINT - Processing {len(terms)} raw terms from AI through StopWordsManager...")
            
            if terms:
                # Clean the AI-provided terms by removing numbers and applying stop words filtering
                cleaned_terms = []
                logger.info(f"[{self.app_name}] 🔍 TRACE: Cleaning terms by removing numbers...")
                for term in terms:
                    # Remove leading numbers like "1. ", "2. ", etc.
                    clean_term = re.sub(r'^\d+\.\s*', '', term.strip())
                    if clean_term:
                        cleaned_terms.append(clean_term)
                
                logger.info(f"[{self.app_name}] 🔍 TRACE: Cleaned {len(cleaned_terms)} terms: {cleaned_terms}")
                logger.info(f"[{self.app_name}] 🔍 TRACE: NOW CALLING extract_key_terms() - THIS SHOULD SHOW TRACES...")
                
                # Join terms and process through StopWordsManager
                terms_text = ' '.join(cleaned_terms)
                logger.info(f"[{self.app_name}] 🔍 TRACE: Terms text to process: '{terms_text}'")
                
                try:
                    processed_terms = self.extract_key_terms(terms_text, max_terms=10)
                    logger.info(f"[{self.app_name}] ✅ TRACE: StopWordsManager processed terms: {processed_terms}")
                    terms = processed_terms
                except Exception as e:
                    logger.error(f"[{self.app_name}] ❌ TRACE: EXCEPTION in extract_key_terms: {type(e).__name__}: {str(e)}")
                    logger.error(f"[{self.app_name}] ❌ TRACE: Keeping original cleaned terms: {cleaned_terms}")
                    terms = cleaned_terms[:10]  # Fallback to cleaned terms
                    
            else:
                # Fallback: extract from summary if no terms provided
                logger.info(f"[{self.app_name}] 🔍 TRACE: No terms from AI, extracting from summary...")
                logger.info(f"[{self.app_name}] 🔍 TRACE: CALLING extract_key_terms() with summary - THIS SHOULD SHOW TRACES...")
                try:
                    terms = self.extract_key_terms(summary)
                    logger.info(f"[{self.app_name}] ✅ TRACE: Extracted terms from summary: {terms}")
                except Exception as e:
                    logger.error(f"[{self.app_name}] ❌ TRACE: EXCEPTION extracting from summary: {type(e).__name__}: {str(e)}")
                    terms = []  # Empty fallback
            
            # ALSO process summary through StopWordsManager for better indexing AND filtering
            logger.info(f"[{self.app_name}] 🔍 TRACE: Processing summary through StopWordsManager for enhanced indexing AND filtering...")
            try:
                # Extract key terms from summary for metadata
                summary_key_terms = self.extract_key_terms(summary, max_terms=20)
                logger.info(f"[{self.app_name}] ✅ TRACE: Summary key terms extracted: {len(summary_key_terms)} terms")
                
                # FILTER the summary text itself by removing stop words and reconstructing
                logger.info(f"[{self.app_name}] 🔍 TRACE: FILTERING summary text by removing stop words...")
                try:
                    stop_words_manager = get_stop_words_manager()
                    
                    # Get stop words for filtering
                    stop_words_set = set()
                    stop_words_set.update(stop_words_manager.get_stop_words('english'))
                    stop_words_set.update(stop_words_manager.get_stop_words('spanish'))
                    stop_words_set.update(stop_words_manager.get_stop_words(self.app_name))
                    
                    logger.info(f"[{self.app_name}] 🔍 TRACE: Using {len(stop_words_set)} stop words for summary filtering")
                    
                    # Split summary into sentences to preserve structure
                    import re
                    sentences = re.split(r'[.!?]+', summary)
                    filtered_sentences = []
                    
                    for sentence in sentences:
                        if sentence.strip():
                            # Split sentence into words and filter
                            words = re.findall(r'\b\w+\b', sentence.lower())
                            filtered_words = [word for word in words if word not in stop_words_set and len(word) > 2]
                            
                            if filtered_words:  # Only keep sentences with meaningful words
                                # Reconstruct sentence maintaining original case for important words
                                original_words = re.findall(r'\b\w+\b', sentence)
                                filtered_sentence_words = []
                                
                                for orig_word in original_words:
                                    if orig_word.lower() in [fw.lower() for fw in filtered_words]:
                                        filtered_sentence_words.append(orig_word)
                                
                                if filtered_sentence_words:
                                    filtered_sentences.append(' '.join(filtered_sentence_words))
                    
                    # Reconstruct summary from filtered sentences
                    if filtered_sentences:
                        filtered_summary = '. '.join(filtered_sentences) + '.'
                        logger.info(f"[{self.app_name}] ✅ TRACE: Summary FILTERED - Original: {len(summary)} chars, Filtered: {len(filtered_summary)} chars")
                        logger.info(f"[{self.app_name}] 🔍 TRACE: Original summary: {summary[:100]}...")
                        logger.info(f"[{self.app_name}] 🔍 TRACE: Filtered summary: {filtered_summary[:100]}...")
                        
                        # Use filtered summary
                        summary = filtered_summary
                    else:
                        logger.warning(f"[{self.app_name}] ⚠️ TRACE: Summary filtering resulted in empty text, keeping original")
                    
                    # Count words for metadata
                    original_words = re.findall(r'\b\w+\b', summary.lower())
                    filtered_word_count = len([word for word in original_words if word not in stop_words_set and len(word) > 2])
                    
                    logger.info(f"[{self.app_name}] ✅ TRACE: Summary processed - {len(original_words)} total words, {filtered_word_count} meaningful words")
                    
                except Exception as e:
                    logger.warning(f"[{self.app_name}] ❌ TRACE: Error filtering summary text: {e}")
                    filtered_word_count = 0
                
                # Add summary key terms to document metadata for enhanced search
                summary_enhanced_metadata = {
                    'summary_key_terms': summary_key_terms,
                    'summary_processed': True,
                    'summary_filtered_words': filtered_word_count,
                    'summary_total_words': len(re.findall(r'\b\w+\b', summary.lower())) if summary else 0
                }
                logger.info(f"[{self.app_name}] 🔍 TRACE: Enhanced summary metadata created")
                
            except Exception as e:
                logger.warning(f"[{self.app_name}] ❌ TRACE: Error processing summary through StopWordsManager: {e}")
                summary_enhanced_metadata = {
                    'summary_key_terms': [],
                    'summary_processed': False
                }
            
            result = {
                'summary': summary,
                'topics': topics[:5],  # Limit to 5 topics
                'key_terms': terms[:10],  # Limit to 10 terms
                'relevance_score': relevance,
                'confidence_score': min(0.9, relevance / 10.0)  # Convert to 0-1 scale
            }
            
            # Add enhanced summary metadata if available
            if 'summary_enhanced_metadata' in locals():
                result.update(summary_enhanced_metadata)
            
            return result
            
        except Exception as e:
            logger.warning(f"[{self.app_name}] Error parsing AI response for {file_name}: {e}")
            # Fallback to simple summary
            return {
                'summary': ai_response[:500] if ai_response else "Summary generation failed",
                'topics': [],
                'key_terms': self.extract_key_terms(ai_response) if ai_response else [],
                'relevance_score': 5.0,
                'confidence_score': 0.5
            }
    
    def _create_empty_summary(self, file_name: str, reason: str) -> Dict[str, Any]:
        """Create empty summary structure for failed cases."""
        return {
            'summary': f"Summary not available for {file_name}. Reason: {reason}",
            'topics': [],
            'key_terms': [],
            'relevance_score': 0.0,
            'confidence_score': 0.0,
            'generated_at': datetime.now().isoformat(),
            'model_used': 'none',
            'application_context': self.app_name,
            'token_count': 0,
            'source_token_count': 0,
            'summarization_method': 'failed',
            'error': reason
        }
    
    def store_summary_to_s3(self, summary_data: Dict[str, Any], document: Dict[str, Any]) -> str:
        """
        Store document summary to S3.
        
        Args:
            summary_data: Summary data dictionary
            document: Original document data
            
        Returns:
            S3 key where summary was stored
        """
        try:
            file_hash = document.get('file_hash', 'unknown')
            file_name = document.get('file_name', 'unknown')
            
            # Create summary object
            summary_object = {
                'document_info': {
                    'file_name': file_name,
                    'file_hash': file_hash,
                    'file_path': document.get('file_path', ''),
                    'file_size': document.get('file_size', 0),
                    'file_extension': document.get('file_extension', ''),
                    'application_id': self.app_name
                },
                'summary_data': summary_data,
                'created_at': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            # Store in S3
            s3_key = f"applications/{self.app_name}/summaries/{file_hash}.json"
            
            # Sanitize metadata values to ensure ASCII compatibility
            safe_document_name = file_name.encode('ascii', 'ignore').decode('ascii')
            if not safe_document_name:
                safe_document_name = f"document_{file_hash[:8]}"
            
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=json.dumps(summary_object, indent=2, ensure_ascii=False),
                ContentType='application/json',
                Metadata={
                    'application_id': self.app_name,
                    'document_name': safe_document_name,
                    'summary_version': '1.0',
                    'original_filename_hash': file_hash[:16]  # Add hash for identification
                }
            )
            
            logger.info(f"[{self.app_name}] Stored summary to S3: {s3_key}")
            return s3_key
            
        except Exception as e:
            logger.error(f"[{self.app_name}] Error storing summary to S3: {e}")
            return ""


class DocumentInventoryManager:
    """
    Manages document inventory for providing context to LLM queries.
    """
    
    def __init__(self, app_name: str, config_manager: MultiAppConfigManager, s3_client):
        """
        Initialize document inventory manager.
        
        Args:
            app_name: Application name
            config_manager: Multi-app configuration manager
            s3_client: S3 client instance
        """
        self.app_name = app_name
        self.config_manager = config_manager
        self.s3_client = s3_client
        self.app_config = config_manager.get_application_config(app_name)
        self.s3_bucket = self.app_config['services']['s3']['bucket']
        
        logger.info(f"[{app_name}] DocumentInventoryManager initialized")
    
    def update_document_inventory(self) -> str:
        """
        Update the complete document inventory in S3.
        
        Returns:
            S3 key where inventory was stored
        """
        try:
            # Get all summaries for this application
            summaries_prefix = f"applications/{self.app_name}/summaries/"
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=summaries_prefix
            )
            
            document_summaries = []
            total_documents = 0
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    if obj['Key'].endswith('.json'):
                        try:
                            # Get summary object
                            summary_response = self.s3_client.get_object(
                                Bucket=self.s3_bucket,
                                Key=obj['Key']
                            )
                            
                            summary_object = json.loads(summary_response['Body'].read())
                            doc_info = summary_object.get('document_info', {})
                            summary_data = summary_object.get('summary_data', {})
                            
                            # Add to inventory
                            document_summaries.append({
                                'file_name': doc_info.get('file_name', 'unknown'),
                                'file_path': doc_info.get('file_path', ''),
                                'document_type': doc_info.get('file_extension', '').replace('.', ''),
                                'summary': summary_data.get('summary', ''),
                                'key_terms': summary_data.get('key_terms', []),
                                'topics': summary_data.get('topics', []),
                                'relevance_score': summary_data.get('relevance_score', 0.0),
                                'last_updated': summary_object.get('created_at', ''),
                                'file_size': doc_info.get('file_size', 0),
                                's3_summary_key': obj['Key']
                            })
                            
                            total_documents += 1
                            
                        except Exception as e:
                            logger.warning(f"[{self.app_name}] Error processing summary {obj['Key']}: {e}")
                            continue
            
            # Create inventory object
            inventory = {
                'application_id': self.app_name,
                'application_name': self.config_manager.applications[self.app_name].get('name', self.app_name.upper()),
                'generated_at': datetime.now().isoformat(),
                'total_documents': total_documents,
                'document_summary': sorted(document_summaries, key=lambda x: x['relevance_score'], reverse=True),
                'statistics': {
                    'total_documents': total_documents,
                    'avg_relevance_score': sum(doc['relevance_score'] for doc in document_summaries) / max(total_documents, 1),
                    'document_types': list(set(doc['document_type'] for doc in document_summaries)),
                    'total_key_terms': len(set(term for doc in document_summaries for term in doc['key_terms'])),
                    'total_topics': len(set(topic for doc in document_summaries for topic in doc['topics']))
                }
            }
            
            # Store inventory in S3
            inventory_key = f"applications/{self.app_name}/inventory/document_inventory_latest.json"
            
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=inventory_key,
                Body=json.dumps(inventory, indent=2, ensure_ascii=False),
                ContentType='application/json',
                Metadata={
                    'application_id': self.app_name,
                    'inventory_version': '1.0',
                    'total_documents': str(total_documents)
                }
            )
            
            # Also store dated version
            dated_key = f"applications/{self.app_name}/inventory/document_inventory_{datetime.now().strftime('%Y-%m-%d')}.json"
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=dated_key,
                Body=json.dumps(inventory, indent=2, ensure_ascii=False),
                ContentType='application/json'
            )
            
            logger.info(f"[{self.app_name}] Updated document inventory: {total_documents} documents")
            logger.info(f"[{self.app_name}] Inventory stored at: {inventory_key}")
            
            return inventory_key
            
        except Exception as e:
            logger.error(f"[{self.app_name}] Error updating document inventory: {e}")
            return ""


class MultiAppS3DocumentTracker:
    """
    Multi-application document tracker that maintains separate version tracking per application.
    Tracks document versions using S3 metadata and OpenSearch document mapping per application.
    """
    
    def __init__(self, app_name: str, config_manager: MultiAppConfigManager, tracking_base_dir: str = "data"):
        """
        Initialize multi-app S3 document version tracker.
        
        Args:
            app_name: Application name
            config_manager: Multi-app configuration manager
            tracking_base_dir: Base directory for tracking files
        """
        self.app_name = app_name
        self.config_manager = config_manager
        self.app_config = config_manager.get_application_config(app_name)
        
        # Initialize connection manager with legacy config for this app
        legacy_config = config_manager.create_legacy_config(app_name)
        
        # Create temporary config file for ConnectionManager
        temp_config_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(legacy_config, temp_config_file, default_flow_style=False)
        temp_config_file.close()
        
        self.conn_manager = ConnectionManager(config_path=temp_config_file.name)
        
        # Clean up temp file
        os.unlink(temp_config_file.name)
        
        self.s3_client = self.conn_manager.get_s3_client()
        self.opensearch_client = self.conn_manager.get_opensearch_client()
        
        # Application-specific S3 and OpenSearch settings
        self.bucket_name = self.app_config['services']['s3']['bucket']
        self.documents_prefix = self.app_config['services']['s3']['documents_prefix']
        self.index_name = self.app_config['opensearch']['index_name']
        
        # Application-specific tracking file
        tracking_dir = Path(tracking_base_dir)
        tracking_dir.mkdir(parents=True, exist_ok=True)
        self.tracking_file = tracking_dir / f"s3_document_versions_{app_name}.json"
        
        self.versions: Dict[str, Dict[str, Any]] = {}
        self.load()
    
    def _calculate_s3_object_hash(self, s3_key: str) -> str:
        """Calculate hash from S3 object ETag and LastModified."""
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            etag = response['ETag'].strip('"')
            last_modified = response['LastModified'].isoformat()
            
            # Combine ETag and LastModified for a unique hash
            combined = f"{etag}_{last_modified}"
            return hashlib.sha256(combined.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for S3 object {s3_key}: {e}")
            return ""
    
    def load(self) -> None:
        """Load version tracking data from local disk."""
        if self.tracking_file.exists():
            with open(self.tracking_file, 'r') as f:
                self.versions = json.load(f)
            logger.info(f"[{self.app_name}] Loaded version tracking for {len(self.versions)} documents")
        else:
            logger.info(f"[{self.app_name}] No existing version tracking file found")
            self.versions = {}
    
    def save(self) -> None:
        """Save version tracking data to local disk."""
        self.tracking_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.tracking_file, 'w') as f:
            json.dump(self.versions, f, indent=2)
        logger.info(f"[{self.app_name}] Saved version tracking for {len(self.versions)} documents")
    
    def has_changed(self, s3_key: str) -> bool:
        """
        Check if an S3 document has changed since last indexing.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            True if document is new or has changed
        """
        if s3_key not in self.versions:
            return True
        
        current_hash = self._calculate_s3_object_hash(s3_key)
        stored_hash = self.versions[s3_key].get('hash')
        
        return current_hash != stored_hash
    
    def update_version(self, s3_key: str, doc_id: str, chunk_ids: list, summary_s3_key: str = "") -> None:
        """
        Update version information for an S3 document.
        
        Args:
            s3_key: S3 object key
            doc_id: Document ID
            chunk_ids: List of chunk IDs generated from this document
            summary_s3_key: S3 key where summary is stored
        """
        file_hash = self._calculate_s3_object_hash(s3_key)
        
        self.versions[s3_key] = {
            'app_name': self.app_name,  # Add application identifier
            'doc_id': doc_id,
            'hash': file_hash,
            'last_synced': datetime.now().isoformat(),
            'chunk_ids': chunk_ids,
            's3_key': s3_key,
            'num_chunks': len(chunk_ids),
            'summary_s3_key': summary_s3_key,  # Track summary location
            'has_summary': bool(summary_s3_key)
        }
        
        logger.info(f"[{self.app_name}] Updated version info for {s3_key}: {len(chunk_ids)} chunks, summary: {bool(summary_s3_key)}")
    
    def get_chunk_ids(self, s3_key: str) -> list:
        """Get chunk IDs for a document by S3 key."""
        if s3_key in self.versions:
            return self.versions[s3_key].get('chunk_ids', [])
        return []
    
    def get_chunk_ids_by_doc_id(self, doc_id: str) -> list:
        """Get chunk IDs for a document by doc_id."""
        for s3_key, info in self.versions.items():
            if info.get('doc_id') == doc_id:
                return info.get('chunk_ids', [])
        return []
    
    def get_s3_key_by_doc_id(self, doc_id: str) -> Optional[str]:
        """Get S3 key by doc_id."""
        for s3_key, info in self.versions.items():
            if info.get('doc_id') == doc_id:
                return s3_key
        return None
    
    def remove_document(self, s3_key: str) -> None:
        """Remove a document from version tracking."""
        if s3_key in self.versions:
            del self.versions[s3_key]
            logger.info(f"[{self.app_name}] Removed {s3_key} from version tracking")
    
    def list_all_documents(self) -> List[Dict[str, Any]]:
        """
        Get list of all indexed documents with their information.
        
        Returns:
            List of document information dictionaries
        """
        docs = []
        for s3_key, info in self.versions.items():
            filename = s3_key.split('/')[-1]  # Extract filename from S3 key
            docs.append({
                'app_name': self.app_name,
                'filename': filename,
                's3_key': s3_key,
                'doc_id': info.get('doc_id', 'N/A'),
                'num_chunks': info.get('num_chunks', len(info.get('chunk_ids', []))),
                'last_synced': info.get('last_synced', 'N/A'),
                'has_summary': info.get('has_summary', False),
                'summary_s3_key': info.get('summary_s3_key', '')
            })
        return docs
    
    def list_s3_documents(self) -> List[str]:
        """
        List all documents in S3 bucket under documents prefix for this application.
        
        Returns:
            List of S3 keys
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.documents_prefix
            )
            
            s3_keys = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    key = obj['Key']
                    # Skip directories and non-document files
                    if not key.endswith('/') and any(key.lower().endswith(ext) for ext in ['.pdf', '.docx', '.txt', '.md', '.xml', '.xlsx', '.xls', '.jpg', '.jpeg', '.png']):
                        s3_keys.append(key)
            
            return s3_keys
        except Exception as e:
            logger.error(f"[{self.app_name}] Error listing S3 documents: {e}")
            return []


class MultiAppAWSIngestionManagerWithSummarization:
    """
    Enhanced multi-application AWS ingestion manager with document summarization.
    """
    
    def __init__(self, app_name: str, config_path: str = "config/multi_app_config.yaml"):
        """
        Initialize enhanced multi-app AWS ingestion manager with summarization.
        
        Args:
            app_name: Application name
            config_path: Path to multi-app configuration file
        """
        self.app_name = app_name
        self.config_path = config_path
        
        # Initialize multi-app configuration
        logger.info(f"Initializing enhanced multi-app configuration for {app_name}...")
        self.config_manager = MultiAppConfigManager(config_path)
        
        # Validate application
        if not self.config_manager.validate_application(app_name):
            raise ValueError(f"Application '{app_name}' not found. Available: {self.config_manager.get_available_applications()}")
        
        self.app_config = self.config_manager.get_application_config(app_name)
        
        # Initialize AWS connections using legacy config
        logger.info(f"[{app_name}] Initializing AWS connections...")
        legacy_config = self.config_manager.create_legacy_config(app_name)
        
        # Create temporary config file for ConnectionManager
        temp_config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(legacy_config, temp_config_file, default_flow_style=False)
        temp_config_file.close()
        
        self.conn_manager = ConnectionManager(config_path=temp_config_file.name)
        
        # Clean up temp file
        os.unlink(temp_config_file.name)
        
        # Test connections
        connection_results = self.conn_manager.test_connections()
        for service, result in connection_results.items():
            if result['status'] == 'success':
                logger.info(f"[{app_name}] ✅ {service}: Connected successfully")
            else:
                logger.error(f"[{app_name}] ❌ {service}: {result['error']}")
                if service in ['opensearch', 'bedrock']:
                    raise Exception(f"Critical service {service} failed: {result['error']}")
        
        # Initialize components
        self.doc_loader = DocumentLoader(self.conn_manager)
        self.indexer = MultiAppOpenSearchIndexer(app_name=app_name, config_path=config_path)
        self.tracker = MultiAppS3DocumentTracker(app_name, self.config_manager)
        
        # Initialize summarization components
        self.summarizer = DocumentSummarizer(app_name, self.config_manager)
        self.inventory_manager = DocumentInventoryManager(app_name, self.config_manager, self.conn_manager.s3_client)
        
        # Ensure OpenSearch index exists
        if not self.indexer.create_index():
            logger.warning(f"[{app_name}] OpenSearch index creation returned False, but continuing...")
        
        logger.info(f"[{app_name}] Enhanced ingestion manager with summarization initialized")
    
    def scan_path_with_summarization(self, path: str = None, dry_run: bool = False, enable_summarization: bool = True) -> None:
        """
        Scan a path and index all new/modified documents with AI summarization.
        
        Args:
            path: S3 path prefix to scan (optional, defaults to app documents prefix)
            dry_run: If True, only show what would be indexed
            enable_summarization: If True, generate summaries for documents
        """
        # Use application's default documents prefix if no path provided
        if path is None:
            path = self.app_config['services']['s3']['documents_prefix']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"[{self.app_name}] ENHANCED SCANNING WITH SUMMARIZATION: {path}")
        if dry_run:
            logger.info("DRY RUN MODE - No documents will be indexed")
        if enable_summarization:
            logger.info("SUMMARIZATION ENABLED - AI summaries will be generated")
        logger.info(f"{'='*60}")
        
        # Get all documents in the specified S3 path
        try:
            response = self.conn_manager.s3_client.list_objects_v2(
                Bucket=self.tracker.bucket_name,
                Prefix=path
            )
            
            s3_documents = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    key = obj['Key']
                    # Skip directories and non-document files
                    if not key.endswith('/') and any(key.lower().endswith(ext) for ext in ['.pdf', '.docx', '.txt', '.md', '.xml', '.xlsx', '.xls', '.jpg', '.jpeg', '.png']):
                        s3_documents.append(key)
            
            if not s3_documents:
                logger.info(f"[{self.app_name}] No documents found in path: {path}")
                return
            
            logger.info(f"[{self.app_name}] Found {len(s3_documents)} documents in path")
            
            # Check for new/modified documents
            new_or_modified = []
            for s3_key in s3_documents:
                if self.tracker.has_changed(s3_key):
                    new_or_modified.append(s3_key)
            
            if not new_or_modified:
                logger.info(f"[{self.app_name}] No new or modified documents found")
                return
            
            logger.info(f"[{self.app_name}] Found {len(new_or_modified)} new/modified documents")
            
            if dry_run:
                logger.info(f"\n[DRY RUN] Documents that would be indexed:")
                for s3_key in new_or_modified:
                    filename = s3_key.split('/')[-1]
                    logger.info(f"  - {filename} (with summarization: {enable_summarization})")
                return
            
            # Index each new/modified document with summarization
            for i, s3_key in enumerate(new_or_modified, 1):
                filename = s3_key.split('/')[-1]
                logger.info(f"\n[{self.app_name}] [{i}/{len(new_or_modified)}] Processing {filename}...")
                try:
                    self._sync_document_with_summarization(s3_key, enable_summarization)
                except Exception as e:
                    logger.error(f"[{self.app_name}] Error processing {filename}: {e}")
                    continue
            
            # Update document inventory if summarization was enabled
            if enable_summarization:
                logger.info(f"\n[{self.app_name}] Updating document inventory...")
                try:
                    inventory_key = self.inventory_manager.update_document_inventory()
                    if inventory_key:
                        logger.info(f"[{self.app_name}] ✅ Document inventory updated: {inventory_key}")
                except Exception as e:
                    logger.error(f"[{self.app_name}] Error updating inventory: {e}")
            
            logger.info(f"\n[{self.app_name}] ✅ Enhanced scan complete! Processed {len(new_or_modified)} documents")
            
        except Exception as e:
            logger.error(f"[{self.app_name}] Error scanning path {path}: {e}")
    
    def _sync_document_with_summarization(self, s3_key: str, enable_summarization: bool = True) -> None:
        """
        Sync a single document from S3 to OpenSearch with optional summarization.
        
        Args:
            s3_key: S3 object key
            enable_summarization: Whether to generate AI summary
        """
        filename = s3_key.split('/')[-1]
        
        # Remove old chunks if document was previously indexed
        old_chunk_ids = self.tracker.get_chunk_ids(s3_key)
        if old_chunk_ids:
            logger.info(f"[{self.app_name}] Removing {len(old_chunk_ids)} old chunks...")
            for chunk_id in old_chunk_ids:
                try:
                    self.conn_manager.opensearch_client.delete(
                        index=self.tracker.index_name,
                        id=chunk_id
                    )
                except Exception as e:
                    logger.debug(f"[{self.app_name}] Could not delete chunk {chunk_id}: {e}")
        
        # Download and process the document from S3
        try:
            # Download file from S3 to temporary location
            temp_file = f"/tmp/{filename}"
            self.conn_manager.s3_client.download_file(
                self.tracker.bucket_name,
                s3_key,
                temp_file
            )
            
            # Load document
            document = self.doc_loader.load_document(temp_file)
            if not document:
                logger.error(f"[{self.app_name}] Failed to load document: {filename}")
                return
            
            # Generate summary if enabled
            summary_s3_key = ""
            if enable_summarization:
                logger.info(f"[{self.app_name}] Generating AI summary for {filename}...")
                try:
                    summary_data = self.summarizer.generate_summary(document)
                    summary_s3_key = self.summarizer.store_summary_to_s3(summary_data, document)
                    
                    # Add summary to document for OpenSearch indexing
                    document['document_summary'] = summary_data['summary']
                    document['summary_metadata'] = {
                        'key_terms': summary_data['key_terms'],
                        'topics': summary_data['topics'],
                        'relevance_score': summary_data['relevance_score'],
                        'confidence_score': summary_data['confidence_score'],
                        'generated_at': summary_data['generated_at'],
                        'model_used': summary_data['model_used'],
                        'summarization_method': summary_data['summarization_method']
                    }
                    
                    logger.info(f"[{self.app_name}] ✓ Generated summary for {filename} (relevance: {summary_data['relevance_score']}/10)")
                    
                except Exception as e:
                    logger.error(f"[{self.app_name}] Error generating summary for {filename}: {e}")
            
            # Check if it's an image file
            file_ext = Path(temp_file).suffix.lower()
            is_image = file_ext in ['.jpg', '.jpeg', '.png']
            
            if is_image:
                # For images, add descriptive content
                if not document.get('content'):
                    image_info = document.get('images', [{}])[0] if document.get('images') else {}
                    image_format = image_info.get('format', 'unknown')
                    image_size = image_info.get('size', (0, 0))
                    
                    document['content'] = f"Image: {filename}\nFormat: {image_format}\nSize: {image_size[0]}x{image_size[1]} pixels\nThis is a visual document that contains graphical information."
                
                # For images, create a single chunk with multimodal embedding
                chunk_id = f"{self.app_name}_{document['file_hash']}_img_0"
                chunk_ids = [chunk_id]
            else:
                # Get the chunk IDs that were created (estimate based on content)
                estimated_chunks = len(document.get('content', '')) // 1000 + 1
                chunk_ids = [f"{self.app_name}_{document['file_hash']}_{i}" for i in range(estimated_chunks)]
            
            # Index the document
            success = self.indexer.index_document(document)
            if not success:
                logger.error(f"[{self.app_name}] Failed to index document: {filename}")
                return
            
            # Update version tracking with summary information
            doc_id_to_track = document.doc_id if hasattr(document, 'doc_id') else document['file_hash']
            self.tracker.update_version(s3_key, doc_id_to_track, chunk_ids, summary_s3_key)
            self.tracker.save()
            
            # Clean up temporary file
            try:
                Path(temp_file).unlink()
            except FileNotFoundError:
                pass
            
            summary_info = f" + summary" if enable_summarization and summary_s3_key else ""
            logger.info(f"[{self.app_name}] ✓ Successfully indexed {filename} ({len(chunk_ids)} chunks{summary_info})")
            
        except Exception as e:
            logger.error(f"[{self.app_name}] Error syncing document {filename}: {e}")
            # Clean up temporary file on error
            try:
                Path(f"/tmp/{filename}").unlink()
            except FileNotFoundError:
                pass
    
    def update_inventory(self) -> None:
        """Update the document inventory in S3."""
        logger.info(f"\n{'='*60}")
        logger.info(f"[{self.app_name}] UPDATING DOCUMENT INVENTORY")
        logger.info(f"{'='*60}")
        
        try:
            inventory_key = self.inventory_manager.update_document_inventory()
            if inventory_key:
                logger.info(f"[{self.app_name}] ✅ Document inventory updated successfully")
                logger.info(f"[{self.app_name}] Inventory location: {inventory_key}")
            else:
                logger.error(f"[{self.app_name}] Failed to update document inventory")
        except Exception as e:
            logger.error(f"[{self.app_name}] Error updating inventory: {e}")
    
    # Delegate other methods to the original functionality
    def clean_orphaned_chunks(self, dry_run: bool = False) -> None:
        """
        Clean orphaned chunks and tracking entries for this application.
        
        Args:
            dry_run: If True, only show what would be cleaned
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"[{self.app_name}] CLEANING ORPHANED CHUNKS AND TRACKING ENTRIES")
        if dry_run:
            logger.info("DRY RUN MODE - No chunks or tracking entries will be deleted")
        logger.info(f"{'='*60}")
        
        # Get all S3 files for this application
        s3_files = set()
        s3_keys = set()
        try:
            response = self.conn_manager.s3_client.list_objects_v2(
                Bucket=self.tracker.bucket_name,
                Prefix=self.tracker.documents_prefix
            )
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    key = obj['Key']
                    filename = key.split('/')[-1]
                    s3_files.add(filename)
                    s3_keys.add(key)
            
            logger.info(f"[{self.app_name}] Found {len(s3_files)} files in S3")
            
        except Exception as e:
            logger.error(f"[{self.app_name}] Error listing S3 files: {e}")
            return
        
        # Find orphaned tracking entries (in tracking but not in S3)
        orphaned_tracking_keys = []
        for s3_key in self.tracker.versions.keys():
            if s3_key not in s3_keys:
                orphaned_tracking_keys.append(s3_key)
        
        # Get all chunks from OpenSearch for this application
        orphaned_chunks = []
        orphaned_files_from_chunks = set()
        try:
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {"match_all": {}},
                            {"term": {"application_id": self.app_name}}  # Filter by application
                        ]
                    }
                },
                "size": 10000,
                "_source": ["source_file", "file_path", "application_id"]
            }
            
            response = self.conn_manager.opensearch_client.search(
                index=self.tracker.index_name,
                body=query
            )
            
            chunks = response["hits"]["hits"]
            logger.info(f"[{self.app_name}] Found {len(chunks)} chunks in OpenSearch")
            
            # Find orphaned chunks (in OpenSearch but file not in S3)
            for chunk in chunks:
                source = chunk["_source"]
                source_file = source.get("source_file", "")
                file_path = source.get("file_path", "")
                
                chunk_file_path = source_file or file_path
                if chunk_file_path:
                    filename = chunk_file_path.split('/')[-1]
                    
                    if filename not in s3_files:
                        orphaned_chunks.append(chunk)
                        orphaned_files_from_chunks.add(filename)
            
        except Exception as e:
            logger.error(f"[{self.app_name}] Error querying OpenSearch: {e}")
            return
        
        # Combine all orphaned items
        total_orphaned_files = set()
        for key in orphaned_tracking_keys:
            filename = key.split('/')[-1]
            total_orphaned_files.add(filename)
        total_orphaned_files.update(orphaned_files_from_chunks)
        
        if not orphaned_chunks and not orphaned_tracking_keys:
            logger.info(f"[{self.app_name}] ✅ No orphaned chunks or tracking entries found")
            return
        
        logger.info(f"[{self.app_name}] Found {len(orphaned_chunks)} orphaned chunks from {len(orphaned_files_from_chunks)} files")
        logger.info(f"[{self.app_name}] Found {len(orphaned_tracking_keys)} orphaned tracking entries")
        logger.info(f"[{self.app_name}] Total orphaned files: {len(total_orphaned_files)}")
        
        if dry_run:
            logger.info(f"\n[DRY RUN] Items that would be cleaned:")
            
            if orphaned_chunks:
                logger.info(f"  OpenSearch chunks to delete:")
                for filename in sorted(orphaned_files_from_chunks):
                    chunk_count = sum(1 for chunk in orphaned_chunks 
                                    if (chunk["_source"].get("source_file", "") or chunk["_source"].get("file_path", "")).endswith(filename))
                    logger.info(f"    - {filename} ({chunk_count} chunks)")
            
            if orphaned_tracking_keys:
                logger.info(f"  Tracking entries to remove:")
                for key in sorted(orphaned_tracking_keys):
                    filename = key.split('/')[-1]
                    logger.info(f"    - {filename} (tracking entry)")
            
            return
        
        # Delete orphaned chunks from OpenSearch
        deleted_chunks_count = 0
        if orphaned_chunks:
            for chunk in orphaned_chunks:
                try:
                    self.conn_manager.opensearch_client.delete(
                        index=self.tracker.index_name,
                        id=chunk["_id"]
                    )
                    deleted_chunks_count += 1
                except Exception as e:
                    logger.error(f"[{self.app_name}] Error deleting chunk {chunk['_id']}: {e}")
        
        # Remove orphaned tracking entries
        deleted_tracking_count = 0
        for key in orphaned_tracking_keys:
            try:
                self.tracker.remove_document(key)
                deleted_tracking_count += 1
            except Exception as e:
                logger.error(f"[{self.app_name}] Error removing tracking entry {key}: {e}")
        
        # Also clean tracking entries for files that had chunks deleted
        for filename in orphaned_files_from_chunks:
            keys_to_remove = []
            for s3_key in self.tracker.versions.keys():
                if s3_key.endswith(filename):
                    keys_to_remove.append(s3_key)
            
            for key in keys_to_remove:
                self.tracker.remove_document(key)
        
        self.tracker.save()
        
        logger.info(f"[{self.app_name}] ✅ Cleaned {deleted_chunks_count} orphaned chunks and {deleted_tracking_count} orphaned tracking entries")
        logger.info(f"[{self.app_name}]    Total orphaned files processed: {len(total_orphaned_files)}")
    
    def remove_by_path(self, path: str, dry_run: bool = False) -> None:
        """
        Remove documents under a specific path from OpenSearch and tracking for this application.
        
        Args:
            path: Path prefix to remove
            dry_run: If True, only show what would be removed
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"[{self.app_name}] REMOVING BY PATH: {path}")
        if dry_run:
            logger.info("DRY RUN MODE - No documents will be removed")
        logger.info(f"{'='*60}")
        
        # Find matching documents in tracking
        matching_keys = []
        for s3_key in self.tracker.versions.keys():
            if path in s3_key:
                matching_keys.append(s3_key)
        
        # Also find matching documents directly in OpenSearch with application filter
        opensearch_matches = []
        try:
            search_body = {
                "size": 10000,
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"application_id": self.app_name}}  # Filter by application
                        ],
                        "should": [
                            {"wildcard": {"file_name": f"*{path}*"}},
                            {"wildcard": {"metadata.s3_key": f"*{path}*"}},
                            {"wildcard": {"source_file": f"*{path}*"}},
                            {"wildcard": {"file_path": f"*{path}*"}}
                        ],
                        "minimum_should_match": 1
                    }
                },
                "_source": ["file_name", "metadata", "source_file", "file_path", "application_id"]
            }
            
            response = self.conn_manager.opensearch_client.search(
                index=self.tracker.index_name,
                body=search_body
            )
            
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                # Try multiple fields to get file path
                file_path = (source.get("source_file") or 
                           source.get("file_path") or 
                           source.get("file_name", ""))
                
                if file_path and path in file_path:
                    opensearch_matches.append({
                        'chunk_id': hit["_id"],
                        'file_path': file_path,
                        'filename': file_path.split('/')[-1]
                    })
            
            logger.info(f"[{self.app_name}] Found {len(opensearch_matches)} chunks in OpenSearch matching path")
            
        except Exception as e:
            logger.error(f"[{self.app_name}] Error searching OpenSearch: {e}")
            opensearch_matches = []
        
        if not matching_keys and not opensearch_matches:
            logger.info(f"[{self.app_name}] No documents found matching path: {path}")
            return
        
        logger.info(f"[{self.app_name}] Found {len(matching_keys)} tracked documents and {len(opensearch_matches)} chunks matching path")
        
        if dry_run:
            logger.info(f"\n[DRY RUN] Items that would be removed:")
            
            if matching_keys:
                logger.info(f"  Tracked documents:")
                for s3_key in matching_keys:
                    filename = s3_key.split('/')[-1]
                    num_chunks = len(self.tracker.get_chunk_ids(s3_key))
                    logger.info(f"    - {filename} ({num_chunks} chunks from tracking)")
            
            if opensearch_matches:
                logger.info(f"  OpenSearch chunks:")
                file_chunks = {}
                for match in opensearch_matches:
                    filename = match['filename']
                    if filename not in file_chunks:
                        file_chunks[filename] = 0
                    file_chunks[filename] += 1
                
                for filename, count in file_chunks.items():
                    logger.info(f"    - {filename} ({count} chunks from OpenSearch)")
            
            return
        
        # PHASE 1: Remove tracked documents
        removed_count = 0
        total_chunks_removed = 0
        
        # Collect all chunk_ids FIRST, before modifying tracking
        documents_to_remove = []
        for s3_key in matching_keys:
            filename = s3_key.split('/')[-1]
            chunk_ids = self.tracker.get_chunk_ids(s3_key)
            documents_to_remove.append({
                's3_key': s3_key,
                'filename': filename,
                'chunk_ids': chunk_ids
            })
        
        # Now remove from OpenSearch and tracking
        for doc_info in documents_to_remove:
            s3_key = doc_info['s3_key']
            filename = doc_info['filename']
            chunk_ids = doc_info['chunk_ids']
            
            try:
                # Remove from OpenSearch using tracked chunk_ids
                chunks_removed_for_doc = 0
                for chunk_id in chunk_ids:
                    try:
                        self.conn_manager.opensearch_client.delete(
                            index=self.tracker.index_name,
                            id=chunk_id
                        )
                        chunks_removed_for_doc += 1
                        total_chunks_removed += 1
                    except Exception as e:
                        logger.warning(f"[{self.app_name}] Could not delete tracked chunk {chunk_id}: {e}")
                
                # Remove from tracking
                self.tracker.remove_document(s3_key)
                removed_count += 1
                logger.info(f"[{self.app_name}]   ✓ Removed {filename} ({chunks_removed_for_doc}/{len(chunk_ids)} tracked chunks)")
                
            except Exception as e:
                logger.error(f"[{self.app_name}]   ✗ Failed to remove {filename}: {e}")
        
        # PHASE 2: Remove any remaining chunks directly from OpenSearch
        untracked_chunks_removed = 0
        processed_files = set()
        
        for match in opensearch_matches:
            chunk_id = match['chunk_id']
            filename = match['filename']
            
            # Skip if this file was already processed in Phase 1
            file_already_processed = any(doc['filename'] == filename for doc in documents_to_remove)
            if file_already_processed:
                continue
            
            try:
                self.conn_manager.opensearch_client.delete(
                    index=self.tracker.index_name,
                    id=chunk_id
                )
                untracked_chunks_removed += 1
                processed_files.add(filename)
            except Exception as e:
                logger.warning(f"[{self.app_name}] Could not delete OpenSearch chunk {chunk_id}: {e}")
        
        if untracked_chunks_removed > 0:
            logger.info(f"[{self.app_name}]   ✓ Removed {untracked_chunks_removed} additional chunks from OpenSearch for {len(processed_files)} files")
        
        self.tracker.save()
        
        total_removed = total_chunks_removed + untracked_chunks_removed
        logger.info(f"[{self.app_name}] ✅ Removed {removed_count} tracked documents ({total_chunks_removed} tracked chunks) + {untracked_chunks_removed} OpenSearch chunks = {total_removed} total chunks")
    
    def remove_by_id(self, doc_id: str, dry_run: bool = False) -> None:
        """
        Remove a document by ID from OpenSearch and tracking for this application.
        
        Args:
            doc_id: Document ID to remove
            dry_run: If True, only show what would be removed
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"[{self.app_name}] REMOVING BY ID: {doc_id}")
        if dry_run:
            logger.info("DRY RUN MODE - No document will be removed")
        logger.info(f"{'='*60}")
        
        # Find document by ID
        s3_key = self.tracker.get_s3_key_by_doc_id(doc_id)
        if not s3_key:
            logger.error(f"[{self.app_name}] Document with ID {doc_id} not found in tracking")
            return
        
        filename = s3_key.split('/')[-1]
        chunk_ids = self.tracker.get_chunk_ids_by_doc_id(doc_id)
        
        logger.info(f"[{self.app_name}] Found document: {filename} ({len(chunk_ids)} chunks)")
        
        if dry_run:
            logger.info(f"[DRY RUN] Would remove: {filename} ({len(chunk_ids)} chunks)")
            return
        
        # Remove from OpenSearch
        removed_chunks = 0
        for chunk_id in chunk_ids:
            try:
                self.conn_manager.opensearch_client.delete(
                    index=self.tracker.index_name,
                    id=chunk_id
                )
                removed_chunks += 1
            except Exception as e:
                logger.debug(f"[{self.app_name}] Could not delete chunk {chunk_id}: {e}")
        
        # Remove from tracking
        self.tracker.remove_document(s3_key)
        self.tracker.save()
        
        logger.info(f"[{self.app_name}] ✅ Removed {filename} ({removed_chunks} chunks)")
    
    def remove_by_pattern(self, pattern: str, dry_run: bool = False) -> None:
        """
        Remove documents matching a regex pattern from OpenSearch and tracking for this application.
        
        Args:
            pattern: Regex pattern to match
            dry_run: If True, only show what would be removed
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"[{self.app_name}] REMOVING BY PATTERN: {pattern}")
        if dry_run:
            logger.info("DRY RUN MODE - No documents will be removed")
        logger.info(f"{'='*60}")
        
        try:
            regex = re.compile(pattern)
        except re.error as e:
            logger.error(f"[{self.app_name}] Invalid regex pattern: {e}")
            return
        
        # Find matching documents in tracking
        matching_keys = []
        for s3_key in self.tracker.versions.keys():
            if regex.search(s3_key):
                matching_keys.append(s3_key)
        
        # Also find matching documents directly in OpenSearch with application filter
        opensearch_matches = []
        try:
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {"match_all": {}},
                            {"term": {"application_id": self.app_name}}  # Filter by application
                        ]
                    }
                },
                "size": 10000,
                "_source": ["source_file", "file_path", "application_id"]
            }
            
            response = self.conn_manager.opensearch_client.search(
                index=self.tracker.index_name,
                body=query
            )
            
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                file_path = source.get("source_file") or source.get("file_path", "")
                if file_path and regex.search(file_path):
                    opensearch_matches.append({
                        'chunk_id': hit["_id"],
                        'file_path': file_path,
                        'filename': file_path.split('/')[-1]
                    })
            
            logger.info(f"[{self.app_name}] Found {len(opensearch_matches)} chunks in OpenSearch matching pattern")
            
        except Exception as e:
            logger.error(f"[{self.app_name}] Error searching OpenSearch: {e}")
            opensearch_matches = []
        
        if not matching_keys and not opensearch_matches:
            logger.info(f"[{self.app_name}] No documents found matching pattern: {pattern}")
            return
        
        logger.info(f"[{self.app_name}] Found {len(matching_keys)} tracked documents and {len(opensearch_matches)} chunks matching pattern")
        
        if dry_run:
            logger.info(f"\n[DRY RUN] Items that would be removed:")
            
            if matching_keys:
                logger.info(f"  Tracked documents:")
                for s3_key in matching_keys:
                    filename = s3_key.split('/')[-1]
                    num_chunks = len(self.tracker.get_chunk_ids(s3_key))
                    logger.info(f"    - {filename} ({num_chunks} chunks from tracking)")
            
            if opensearch_matches:
                logger.info(f"  OpenSearch chunks:")
                file_chunks = {}
                for match in opensearch_matches:
                    filename = match['filename']
                    if filename not in file_chunks:
                        file_chunks[filename] = 0
                    file_chunks[filename] += 1
                
                for filename, count in file_chunks.items():
                    logger.info(f"    - {filename} ({count} chunks from OpenSearch)")
            
            return
        
        # PHASE 1: Remove tracked documents
        removed_count = 0
        total_chunks_removed = 0
        
        # Collect all chunk_ids FIRST, before modifying tracking
        documents_to_remove = []
        for s3_key in matching_keys:
            filename = s3_key.split('/')[-1]
            chunk_ids = self.tracker.get_chunk_ids(s3_key)
            documents_to_remove.append({
                's3_key': s3_key,
                'filename': filename,
                'chunk_ids': chunk_ids
            })
        
        # Now remove from OpenSearch and tracking
        for doc_info in documents_to_remove:
            s3_key = doc_info['s3_key']
            filename = doc_info['filename']
            chunk_ids = doc_info['chunk_ids']
            
            try:
                # Remove from OpenSearch
                chunks_removed_for_doc = 0
                for chunk_id in chunk_ids:
                    try:
                        self.conn_manager.opensearch_client.delete(
                            index=self.tracker.index_name,
                            id=chunk_id
                        )
                        chunks_removed_for_doc += 1
                        total_chunks_removed += 1
                    except Exception as e:
                        logger.warning(f"[{self.app_name}] Could not delete chunk {chunk_id}: {e}")
                
                # Remove from tracking
                self.tracker.remove_document(s3_key)
                removed_count += 1
                logger.info(f"[{self.app_name}]   ✓ Removed {filename} ({chunks_removed_for_doc}/{len(chunk_ids)} chunks)")
                
            except Exception as e:
                logger.error(f"[{self.app_name}]   ✗ Failed to remove {filename}: {e}")
        
        # PHASE 2: Remove untracked chunks directly from OpenSearch
        untracked_chunks_removed = 0
        processed_files = set()
        
        for match in opensearch_matches:
            chunk_id = match['chunk_id']
            filename = match['filename']
            
            # Skip if this file was already processed in Phase 1
            file_already_processed = any(doc['filename'] == filename for doc in documents_to_remove)
            if file_already_processed:
                continue
            
            try:
                self.conn_manager.opensearch_client.delete(
                    index=self.tracker.index_name,
                    id=chunk_id
                )
                untracked_chunks_removed += 1
                processed_files.add(filename)
            except Exception as e:
                logger.warning(f"[{self.app_name}] Could not delete untracked chunk {chunk_id}: {e}")
        
        if untracked_chunks_removed > 0:
            logger.info(f"[{self.app_name}]   ✓ Removed {untracked_chunks_removed} untracked chunks from {len(processed_files)} files")
        
        self.tracker.save()
        
        total_removed = total_chunks_removed + untracked_chunks_removed
        logger.info(f"[{self.app_name}] ✅ Removed {removed_count} tracked documents ({total_chunks_removed} chunks) + {untracked_chunks_removed} untracked chunks = {total_removed} total chunks")
    
    def list_comprehensive(self) -> None:
        """
        List comprehensive status showing OpenSearch, Tracking, and S3 status for this application.
        Enhanced version includes summary information.
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"[{self.app_name}] COMPREHENSIVE DOCUMENT STATUS (Enhanced with Summaries)")
        logger.info(f"{'='*80}")
        
        # Get data from all sources
        tracking_docs = {info['s3_key']: info for info in self.tracker.list_all_documents()}
        s3_files = set(self.tracker.list_s3_documents())
        
        # Get OpenSearch documents for this application
        opensearch_docs = {}
        try:
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {"match_all": {}},
                            {"term": {"application_id": self.app_name}}  # Filter by application
                        ]
                    }
                },
                "size": 10000,
                "_source": ["source_file", "file_path", "application_id"]
            }
            
            response = self.conn_manager.opensearch_client.search(
                index=self.tracker.index_name,
                body=query
            )
            
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                file_path = source.get("source_file") or source.get("file_path", "")
                if file_path:
                    if file_path not in opensearch_docs:
                        opensearch_docs[file_path] = 0
                    opensearch_docs[file_path] += 1
                    
        except Exception as e:
            logger.error(f"[{self.app_name}] Error querying OpenSearch: {e}")
        
        # Combine all files
        all_files = set()
        all_files.update(tracking_docs.keys())
        all_files.update(s3_files)
        
        # Add files from OpenSearch
        for file_path in opensearch_docs.keys():
            # Try to match with S3 keys first
            matched = False
            for s3_file in s3_files:
                if file_path.endswith(s3_file.split('/')[-1]):
                    all_files.add(s3_file)
                    matched = True
                    break
            
            # If no S3 match found, add the OpenSearch file path directly
            if not matched:
                all_files.add(file_path)
        
        if not all_files:
            logger.info(f"[{self.app_name}] No files found in any source")
            return
        
        # Prepare table data
        table_data = []
        for file_key in sorted(all_files):
            filename = file_key.split('/')[-1]
            
            # Check tracking
            in_tracking = "✅" if file_key in tracking_docs else "❌"
            tracking_chunks = tracking_docs[file_key]['num_chunks'] if file_key in tracking_docs else 0
            has_summary = "✅" if file_key in tracking_docs and tracking_docs[file_key].get('has_summary', False) else "❌"
            
            # Check S3
            in_s3 = "✅" if file_key in s3_files else "❌"
            
            # Check OpenSearch (match by filename)
            opensearch_chunks = 0
            in_opensearch = "❌"
            for os_path, chunk_count in opensearch_docs.items():
                if os_path.endswith(filename):
                    opensearch_chunks = chunk_count
                    in_opensearch = "✅"
                    break
            
            # Status summary
            status = "🟢 OK"
            if in_tracking == "❌" and in_opensearch == "✅":
                status = "🟡 UNTRACKED"
            elif in_s3 == "❌" and (in_tracking == "✅" or in_opensearch == "✅"):
                status = "🔴 ORPHANED"
            elif in_opensearch == "❌" and in_tracking == "✅":
                status = "🟠 NOT_INDEXED"
            
            table_data.append([
                filename[:35] + "..." if len(filename) > 35 else filename,
                in_opensearch,
                opensearch_chunks,
                in_tracking,
                tracking_chunks,
                has_summary,
                in_s3,
                status
            ])
        
        # Display table
        headers = ['Filename', 'OpenSearch', 'OS Chunks', 'Tracking', 'TR Chunks', 'Summary', 'S3', 'Status']
        print(f"\n[{self.app_name}] Document Status (Enhanced):")
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        
        # Summary
        total_files = len(all_files)
        ok_files = sum(1 for row in table_data if row[7] == "🟢 OK")
        orphaned_files = sum(1 for row in table_data if row[7] == "🔴 ORPHANED")
        untracked_files = sum(1 for row in table_data if row[7] == "🟡 UNTRACKED")
        not_indexed_files = sum(1 for row in table_data if row[7] == "🟠 NOT_INDEXED")
        files_with_summaries = sum(1 for row in table_data if row[5] == "✅")
        
        logger.info(f"\n[{self.app_name}] 📊 ENHANCED SUMMARY:")
        logger.info(f"Total files: {total_files}")
        logger.info(f"🟢 OK: {ok_files}")
        logger.info(f"🔴 Orphaned: {orphaned_files}")
        logger.info(f"🟡 Untracked: {untracked_files}")
        logger.info(f"🟠 Not indexed: {not_indexed_files}")
        logger.info(f"📝 With AI summaries: {files_with_summaries}")
        logger.info(f"📊 Summary coverage: {files_with_summaries}/{total_files} ({(files_with_summaries/max(total_files,1)*100):.1f}%)")


def list_applications(config_path: str = "config/multi_app_config.yaml") -> None:
    """List all available applications."""
    try:
        config_manager = MultiAppConfigManager(config_path)
        apps = config_manager.get_available_applications()
        
        logger.info(f"\n📱 Available Applications:")
        logger.info("=" * 60)
        
        for app_name in apps:
            app_config = config_manager.get_application_config(app_name)
            # Get the raw application config for display
            raw_app_config = config_manager.applications.get(app_name, {})
            
            logger.info(f"🔹 ID: {app_name}")
            logger.info(f"   Name: {raw_app_config.get('name', app_name.upper())}")
            logger.info(f"   Description: {raw_app_config.get('description', 'No description available')}")
            logger.info(f"   Index: {app_config['opensearch']['index_name']}")
            logger.info(f"   S3 Bucket: {app_config['services']['s3']['bucket']}")
            logger.info(f"   S3 Prefix: {app_config['services']['s3']['documents_prefix']}")
            logger.info("-" * 40)
        
    except Exception as e:
        logger.error(f"Error listing applications: {e}")


def main():
    """Main function with command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enhanced Multi-Application AWS Ingestion Manager with Document Summarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

LISTAR APLICACIONES:
  # Ver aplicaciones disponibles
  python multi_app_aws_ingestion_manager_with_summarization.py list-apps

INDEXACIÓN CON RESÚMENES:
  # Escanear e indexar documentos con generación de resúmenes
  python multi_app_aws_ingestion_manager_with_summarization.py --app sap scan
  
  # Escanear sin generar resúmenes
  python multi_app_aws_ingestion_manager_with_summarization.py --app sap scan --no-summarization
  
  # Escanear ruta específica con resúmenes
  python multi_app_aws_ingestion_manager_with_summarization.py --app darwin scan --path documents/nuevos/
  
  # Modo dry-run para ver qué se indexaría
  python multi_app_aws_ingestion_manager_with_summarization.py --app sap scan --dry-run

GESTIÓN DE INVENTARIO:
  # Actualizar inventario de documentos
  python multi_app_aws_ingestion_manager_with_summarization.py --app sap update-inventory

FUNCIONES HEREDADAS:
  # Limpiar chunks huérfanos
  python multi_app_aws_ingestion_manager_with_summarization.py --app sap clean
  
  # Eliminar documentos por ruta
  python multi_app_aws_ingestion_manager_with_summarization.py --app sap remove --path documents/old/
  
  # Ver estado completo de documentos
  python multi_app_aws_ingestion_manager_with_summarization.py --app sap list
        """
    )
    
    # Global options
    parser.add_argument(
        '--config',
        default='config/multi_app_config.yaml',
        help='Archivo de configuración multi-aplicación (default: config/multi_app_config.yaml)'
    )
    
    parser.add_argument(
        '--app',
        help='Aplicación a usar (requerido para todos los comandos excepto list-apps)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comandos disponibles')
    
    # List-apps command
    list_apps_parser = subparsers.add_parser('list-apps', help='Listar aplicaciones disponibles')
    
    # Enhanced scan command
    scan_parser = subparsers.add_parser('scan', help='Escanear e indexar documentos con resúmenes')
    scan_parser.add_argument('--path', help='Ruta S3 a escanear (opcional, usa prefix por defecto de la app)')
    scan_parser.add_argument('--dry-run', action='store_true', help='Solo mostrar qué se indexaría')
    scan_parser.add_argument('--no-summarization', action='store_true', help='Desactivar generación de resúmenes')
    
    # Update inventory command
    inventory_parser = subparsers.add_parser('update-inventory', help='Actualizar inventario de documentos')
    
    # Legacy commands (simplified for now)
    clean_parser = subparsers.add_parser('clean', help='Limpiar chunks huérfanos')
    clean_parser.add_argument('--dry-run', action='store_true', help='Solo mostrar qué se limpiaría')
    
    remove_parser = subparsers.add_parser('remove', help='Eliminar documentos por ruta')
    remove_parser.add_argument('--path', required=True, help='Ruta a eliminar')
    remove_parser.add_argument('--dry-run', action='store_true', help='Solo mostrar qué se eliminaría')
    
    # Remove-by-id command
    remove_id_parser = subparsers.add_parser('remove-by-id', help='Eliminar documento por ID')
    remove_id_parser.add_argument('id', help='ID del documento a eliminar')
    remove_id_parser.add_argument('--dry-run', action='store_true', help='Solo mostrar qué se eliminaría')
    
    # Remove-pattern command
    remove_pattern_parser = subparsers.add_parser('remove-pattern', help='Eliminar documentos por patrón regex')
    remove_pattern_parser.add_argument('pattern', help='Patrón regex para buscar documentos')
    remove_pattern_parser.add_argument('--dry-run', action='store_true', help='Solo mostrar qué se eliminaría')
    
    list_parser = subparsers.add_parser('list', help='Listar estado completo de documentos')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Handle list-apps command (doesn't require --app)
    if args.command == 'list-apps':
        list_applications(config_path=args.config)
        return 0
    
    # All other commands require --app
    if not args.app:
        logger.error("Error: --app is required for all commands except 'list-apps'")
        parser.print_help()
        return 1
    
    # Initialize enhanced manager
    try:
        manager = MultiAppAWSIngestionManagerWithSummarization(app_name=args.app, config_path=args.config)
    except Exception as e:
        logger.error(f"Failed to initialize Enhanced Multi-App AWS Ingestion Manager: {e}")
        return 1
    
    # Execute command
    try:
        if args.command == 'scan':
            enable_summarization = not args.no_summarization
            manager.scan_path_with_summarization(args.path, dry_run=args.dry_run, enable_summarization=enable_summarization)
        
        elif args.command == 'update-inventory':
            manager.update_inventory()
        
        elif args.command == 'clean':
            manager.clean_orphaned_chunks(dry_run=args.dry_run)
        
        elif args.command == 'remove':
            manager.remove_by_path(args.path, dry_run=args.dry_run)
        
        elif args.command == 'remove-by-id':
            manager.remove_by_id(args.id, dry_run=args.dry_run)
        
        elif args.command == 'remove-pattern':
            manager.remove_by_pattern(args.pattern, dry_run=args.dry_run)
        
        elif args.command == 'list':
            manager.list_comprehensive()
        
        return 0
        
    except Exception as e:
        logger.error(f"Error executing command '{args.command}': {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
