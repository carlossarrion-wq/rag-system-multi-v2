#!/usr/bin/env python3
"""
Multi-Application Conversational RAG Chat Interface
Supports multiple applications with separate indices, S3 buckets, and system prompts
"""

import sys
import argparse
import time
from typing import Dict, Any, List, Generator, Union, Optional
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.multi_app_config_manager import MultiAppConfigManager
from src.agent.advanced_conversational_agent import AdvancedConversationalAgent, ConversationResponse
from src.agent.advanced_memory import MemoryContext
from src.agent.reasoning_agent_fixed import ReasoningResult
from src.agent.document_context_enhancer import DocumentContextEnhancer
from src.utils.connection_manager import ConnectionManager
from src.generation.structured_response_parser import StructuredResponseParser
import tempfile
import os
import yaml


class MultiAppStreamingConversationResponse:
    """Respuesta de conversación con streaming para multi-aplicación"""
    
    def __init__(self, 
                 stream_generator: Generator,
                 confidence_score: float,
                 confidence_level: str,
                 confidence_emoji: str,
                 sources: List[Dict[str, Any]],
                 reasoning_trace: List[str],
                 memory_context: str,
                 execution_time: float,
                 metadata: Dict[str, Any],
                 application_info: Dict[str, str]):
        self.stream_generator = stream_generator
        self.confidence_score = confidence_score
        self.confidence_level = confidence_level
        self.confidence_emoji = confidence_emoji
        self.sources = sources
        self.reasoning_trace = reasoning_trace
        self.memory_context = memory_context
        self.execution_time = execution_time
        self.metadata = metadata
        self.application_info = application_info
        self.answer = ""  # Se llenará con el streaming
    
    def get_stream(self):
        """Obtiene el generador de streaming"""
        return self.stream_generator


class MultiAppConversationalAgent(AdvancedConversationalAgent):
    """Agente conversacional multi-aplicación con soporte para streaming"""
    
    def __init__(self, app_name: Optional[str] = None, 
                 config_path: str = "config/multi_app_config.yaml"):
        """
        Initialize multi-application conversational agent.
        
        Args:
            app_name: Name of the application to use
            config_path: Path to multi-application configuration
        """
        self.config_manager = MultiAppConfigManager(config_path)
        self.app_name = app_name or self.config_manager.default_app
        self.application_info = self.config_manager.get_application_info(self.app_name)
        
        # Validate application
        if not self.config_manager.validate_application(self.app_name):
            available_apps = ', '.join(self.config_manager.get_available_applications())
            raise ValueError(f"Application '{self.app_name}' not found. Available: {available_apps}")
        
        # Create temporary legacy config file for compatibility
        self.legacy_config = self.config_manager.create_legacy_config(self.app_name)
        self.temp_config_file = self._create_temp_config_file()
        
        # Initialize components needed for parent class
        from src.generation.llm_client_fixed import LLMClient
        from src.generation.citation_manager_fixed import FixedCitationManager as CitationManager
        from src.retrieval.specialized_retrievers import RetrieverFactory
        from src.utils.config_loader import ConfigLoader
        
        # Load config using the temporary config file
        config_loader = ConfigLoader()
        config = config_loader.load_config(self.temp_config_file)
        
        # Initialize connection manager
        connection_manager = ConnectionManager(config_path=self.temp_config_file)
        
        # Initialize components with correct parameters
        llm_client = LLMClient(config_path=self.temp_config_file)
        
        # RetrieverFactory is a static class, just pass the class itself
        retriever_factory = RetrieverFactory
        
        citation_manager = CitationManager()
        
        # Initialize document context enhancer for context reduction
        try:
            document_enhancer = DocumentContextEnhancer(self.app_name, config_path)
            # Connect document enhancer to LLM client for context reduction
            llm_client.document_enhancer = document_enhancer
            print(f"🔗 Document enhancer connected to LLM client for context reduction")
        except Exception as e:
            print(f"⚠️  Warning: Could not connect document enhancer to LLM client: {e}")
        
        # Initialize parent class with proper components
        super().__init__(
            llm_client=llm_client,
            retriever_factory=retriever_factory,
            citation_manager=citation_manager,
            memory_file=f"data/memory/conversation_memory_{self.app_name}.json",
            config_path=self.temp_config_file
        )
        
        # Override system prompt with application-specific one
        self.system_prompt = self.config_manager.get_system_prompt(self.app_name)
        
        # Override LLM client's system prompt method to use application-specific prompt
        original_get_system_prompt = self.llm_client._get_system_prompt
        self.llm_client._get_system_prompt = lambda custom_prompt=None: custom_prompt or self.system_prompt
        
        # Initialize document context enhancer for 2048+ char context (Haiku 3 caching)
        try:
            self.context_enhancer = DocumentContextEnhancer(self.app_name, config_path)
            print(f"📋 Document context enhancer initialized (Haiku 3 caching optimization)")
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize context enhancer: {e}")
            self.context_enhancer = None
        
        # Initialize structured response parser
        self.response_parser = StructuredResponseParser()
        
        print(f"🚀 Multi-App RAG Agent initialized")
        print(f"📱 Application: {self.application_info['name']}")
        print(f"🔍 Index: {self.config_manager.get_opensearch_index_name(self.app_name)}")
        print(f"📦 S3 Bucket: {self.config_manager.get_s3_config(self.app_name)['bucket']}")
        print(f"💬 Custom system prompt loaded")
        print("-" * 60)
    
    def _create_temp_config_file(self) -> str:
        """Create temporary configuration file for legacy compatibility."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(self.legacy_config, temp_file, default_flow_style=False)
        temp_file.close()
        return temp_file.name
    
    def __del__(self):
        """Clean up temporary configuration file."""
        if hasattr(self, 'temp_config_file') and os.path.exists(self.temp_config_file):
            os.unlink(self.temp_config_file)
    
    def process_query(
        self,
        query: str,
        stream: bool = False,
        max_results: int = 10
    ) -> Union[ConversationResponse, MultiAppStreamingConversationResponse]:
        """
        Process query with application-specific configuration.
        
        Args:
            query: User query
            stream: Whether to use streaming
            max_results: Maximum number of search results
            
        Returns:
            ConversationResponse or MultiAppStreamingConversationResponse
        """
        start_time = time.time()

        try:
            # Get application-specific RAG config
            rag_config = self.config_manager.get_rag_config(self.app_name)
            max_results = min(max_results, rag_config.get('search', {}).get('max_results', 8))
            
            # PASO 1: Obtener contexto de memoria
            memory_context = self.memory.get_relevant_context(query, max_turns=5)

            # PASO 2: Análisis de razonamiento
            conversation_history = self._format_memory_for_reasoning(memory_context)
            reasoning_result = self.reasoning_agent.analyze_query(
                query,
                conversation_history=conversation_history
            )

            # PASO 3: Verificar si necesita clarificación
            if self.reasoning_agent.should_ask_clarification(reasoning_result):
                clarification = self.reasoning_agent.generate_clarification_question(reasoning_result, query)
                return self._create_clarification_response(clarification, reasoning_result, time.time() - start_time)

            # PASO 4: Ejecutar estrategia de búsqueda
            orchestration_result = self.tool_orchestrator.execute_strategy(
                reasoning=reasoning_result,
                original_query=query,
                top_k=max_results
            )

            # PASO 5: Manejar respuestas conversacionales
            if orchestration_result.strategy_used == "conversational":
                return self._generate_conversational_response(
                    query=query,
                    memory_context=memory_context,
                    reasoning_result=reasoning_result,
                    start_time=start_time,
                    stream=stream
                )

            # PASO 6: Procesar citas (antes de generar respuesta)
            processed_sources = self.citation_manager.process_sources(
                orchestration_result.final_results
            )

            # PASO 7: Generar respuesta (con o sin streaming)
            if stream:
                return self._generate_streaming_response_multi_app(
                    query=query,
                    search_results=orchestration_result.final_results,
                    memory_context=memory_context,
                    reasoning_result=reasoning_result,
                    processed_sources=processed_sources,
                    start_time=start_time
                )
            else:
                # Modo no-streaming con prompt personalizado
                response_data = self._generate_response_with_confidence_multi_app(
                    query,
                    search_results=orchestration_result.final_results,
                    memory_context=memory_context,
                    reasoning_result=reasoning_result
                )

                # Crear respuesta final
                final_response = ConversationResponse(
                    answer=response_data['answer'],
                    confidence_score=response_data['confidence_score'],
                    confidence_level=response_data['confidence_level'],
                    confidence_emoji=response_data['confidence_emoji'],
                    sources=processed_sources,
                    reasoning_trace=reasoning_result.reasoning_trace,
                    memory_context=memory_context.conversation_summary,
                    execution_time=time.time() - start_time,
                    metadata={
                        'search_strategy': reasoning_result.search_strategy,
                        'tools_used': reasoning_result.tools_to_use,
                        'results_found': len(orchestration_result.final_results),
                        'memory_confidence': memory_context.memory_confidence,
                        'reasoning_confidence': reasoning_result.confidence,
                        'application': self.application_info,
                        'confidence_rationale': response_data.get('confidence_rationale', ''),
                        # Add structured response data
                        'structured_response': response_data.get('structured_response'),
                        'display_data': response_data.get('display_data'),
                        'is_structured': response_data.get('is_structured', False),
                        'key_points': response_data.get('key_points', []),
                        'follow_up_questions': response_data.get('follow_up_questions', []),
                        'related_topics': response_data.get('related_topics', []),
                        'warnings': response_data.get('warnings', [])
                    }
                )

                # Actualizar memoria con respuesta limpia
                # CRITICAL: Always save only the clean answer content, never the raw JSON
                structured_data = final_response.metadata.get('structured_response')
                if structured_data and 'answer' in structured_data:
                    # Use the clean answer from structured JSON
                    clean_answer = structured_data['answer']
                else:
                    # For non-structured responses, try to parse the raw answer to extract clean content
                    try:
                        parsed_data, is_structured = self.response_parser.parse_response(final_response.answer)
                        if is_structured and 'answer' in parsed_data:
                            clean_answer = parsed_data['answer']
                        else:
                            # If parsing fails, use the raw answer but remove any JSON-like content
                            clean_answer = self._extract_clean_text_from_response(final_response.answer)
                    except:
                        clean_answer = self._extract_clean_text_from_response(final_response.answer)
                
                self.memory.add_conversation_turn(query, clean_answer)

                return final_response

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error processing query: {e}")
            
            # Respuesta de error
            error_response = ConversationResponse(
                answer=f"Lo siento, ocurrió un error procesando tu consulta: {str(e)}",
                confidence_score=0.0,
                confidence_level="baja",
                confidence_emoji="🔴",
                sources=[],
                reasoning_trace=[f"Error: {str(e)}"],
                memory_context="",
                execution_time=time.time() - start_time,
                metadata={'error': str(e), 'application': self.application_info}
            )
            
            return error_response
    
    def _generate_response_with_confidence_multi_app(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        memory_context: MemoryContext,
        reasoning_result: ReasoningResult
    ) -> Dict[str, Any]:
        """Generate response with application-specific system prompt and enhanced context."""
        
        # Get enhanced context for Haiku 3 caching optimization
        enhanced_context_data = None
        if self.context_enhancer:
            try:
                enhanced_context_data = self.context_enhancer.get_enhanced_context(query)
                print(f"📋 Enhanced context: {enhanced_context_data['context_length']} chars (Cache optimized: {enhanced_context_data['cache_optimized']})")
            except Exception as e:
                print(f"⚠️  Warning: Could not get enhanced context: {e}")
        
        # Preparar contexto expandido para el LLM
        expanded_results = []
        for result in search_results:
            expanded_result = {
                'text': result.get('text', result.get('content', '')),
                'source': result.get('source_file', result.get('title', result.get('source', 'Fuente desconocida'))),
                'score': result.get('score', 0.0),
                'metadata': result.get('metadata', {})
            }
            expanded_results.append(expanded_result)

        # Build enhanced system prompt with document context
        # CRITICAL: Preserve JSON format instructions and add context as structured data
        enhanced_system_prompt = self.system_prompt
        if enhanced_context_data and enhanced_context_data.get('enhanced_context'):
            # Insert enhanced context BEFORE the JSON format instructions
            # Structure the additional context as JSON-compatible information
            json_format_marker = "<FORMATO DE RESPUESTA ESTRUCTURADA>"
            if json_format_marker in self.system_prompt:
                parts = self.system_prompt.split(json_format_marker)
                enhanced_system_prompt = f"""{parts[0]}

<CONTEXTO DOCUMENTAL ADICIONAL>
Tienes acceso a un inventario completo de documentos y resúmenes adicionales:

{enhanced_context_data['enhanced_context']}

IMPORTANTE: Utiliza este inventario documental para proporcionar respuestas más completas y precisas. Referencia documentos específicos cuando sea relevante. Esta información adicional debe integrarse naturalmente en tu respuesta JSON.
</CONTEXTO DOCUMENTAL ADICIONAL>

{json_format_marker}{parts[1]}"""
            else:
                # Fallback if marker not found - append context but preserve original prompt
                enhanced_system_prompt = f"""{self.system_prompt}

<CONTEXTO DOCUMENTAL ADICIONAL>
Tienes acceso a un inventario completo de documentos y resúmenes adicionales:

{enhanced_context_data['enhanced_context']}

IMPORTANTE: Utiliza este inventario documental para proporcionar respuestas más completas y precisas. Referencia documentos específicos cuando sea relevante. Esta información adicional debe integrarse naturalmente en tu respuesta JSON.
</CONTEXTO DOCUMENTAL ADICIONAL>"""

        # Temporarily override the LLM client's system prompt method
        original_get_system_prompt = self.llm_client._get_system_prompt
        self.llm_client._get_system_prompt = lambda custom_prompt=None: enhanced_system_prompt

        try:
            # Usar el LLM client con prompt personalizado y contexto mejorado
            llm_response = self.llm_client.generate_with_citations(
                query=query,
                expanded_results=expanded_results
            )
        finally:
            # Restore original system prompt method
            self.llm_client._get_system_prompt = original_get_system_prompt

        # The LLM client has already parsed the response - use the structured data directly
        parsed_data = llm_response.get('structured_response')
        is_structured = llm_response.get('is_structured', False)
        
        # DEBUG: Log parsing results
        print(f"🔍 DEBUG - LLM Client Structured Data:")
        print(f"   - Is Structured: {is_structured}")
        print(f"   - LLM Response Keys: {list(llm_response.keys())}")
        if is_structured and parsed_data and 'sources' in parsed_data:
            sources_count = len(parsed_data['sources'])
            print(f"   - JSON Sources Found: {sources_count}")
            if sources_count > 0:
                first_source = parsed_data['sources'][0]
                print(f"   - First Source Relevance: {first_source.get('relevance_score', 'N/A')}")
        else:
            print(f"   - No JSON sources found in structured data")
        
        # Extract confidence information
        if is_structured and parsed_data and 'confidence' in parsed_data:
            # Use structured confidence data
            confidence_info = parsed_data['confidence']
            confidence_score = confidence_info.get('score', 0.7)
            confidence_level = confidence_info.get('level', 'media')
            confidence_rationale = confidence_info.get('rationale', '')
        else:
            # Fallback to legacy confidence extraction
            confidence_score = llm_response.get('confidence_score')
            confidence_rationale = llm_response.get('confidence_rationale', '')
            
            if confidence_score is None:
                confidence_score = 0.7  # Default value when None
            
            if confidence_score >= 0.8:
                confidence_level = "alta"
            elif confidence_score >= 0.6:
                confidence_level = "media"
            else:
                confidence_level = "baja"
        
        # Determine emoji based on confidence level
        if confidence_level in ['alta', 'high', 'very_high']:
            confidence_emoji = "🟢"
        elif confidence_level in ['media', 'medium']:
            confidence_emoji = "🟡"
        else:
            confidence_emoji = "🔴"

        # Determine the final answer to return
        if is_structured and parsed_data and 'answer' in parsed_data:
            final_answer = parsed_data['answer']
        else:
            final_answer = llm_response['answer']

        return {
            'answer': final_answer,
            'confidence_score': confidence_score,
            'confidence_level': confidence_level,
            'confidence_emoji': confidence_emoji,
            'confidence_rationale': confidence_rationale,
            'structured_response': parsed_data if is_structured else None,
            'is_structured': is_structured,
            'key_points': parsed_data.get('key_points', []) if is_structured else [],
            'follow_up_questions': parsed_data.get('follow_up_questions', []) if is_structured else [],
            'related_topics': parsed_data.get('related_topics', []) if is_structured else [],
            'warnings': parsed_data.get('warnings', []) if is_structured else []
        }
    
    def _generate_streaming_response_multi_app(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        memory_context: MemoryContext,
        reasoning_result: ReasoningResult,
        processed_sources: List[Dict[str, Any]],
        start_time: float
    ) -> MultiAppStreamingConversationResponse:
        """Genera respuesta con streaming y información de aplicación"""
        
        # Get enhanced context for Haiku 3 caching optimization
        enhanced_context_data = None
        if self.context_enhancer:
            try:
                enhanced_context_data = self.context_enhancer.get_enhanced_context(query)
                print(f"📋 Enhanced context: {enhanced_context_data['context_length']} chars (Cache optimized: {enhanced_context_data['cache_optimized']})")
            except Exception as e:
                print(f"⚠️  Warning: Could not get enhanced context: {e}")
        
        # Preparar contexto expandido para el LLM
        expanded_results = []
        for result in search_results:
            expanded_result = {
                'text': result.get('text', result.get('content', '')),
                'source': result.get('source_file', result.get('title', result.get('source', 'Fuente desconocida'))),
                'score': result.get('score', 0.0),
                'metadata': result.get('metadata', {})
            }
            expanded_results.append(expanded_result)

        # Build enhanced system prompt with document context
        enhanced_system_prompt = self.system_prompt
        if enhanced_context_data and enhanced_context_data.get('enhanced_context'):
            enhanced_system_prompt = f"""{self.system_prompt}

{enhanced_context_data['enhanced_context']}

IMPORTANT: Use the above document inventory and summaries to provide more comprehensive and accurate responses. Reference specific documents when relevant."""

        # Temporarily override the LLM client's system prompt method
        original_get_system_prompt = self.llm_client._get_system_prompt
        self.llm_client._get_system_prompt = lambda custom_prompt=None: enhanced_system_prompt

        # Crear generador de streaming
        def stream_generator():
            full_text = []
            final_confidence_data = {
                'confidence_score': 0.7,
                'confidence_level': 'media',
                'confidence_emoji': '🟡'
            }
            
            try:
                # Usar el método de streaming del LLM client
                streaming_chunks = self.llm_client.generate_with_citations_streaming(
                    query=query,
                    expanded_results=expanded_results
                )
                
                final_yielded = False
                
                for chunk in streaming_chunks:
                    if chunk['type'] == 'chunk':
                        text = chunk['text']
                        full_text.append(text)
                        yield {
                            'type': 'text',
                            'content': text
                        }
                    elif chunk['type'] == 'complete':
                        # Extract metadata from completion chunk
                        metadata = chunk.get('metadata', {})
                        final_text = metadata.get('full_text', ''.join(full_text))
                        
                        # Extract confidence from the complete response
                        if metadata.get('confidence_score') is not None:
                            confidence_score = metadata['confidence_score']
                            if confidence_score >= 0.8:
                                confidence_level = "alta"
                                confidence_emoji = "🟢"
                            elif confidence_score >= 0.6:
                                confidence_level = "media"
                                confidence_emoji = "🟡"
                            else:
                                confidence_level = "baja"
                                confidence_emoji = "🔴"
                            
                            final_confidence_data = {
                                'confidence_score': confidence_score,
                                'confidence_level': confidence_level,
                                'confidence_emoji': confidence_emoji
                            }
                        else:
                            # Fallback to text analysis
                            final_confidence_data = self._extract_confidence_from_text(final_text)
                        
                        yield {
                            'type': 'final',
                            'confidence_score': final_confidence_data['confidence_score'],
                            'confidence_level': final_confidence_data['confidence_level'],
                            'confidence_emoji': final_confidence_data['confidence_emoji'],
                            'full_text': final_text,
                            'application': self.application_info
                        }
                        
                        # Actualizar memoria con respuesta limpia
                        # Parse the final text to extract clean answer if it's structured
                        try:
                            parsed_data, is_structured = self.response_parser.parse_response(final_text)
                            clean_answer = parsed_data.get('answer') if is_structured and 'answer' in parsed_data else final_text
                        except:
                            clean_answer = final_text
                        self.memory.add_conversation_turn(query, clean_answer)
                        final_yielded = True
                        break
                
                # If no complete chunk was received, handle the accumulated text
                if not final_yielded:
                    final_text = ''.join(full_text)
                    if final_text:
                        final_confidence_data = self._extract_confidence_from_text(final_text)
                        
                        yield {
                            'type': 'final',
                            'confidence_score': final_confidence_data['confidence_score'],
                            'confidence_level': final_confidence_data['confidence_level'],
                            'confidence_emoji': final_confidence_data['confidence_emoji'],
                            'full_text': final_text,
                            'application': self.application_info
                        }
                        
                        # Actualizar memoria con respuesta limpia
                        try:
                            parsed_data, is_structured = self.response_parser.parse_response(final_text)
                            clean_answer = parsed_data.get('answer') if is_structured and 'answer' in parsed_data else final_text
                        except:
                            clean_answer = final_text
                        self.memory.add_conversation_turn(query, clean_answer)
                        
            except Exception as e:
                yield {
                    'type': 'error',
                    'error': str(e)
                }
            finally:
                # Restore original system prompt method
                self.llm_client._get_system_prompt = original_get_system_prompt

        # Crear respuesta de streaming
        streaming_response = MultiAppStreamingConversationResponse(
            stream_generator=stream_generator(),
            confidence_score=0.0,  # Se actualizará durante el streaming
            confidence_level="media",  # Se actualizará durante el streaming
            confidence_emoji="🟡",  # Se actualizará durante el streaming
            sources=processed_sources,
            reasoning_trace=reasoning_result.reasoning_trace,
            memory_context=memory_context.conversation_summary,
            execution_time=time.time() - start_time,
            metadata={
                'search_strategy': reasoning_result.search_strategy,
                'tools_used': reasoning_result.tools_to_use,
                'results_found': len(search_results),
                'memory_confidence': memory_context.memory_confidence,
                'reasoning_confidence': reasoning_result.confidence,
                'streaming': True,
                'application': self.application_info
            },
            application_info=self.application_info
        )

        return streaming_response
    
    def _generate_conversational_response(
        self,
        query: str,
        memory_context: MemoryContext,
        reasoning_result: ReasoningResult,
        start_time: float,
        stream: bool = False
    ) -> Union[ConversationResponse, MultiAppStreamingConversationResponse]:
        """
        Genera respuesta conversacional sin búsqueda de documentos.
        
        Args:
            query: Consulta del usuario
            memory_context: Contexto de memoria
            reasoning_result: Resultado del razonamiento
            start_time: Tiempo de inicio
            stream: Si usar streaming
            
        Returns:
            Respuesta conversacional
        """
        # Generar respuesta conversacional usando el LLM sin contexto de documentos
        conversational_prompt = f"""Eres un asistente especializado en {self.application_info['name']}.
        
Descripción del sistema: {self.application_info['description']}

El usuario te ha saludado o hecho una pregunta conversacional: "{query}"

Responde de manera amigable y profesional. Si es un saludo, preséntate brevemente y explica cómo puedes ayudar con información sobre {self.application_info['name']}.

No busques en documentos para esta respuesta, simplemente sé conversacional y útil."""

        try:
            # Usar el LLM client para generar respuesta conversacional
            llm_response = self.llm_client.generate(
                query=query,
                context="",
                system_prompt=conversational_prompt,
                max_tokens=300,
                temperature=0.7
            )
            
            answer = llm_response.get('answer', 'Hola! Soy tu asistente especializado. ¿En qué puedo ayudarte?')
            
            # Crear respuesta final
            final_response = ConversationResponse(
                answer=answer,
                confidence_score=0.95,  # Alta confianza para respuestas conversacionales
                confidence_level="alta",
                confidence_emoji="🟢",
                sources=[],  # No hay fuentes para respuestas conversacionales
                reasoning_trace=[reasoning_result.reasoning_trace],
                memory_context=memory_context.conversation_summary,
                execution_time=time.time() - start_time,
                metadata={
                    'search_strategy': 'conversational',
                    'tools_used': [],
                    'results_found': 0,
                    'memory_confidence': memory_context.memory_confidence,
                    'reasoning_confidence': reasoning_result.confidence,
                    'application': self.application_info,
                    'conversational': True
                }
            )
            
            # Actualizar memoria
            self.memory.add_conversation_turn(query, final_response.answer)
            
            return final_response
            
        except Exception as e:
            # Fallback a respuesta simple
            fallback_answer = f"¡Hola! Soy tu asistente especializado en {self.application_info['name']}. ¿En qué puedo ayudarte hoy?"
            
            final_response = ConversationResponse(
                answer=fallback_answer,
                confidence_score=0.9,
                confidence_level="alta",
                confidence_emoji="🟢",
                sources=[],
                reasoning_trace=[f"Conversational fallback: {str(e)}"],
                memory_context=memory_context.conversation_summary,
                execution_time=time.time() - start_time,
                metadata={
                    'search_strategy': 'conversational',
                    'tools_used': [],
                    'results_found': 0,
                    'application': self.application_info,
                    'conversational': True,
                    'fallback': True
                }
            )
            
            self.memory.add_conversation_turn(query, final_response.answer)
            return final_response

    def _extract_confidence_from_text(self, text: str) -> Dict[str, Any]:
        """Extrae información de confianza del texto generado"""
        import re
        
        # PASO 1: Buscar patrones explícitos de confianza (CONFIDENCE: XX%)
        explicit_patterns = [
            r'CONFIDENCE:\s*(\d+)%',
            r'confidence:\s*(\d+)%',
            r'Confianza:\s*(\d+)%',
            r'confianza:\s*(\d+)%',
            r'(\d+)%\s*confidence',
            r'(\d+)%\s*confianza'
        ]
        
        confidence_score = None
        for pattern in explicit_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                confidence_score = float(match.group(1)) / 100.0
                break
        
        # PASO 2: Si no se encuentra patrón explícito, usar análisis heurístico
        if confidence_score is None:
            # Buscar indicadores de alta confianza
            high_confidence_indicators = [
                "según el documento", "claramente establece", "específicamente menciona",
                "detalladamente describe", "explícitamente indica", "documentación específica",
                "evidencias documentales", "información específica", "datos concretos"
            ]
            
            # Buscar indicadores de baja confianza
            low_confidence_indicators = [
                "no tengo información", "no encuentro", "no está claro",
                "podría ser", "posiblemente", "no estoy seguro", "no se menciona",
                "no existe información específica", "conocimiento general", "no se encuentra"
            ]
            
            text_lower = text.lower()
            
            high_count = sum(1 for indicator in high_confidence_indicators if indicator in text_lower)
            low_count = sum(1 for indicator in low_confidence_indicators if indicator in text_lower)
            
            if low_count > 0:
                confidence_score = 0.4
            elif high_count > 0:
                confidence_score = 0.9
            else:
                confidence_score = 0.7
        
        # PASO 3: Determinar nivel y emoji basado en el score
        # Ensure confidence_score is not None
        if confidence_score is None:
            confidence_score = 0.7  # Default value when None
            
        if confidence_score >= 0.8:
            confidence_level = "alta"
            confidence_emoji = "🟢"
        elif confidence_score >= 0.6:
            confidence_level = "media"
            confidence_emoji = "🟡"
        else:
            confidence_level = "baja"
            confidence_emoji = "🔴"
        
        return {
            'confidence_score': confidence_score,
            'confidence_level': confidence_level,
            'confidence_emoji': confidence_emoji
        }
    
    def _extract_clean_text_from_response(self, raw_response: str) -> str:
        """
        Extract clean text from a response that might contain JSON or other formatting.
        This method removes JSON structures and extracts only the readable answer content.
        
        Args:
            raw_response: The raw response text that might contain JSON or formatting
            
        Returns:
            Clean text suitable for memory storage and display
        """
        if not raw_response or not raw_response.strip():
            return ""
        
        # Try to parse as JSON first
        try:
            import json
            import re
            
            # Remove any text before the first {
            json_start = raw_response.find('{')
            if json_start != -1:
                json_part = raw_response[json_start:]
                
                # Try to parse the JSON
                parsed = json.loads(json_part)
                
                # Extract the answer field if it exists
                if isinstance(parsed, dict) and 'answer' in parsed:
                    return parsed['answer'].strip()
                elif isinstance(parsed, dict) and 'response' in parsed:
                    return parsed['response'].strip()
                elif isinstance(parsed, dict) and 'content' in parsed:
                    return parsed['content'].strip()
        except (json.JSONDecodeError, ValueError):
            pass
        
        # If JSON parsing fails, try to extract text before any JSON-like structure
        import re
        
        # Remove confidence patterns
        confidence_patterns = [
            r'CONFIDENCE:\s*\d+%',
            r'confidence:\s*\d+%',
            r'Confianza:\s*\d+%',
            r'confianza:\s*\d+%'
        ]
        
        clean_text = raw_response
        for pattern in confidence_patterns:
            clean_text = re.sub(pattern, '', clean_text, flags=re.IGNORECASE)
        
        # Remove JSON-like structures (anything between { and })
        clean_text = re.sub(r'\{[^{}]*\}', '', clean_text)
        
        # Remove multiple whitespaces and newlines
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        # Remove leading/trailing whitespace
        clean_text = clean_text.strip()
        
        # If we still have content, return it
        if clean_text:
            return clean_text
        
        # Last resort: return the original response
        return raw_response.strip()
    
    def _display_comprehensive_response(self, response):
        """Display comprehensive structured response information"""
        # === ASSISTANT RESPONSE SECTION ===
        print(f"\n{'='*60}")
        print(f"🤖  ASSISTANT RESPONSE")
        print(f"{'='*60}")
        
        # Check if we have structured response data
        structured_data = response.metadata.get('structured_response') if hasattr(response, 'metadata') else None
        
        if structured_data and 'answer' in structured_data:
            # Display the clean answer from structured JSON
            clean_answer = structured_data['answer']
            print(f"{clean_answer}")
        else:
            # Fallback to original answer for non-structured responses
            print(f"{response.answer}")
        
        # === CONFIDENCE SECTION ===
        print(f"\n{'='*60}")
        print(f"📊 CONFIDENCE ASSESSMENT")
        print(f"{'='*60}")
        print(f"{response.confidence_emoji} Score: {response.confidence_score*100:.1f}% ({response.confidence_level})")
        
        # Show confidence rationale if available
        if hasattr(response, 'metadata') and 'confidence_rationale' in response.metadata:
            rationale = response.metadata['confidence_rationale']
            if rationale and rationale.strip():
                print(f"💭 Rationale: {rationale}")
        
        # Show confidence factors if available in structured response
        if hasattr(response, 'metadata') and response.metadata.get('structured_response'):
            structured_data = response.metadata['structured_response']
            if 'confidence' in structured_data and 'factors' in structured_data['confidence']:
                factors = structured_data['confidence']['factors']
                print(f"\n🔍 Confidence Factors:")
                for factor_name, factor_data in factors.items():
                    score = factor_data.get('score', 0)
                    explanation = factor_data.get('explanation', '')
                    print(f"   • {factor_name.replace('_', ' ').title()}: {score}/30 - {explanation}")
        
        # === SOURCES SECTION ===
        # Priority: Use sources from structured JSON response if available, otherwise fallback to citation manager sources
        sources_to_display = []
        
        # Try to get sources from structured JSON response first
        if hasattr(response, 'metadata') and response.metadata.get('structured_response'):
            structured_data = response.metadata['structured_response']
            json_sources = structured_data.get('sources', [])
            if json_sources:
                sources_to_display = json_sources  # Show all sources returned by LLM
        
        # Fallback to citation manager sources if no JSON sources
        if not sources_to_display and response.sources:
            sources_to_display = response.sources  # Show all fallback sources too
        
        if sources_to_display:
            displayed_count = len(sources_to_display)
            
            print(f"\n{'='*60}")
            print(f"📚 USED SOURCES ({displayed_count} documents)")
            print(f"{'='*60}")
            for source in sources_to_display:
                # Handle both JSON sources and citation manager sources
                title = source.get('title', source.get('source', 'Documento sin título'))
                relevance = source.get('relevance_score', source.get('score', 0.0))
                excerpt = source.get('excerpt', source.get('text', ''))
                
                # Use the actual ID from the JSON response instead of sequential numbering
                source_id = source.get('id', '[?]')  # Get the actual ID like [2], [6], etc.
                
                print(f"{source_id} {title}")
                print(f"    Relevance: {relevance:.2f}")
                if excerpt and len(excerpt) > 0:
                    # Truncate excerpt if too long
                    excerpt_display = excerpt[:150] + "..." if len(excerpt) > 150 else excerpt
                    print(f"    Excerpt: {excerpt_display}")
                print()
        
        # === STRUCTURED INFORMATION SECTION ===
        if hasattr(response, 'metadata'):
            # Key Points
            key_points = response.metadata.get('key_points', [])
            if key_points:
                print(f"{'='*60}")
                print(f"🔑 KEY POINTS")
                print(f"{'='*60}")
                for i, point in enumerate(key_points, 1):
                    print(f"{i}. {point}")
                print()
            
            # Follow-up Questions
            follow_ups = response.metadata.get('follow_up_questions', [])
            if follow_ups:
                print(f"{'='*60}")
                print(f"❓ FOLLOW-UP QUESTIONS")
                print(f"{'='*60}")
                for i, question in enumerate(follow_ups, 1):
                    print(f"{i}. {question}")
                print()
            
            # Related Topics
            related_topics = response.metadata.get('related_topics', [])
            if related_topics:
                print(f"{'='*60}")
                print(f"🔗 RELATED TOPICS")
                print(f"{'='*60}")
                for i, topic in enumerate(related_topics, 1):
                    print(f"{i}. {topic}")
                print()
            
            # Warnings
            warnings = response.metadata.get('warnings', [])
            if warnings:
                print(f"{'='*60}")
                print(f"⚠️  WARNINGS")
                print(f"{'='*60}")
                for warning in warnings:
                    print(f"• {warning}")
                print()
        
        # === EXECUTION METADATA SECTION ===
        print(f"{'='*60}")
        print(f"⚙️  EXECUTION METADATA")
        print(f"{'='*60}")
        
        # Basic execution info - KEEP
        print(f"⏱️  Execution Time: {response.execution_time:.2f}s")
        print(f"📱 Application: {response.metadata.get('application', {}).get('name', 'Unknown')}")
        
        # Model info - POPULATE with configured model from YAML
        configured_model = self.config_manager.config.get('bedrock', {}).get('llm_model', 'Unknown')
        print(f"🤖 Model: {configured_model}")
        
        # Search and reasoning info
        tools_used = response.metadata.get('tools_used', [])
        results_found = response.metadata.get('results_found', 0)
        
        print(f"🔧 Tools Used: {', '.join(tools_used) if tools_used else 'None'}")
        print(f"📊 Results Found: {results_found}")
        
        # Memory and reasoning confidence (from RAG system components)
        memory_confidence = response.metadata.get('memory_confidence', 0)
        reasoning_confidence = response.metadata.get('reasoning_confidence', 0)
        
        if memory_confidence:
            print(f"🧠 Memory Confidence: {memory_confidence:.2f}")
        if reasoning_confidence:
            print(f" Reasoning Confidence: {reasoning_confidence:.2f}")
        
        # Cache metrics if available
        if hasattr(response, 'metadata') and response.metadata.get('structured_response'):
            structured_data = response.metadata['structured_response']
            metadata = structured_data.get('metadata', {})
            cache_metrics = metadata.get('cache_metrics')
            
            if cache_metrics:
                print(f"\n💾 Cache Metrics:")
                cache_hit = cache_metrics.get('cache_hit', False)
                cache_tokens = cache_metrics.get('cache_tokens', 0)
                print(f"   Cache Hit: {'Yes' if cache_hit else 'No'}")
                if cache_tokens:
                    print(f"   Cache Tokens: {cache_tokens:,}")
        
        print(f"{'='*60}")


def print_application_info(config_manager: MultiAppConfigManager):
    """Print information about available applications."""
    apps_info = config_manager.list_applications_info()
    
    print("\n📱 Available Applications:")
    print("=" * 60)
    for app in apps_info:
        print(f"🔹 ID: {app['id']}")
        print(f"   Name: {app['name']}")
        print(f"   Description: {app['description']}")
        print(f"   Index: {app['index_name']}")
        print(f"   S3 Bucket: {app['s3_bucket']}")
        print(f"   S3 Prefix: {app['s3_prefix']}")
        print("-" * 40)


def main():
    """Main function for multi-application chat interface."""
    parser = argparse.ArgumentParser(
        description="Multi-Application Conversational RAG Chat Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 multi_app_chat.py --app gadea
  python3 multi_app_chat.py --app pds --stream
  python3 multi_app_chat.py --list-apps
  python3 multi_app_chat.py --app erp_financiero --max-results 10
        """
    )
    
    parser.add_argument(
        '--app', '-a',
        type=str,
        help='Application name to use (e.g., gadea, pds, erp_financiero)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/multi_app_config.yaml',
        help='Path to multi-application configuration file'
    )
    
    parser.add_argument(
        '--stream', '-s',
        action='store_true',
        help='Enable streaming responses'
    )
    
    parser.add_argument(
        '--max-results', '-m',
        type=int,
        default=8,
        help='Maximum number of search results (default: 8)'
    )
    
    parser.add_argument(
        '--list-apps', '-l',
        action='store_true',
        help='List available applications and exit'
    )
    
    parser.add_argument(
        '--validate', '-v',
        type=str,
        help='Validate configuration for specific application'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize configuration manager
        config_manager = MultiAppConfigManager(args.config)
        
        # Handle list applications
        if args.list_apps:
            print_application_info(config_manager)
            return
        
        # Handle validation
        if args.validate:
            validation_result = config_manager.validate_configuration(args.validate)
            print(f"\n🔍 Validation for application '{args.validate}':")
            print("=" * 50)
            print(f"Valid: {'✅' if validation_result['valid'] else '❌'}")
            
            if validation_result['errors']:
                print("\n❌ Errors:")
                for error in validation_result['errors']:
                    print(f"  - {error}")
            
            if validation_result['warnings']:
                print("\n⚠️  Warnings:")
                for warning in validation_result['warnings']:
                    print(f"  - {warning}")
            
            return
        
        # Initialize agent
        agent = MultiAppConversationalAgent(
            app_name=args.app,
            config_path=args.config
        )
        
        print(f"\n💬 Multi-Application RAG Chat Interface")
        print(f"Type 'quit', 'exit', or 'bye' to end the conversation")
        print(f"Type 'help' for available commands")
        print(f"Streaming: {'Enabled' if args.stream else 'Disabled'}")
        print("=" * 60)
        
        while True:
            try:
                # Get user input
                user_input = input(f"\n[{agent.application_info['name']}] You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    print("\n📖 Available commands:")
                    print("  help - Show this help message")
                    print("  quit/exit/bye - End the conversation")
                    print("  switch <app_name> - Switch to different application")
                    print("  info - Show current application information")
                    print("  apps - List available applications")
                    continue
                
                if user_input.lower().startswith('switch '):
                    new_app = user_input[7:].strip()
                    try:
                        agent = MultiAppConversationalAgent(
                            app_name=new_app,
                            config_path=args.config
                        )
                        print(f"✅ Switched to application: {agent.application_info['name']}")
                    except ValueError as e:
                        print(f"❌ Error switching application: {e}")
                    continue
                
                if user_input.lower() == 'info':
                    print(f"\n📱 Current Application Information:")
                    print(f"  Name: {agent.application_info['name']}")
                    print(f"  Description: {agent.application_info['description']}")
                    print(f"  ID: {agent.application_info['app_id']}")
                    continue
                
                if user_input.lower() == 'apps':
                    print_application_info(config_manager)
                    continue
                
                # Process query with enhanced streaming display
                if args.stream:
                    print(f"\n🤔 Analyzing query...", end='', flush=True)
                else:
                    print(f"\n🤔 Processing query...")
                
                start_time = time.time()
                
                response = agent.process_query(
                    query=user_input,
                    stream=args.stream,
                    max_results=args.max_results
                )
                
                if args.stream and hasattr(response, 'get_stream'):
                    # Enhanced streaming response display
                    print(" ✓\n🔍 Searching documents...", end='', flush=True)
                    time.sleep(0.5)  # Brief pause for visual effect
                    print(" ✓\n🤖 Generating response: ", end='', flush=True)
                    
                    full_response_text = ""
                    chunk_count = 0
                    
                    for chunk in response.get_stream():
                        if chunk['type'] == 'text':
                            content = chunk['content']
                            print(content, end='', flush=True)
                            full_response_text += content
                            chunk_count += 1
                            
                            # Add periodic visual feedback for long responses
                            if chunk_count % 50 == 0:  # Every 50 chunks
                                print("", end='', flush=True)  # Ensure output is flushed
                                
                        elif chunk['type'] == 'final':
                            confidence_score = chunk.get('confidence_score', 0.7)
                            confidence_level = chunk.get('confidence_level', 'media')
                            confidence_emoji = chunk.get('confidence_emoji', '🟡')
                            
                            print(f"\n\n✅ Complete!")
                            print(f"{confidence_emoji} Confidence: {confidence_score*100:.0f}% ({confidence_level})")
                            print(f"⏱️  Time: {response.execution_time:.2f}s")
                            print(f"📊 Chunks: {chunk_count}")
                            
                            if response.sources:
                                print(f"📚 Sources: {len(response.sources)} documents")
                                
                            # Show application context
                            app_info = chunk.get('application', {})
                            if app_info:
                                print(f"📱 Context: {app_info.get('name', 'Unknown')}")
                                
                        elif chunk['type'] == 'error':
                            print(f"\n\n❌ Streaming Error: {chunk['error']}")
                            print("Falling back to non-streaming mode...")
                            
                            # Attempt fallback to non-streaming
                            try:
                                fallback_response = agent.process_query(
                                    query=user_input,
                                    stream=False,
                                    max_results=args.max_results
                                )
                                print(f"\n🤖 Assistant: {fallback_response.answer}")
                                print(f"\n{fallback_response.confidence_emoji} Confidence: {fallback_response.confidence_score*100:.0f}% ({fallback_response.confidence_level})")
                                print(f"⏱️  Time: {fallback_response.execution_time:.2f}s")
                                if fallback_response.sources:
                                    print(f"📚 Sources: {len(fallback_response.sources)} documents")
                            except Exception as fallback_error:
                                print(f"❌ Fallback also failed: {fallback_error}")
                            break
                else:
                    # Handle regular response with comprehensive structured display
                    agent._display_comprehensive_response(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue
    
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
