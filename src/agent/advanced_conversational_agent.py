"""
AdvancedConversationalAgent - Agente conversacional avanzado con confidence scoring

Este módulo implementa un agente conversacional completo que integra:
- ReasoningAgent para análisis de consultas
- ToolOrchestrator para ejecución de búsquedas
- AdvancedMemory para contexto conversacional
- CitationManager para gestión de fuentes
- Confidence scoring completo con indicadores visuales
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time

from .reasoning_agent_fixed import FixedReasoningAgent, ReasoningResult
from .tool_orchestrator import ToolOrchestrator, OrchestrationResult
from .advanced_memory import AdvancedMemory, MemoryContext
from ..generation.llm_client_fixed import LLMClient
from ..generation.citation_manager_fixed import FixedCitationManager as CitationManager
from ..retrieval.specialized_retrievers import RetrieverFactory

logger = logging.getLogger(__name__)


@dataclass
class ConversationResponse:
    """Respuesta completa del agente conversacional"""
    answer: str
    confidence_score: float
    confidence_level: str  # "Alta", "Media", "Baja"
    confidence_emoji: str  # "🟢", "🟡", "🔴"
    sources: List[Dict[str, Any]]
    reasoning_trace: str
    memory_context: str
    execution_time: float
    metadata: Dict[str, Any]


class AdvancedConversationalAgent:
    """
    Agente conversacional avanzado que integra todos los componentes
    para proporcionar respuestas con confidence scoring completo.
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        retriever_factory: RetrieverFactory,
        citation_manager: CitationManager,
        memory_file: Optional[str] = None,
        acronym_dict_path: Optional[str] = None,
        config_path: str = "config/multi_app_config.yaml",
        application: str = "darwin",
        session_id: Optional[str] = None
    ):
        """
        Inicializa el agente conversacional avanzado.
        
        Args:
            llm_client: Cliente LLM para generación
            retriever_factory: Factory para crear retrievers
            citation_manager: Gestor de citas y fuentes
            memory_file: Archivo para persistir memoria
            acronym_dict_path: Ruta al diccionario de acrónimos
            config_path: Ruta al archivo de configuración
            application: Application name for context
            session_id: ID de sesión para aislamiento de conversaciones (opcional)
        """
        self.llm_client = llm_client
        self.citation_manager = citation_manager
        self.session_id = session_id
        self.application = application
        
        # Inicializar componentes
        self.reasoning_agent = FixedReasoningAgent(
            llm_client=llm_client,
            acronym_dict_path=acronym_dict_path
        )
        
        self.tool_orchestrator = ToolOrchestrator(
            retriever_factory=retriever_factory,
            config_path=config_path,
            application=application,
            max_workers=3,
            timeout_seconds=30
        )
        
        self.memory = AdvancedMemory(
            llm_client=llm_client,
            max_short_term_turns=10,
            max_long_term_turns=100,
            memory_file=memory_file,
            session_id=session_id
        )
        
        # Configuración de confidence scoring
        self.confidence_thresholds = {
            'high': 0.8,    # 80% - Alta confianza 🟢
            'medium': 0.6   # 60% - Media confianza 🟡
                           # < 60% - Baja confianza 🔴
        }
        
        logger.info("AdvancedConversationalAgent initialized")
    
    def process_query(
        self,
        query: str,
        max_results: int = 10
    ) -> ConversationResponse:
        """
        Procesa una consulta del usuario y genera una respuesta.

        Args:
            query: Consulta del usuario
            max_results: Máximo número de resultados de búsqueda

        Returns:
            ConversationResponse con la respuesta generada
        """
        start_time = time.time()
        
        try:
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
            
            
            # PASO 5: Generar respuesta con LLM
            response_data = self._generate_response_with_confidence(
                query,
                search_results=orchestration_result.final_results,
                memory_context=memory_context,
                reasoning_result=reasoning_result
            )
            
            # PASO 6: Procesar citas
            processed_sources = self.citation_manager.process_sources(
                orchestration_result.final_results
            )
            
            # PASO 7: Crear respuesta final
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
                    # Pass through structured response data from LLM client
                    'structured_response': response_data.get('structured_response'),
                    'confidence_rationale': response_data.get('confidence_rationale'),
                    'key_points': response_data.get('key_points', []),
                    'follow_up_questions': response_data.get('follow_up_questions', []),
                    'related_topics': response_data.get('related_topics', []),
                    'warnings': response_data.get('warnings', [])
                }
            )
            
            # PASO 8: Guardar en memoria
            self.memory.add_conversation_turn(
                user_message=query,
                assistant_response=response_data['answer'],
                confidence_score=response_data['confidence_score'],
                metadata={
                    'sources_count': len(processed_sources),
                    'search_strategy': reasoning_result.search_strategy
                }
            )
            
            logger.info(f"Query processed in {final_response.execution_time:.2f}s (confidence: {final_response.confidence_score:.2f})")
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return self._create_error_response(str(e), time.time() - start_time)
    
    def _format_memory_for_reasoning(self, memory_context: MemoryContext) -> List[Dict[str, str]]:
        """Formatea el contexto de memoria para el ReasoningAgent"""
        
        conversation_history = []
        
        for turn in memory_context.relevant_history:
            conversation_history.append({
                'user': turn.user_message,
                'assistant': turn.assistant_response[:200] + "..." if len(turn.assistant_response) > 200 else turn.assistant_response
            })
        
        return conversation_history
    
    def _generate_response_with_confidence(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        memory_context: MemoryContext,
        reasoning_result: ReasoningResult
    ) -> Dict[str, Any]:
        """Genera respuesta con confidence scoring usando el LLM"""
        
        # Preparar contexto expandido para el LLM
        expanded_results = []
        for result in search_results:
            expanded_result = {
                'text': result.get('text', result.get('content', '')),
                'source': result.get('source_file', result.get('title', result.get('source', 'Fuente desconocida'))),
                'score': result.get('score', 0.0),
                'metadata': result.get('metadata', {}),
                # FIXED: Pass through title and filename fields from HybridRetrieverFixed
                'title': result.get('title', ''),
                'file_name': result.get('file_name', ''),
                'source_file': result.get('source_file', ''),
                'chunk_id': result.get('chunk_id', ''),
                'doc_id': result.get('doc_id', '')
            }
            expanded_results.append(expanded_result)

        # IMPORTANT: Do NOT override the system prompt here
        # Let the LLM client use the system prompt configured in the multi-app system
        # This ensures JSON format is maintained when required
        
        # Generar respuesta usando el LLM client (que usará el system prompt configurado)
        try:
            result = self.llm_client.generate_with_citations(
                query=query,
                expanded_results=expanded_results
            )
            
            answer = result['answer']
            confidence_score = result.get('confidence_score', 0.5)  # El LLMClient ya extrae el confidence
            
            # Determinar nivel y emoji de confianza
            # Note: confidence_score from LLMClient is already in 0.0-1.0 range, no need to divide by 100
            confidence_level, confidence_emoji = self._determine_confidence_level(confidence_score)
            
            return {
                'answer': answer,
                'confidence_score': confidence_score,
                'confidence_level': confidence_level,
                'confidence_emoji': confidence_emoji,
                # Pass through all structured response data from LLM client
                'structured_response': result.get('structured_response'),
                'confidence_rationale': result.get('confidence_rationale'),
                'key_points': result.get('key_points', []),
                'follow_up_questions': result.get('follow_up_questions', []),
                'related_topics': result.get('related_topics', []),
                'warnings': result.get('warnings', [])
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'answer': f"Lo siento, hubo un error al generar la respuesta: {str(e)}",
                'confidence_score': 0.0,
                'confidence_level': "Baja",
                'confidence_emoji': "🔴"
            }
    
    def _determine_confidence_level(self, confidence_score: float) -> Tuple[str, str]:
        """Determina el nivel de confianza y emoji correspondiente"""
        
        if confidence_score >= self.confidence_thresholds['high']:
            return "Alta", "🟢"
        elif confidence_score >= self.confidence_thresholds['medium']:
            return "Media", "🟡"
        else:
            return "Baja", "🔴"
    
    def _create_clarification_response(
        self,
        clarification: str,
        reasoning_result: ReasoningResult,
        execution_time: float
    ) -> ConversationResponse:
        """Crea una respuesta de clarificación"""
        
        return ConversationResponse(
            answer=clarification,
            confidence_score=0.5,  # Neutral para clarificaciones
            confidence_level="Media",
            confidence_emoji="🟡",
            sources=[],
            reasoning_trace=reasoning_result.reasoning_trace,
            memory_context="Solicitando clarificación",
            execution_time=execution_time,
            metadata={
                'type': 'clarification',
                'reasoning_confidence': reasoning_result.confidence
            }
        )
    
    def _create_error_response(self, error_message: str, execution_time: float) -> ConversationResponse:
        """Crea una respuesta de error"""
        
        return ConversationResponse(
            answer=f"Lo siento, ocurrió un error al procesar tu consulta: {error_message}",
            confidence_score=0.0,
            confidence_level="Baja",
            confidence_emoji="🔴",
            sources=[],
            reasoning_trace="Error en procesamiento",
            memory_context="Error",
            execution_time=execution_time,
            metadata={
                'type': 'error',
                'error': error_message
            }
        )
    
    def format_response_for_display(self, response: ConversationResponse) -> str:
        """Formatea la respuesta para mostrar al usuario con confidence scoring"""
        
        # Construir respuesta formateada
        formatted_parts = []
        
        # Respuesta principal con confidence scoring
        confidence_display = f"{response.confidence_emoji} {response.confidence_score:.0f}% ({response.confidence_level} confianza)"
        formatted_parts.append(f"{response.answer}\n\n**Confianza**: {confidence_display}")
        
        # Fuentes si están disponibles
        if response.sources:
            formatted_parts.append("\n**Fuentes consultadas:**")
            for i, source in enumerate(response.sources[:3], 1):  # Mostrar top 3 fuentes
                source_name = source.get('source', 'Fuente desconocida')
                formatted_parts.append(f"{i}. {source_name}")
        
        # Información de contexto (opcional, para debugging)
        if response.metadata.get('search_strategy'):
            strategy = response.metadata['search_strategy']
            tools = ', '.join(response.metadata.get('tools_used', []))
            formatted_parts.append(f"\n*Estrategia: {strategy} | Herramientas: {tools}*")
        
        return "\n".join(formatted_parts)
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del agente"""
        
        memory_stats = self.memory.get_memory_stats()
        orchestrator_stats = self.tool_orchestrator.get_execution_stats()
        
        return {
            'memory': memory_stats,
            'orchestrator': orchestrator_stats,
            'confidence_thresholds': self.confidence_thresholds,
            'components_initialized': {
                'reasoning_agent': self.reasoning_agent is not None,
                'tool_orchestrator': self.tool_orchestrator is not None,
                'memory': self.memory is not None,
                'citation_manager': self.citation_manager is not None
            }
        }
    
    def clear_conversation_memory(self) -> None:
        """Limpia la memoria conversacional"""
        
        self.memory.clear_memory()
    
    def set_confidence_thresholds(self, high: float, medium: float) -> None:
        """Configura los umbrales de confianza"""
        
        if not (0 <= medium <= high <= 1):
            raise ValueError("Thresholds must satisfy: 0 <= medium <= high <= 1")
        
        self.confidence_thresholds['high'] = high
        self.confidence_thresholds['medium'] = medium
