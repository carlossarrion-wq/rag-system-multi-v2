"""
AdvancedMemory - Sistema de memoria avanzado para conversaciones

Este módulo maneja la memoria conversacional con capacidades avanzadas:
- Memoria a corto y largo plazo
- Resumen automático de conversaciones largas
- Extracción de entidades y temas
- Persistencia de memoria entre sesiones
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Un turno de conversación (usuario + asistente)"""
    user_message: str
    assistant_response: str
    timestamp: datetime
    entities: List[str] = None
    topics: List[str] = None
    confidence_score: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.entities is None:
            self.entities = []
        if self.topics is None:
            self.topics = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MemoryContext:
    """Contexto de memoria para una consulta"""
    relevant_history: List[ConversationTurn]
    extracted_entities: List[str]
    current_topics: List[str]
    conversation_summary: str
    memory_confidence: float


class AdvancedMemory:
    """
    Sistema de memoria avanzado que mantiene contexto conversacional
    con capacidades de resumen y extracción de entidades.
    
    MODIFICADO: Ahora soporta sesiones para conversaciones concurrentes.
    """
    
    def __init__(
        self,
        llm_client,
        max_short_term_turns: int = 10,
        max_long_term_turns: int = 100,
        memory_file: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """
        Inicializa el sistema de memoria.
        
        Args:
            llm_client: Cliente LLM para procesamiento de memoria
            max_short_term_turns: Máximo turnos en memoria a corto plazo
            max_long_term_turns: Máximo turnos en memoria a largo plazo
            memory_file: Archivo para persistir memoria (opcional)
            session_id: ID de sesión para aislamiento de conversaciones (opcional)
        """
        self.llm_client = llm_client
        self.max_short_term_turns = max_short_term_turns
        self.max_long_term_turns = max_long_term_turns
        self.session_id = session_id
        
        # Memoria a corto plazo (turnos recientes)
        self.short_term_memory: List[ConversationTurn] = []
        
        # Memoria a largo plazo (resúmenes y turnos importantes)
        self.long_term_memory: List[ConversationTurn] = []
        
        # Resúmenes de conversaciones pasadas
        self.conversation_summaries: List[str] = []
        
        # Entidades extraídas acumuladas
        self.accumulated_entities: Dict[str, int] = {}  # entidad -> frecuencia
        
        # Temas recurrentes
        self.recurring_topics: Dict[str, int] = {}  # tema -> frecuencia
        
        # Archivo de persistencia (modificado para sesiones)
        self.memory_file = self._resolve_memory_file(memory_file, session_id)
        if self.memory_file:
            self._load_memory()
        
        session_info = f" (Session: {session_id})" if session_id else ""
        logger.info(f"AdvancedMemory initialized with {max_short_term_turns} short-term, {max_long_term_turns} long-term turns{session_info}")
    
    def add_conversation_turn(
        self,
        user_message: str,
        assistant_response: str,
        confidence_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Añade un nuevo turno de conversación a la memoria.
        
        Args:
            user_message: Mensaje del usuario
            assistant_response: Respuesta del asistente
            confidence_score: Score de confianza de la respuesta
            metadata: Metadatos adicionales
        """
        # Crear turno de conversación
        turn = ConversationTurn(
            user_message=user_message,
            assistant_response=assistant_response,
            timestamp=datetime.now(),
            confidence_score=confidence_score,
            metadata=metadata or {}
        )
        
        # Extraer entidades y temas del turno
        self._extract_entities_and_topics(turn)
        
        # Añadir a memoria a corto plazo
        self.short_term_memory.append(turn)
        
        # Gestionar límites de memoria
        self._manage_memory_limits()
        
        # Persistir si está configurado
        if self.memory_file:
            self._save_memory()
        
        logger.debug(f"Added conversation turn with {len(turn.entities)} entities, {len(turn.topics)} topics")
    
    def get_relevant_context(
        self,
        current_query: str,
        max_turns: int = 5
    ) -> MemoryContext:
        """
        Obtiene contexto relevante para la consulta actual.
        
        Args:
            current_query: Consulta actual del usuario
            max_turns: Máximo número de turnos a incluir
        
        Returns:
            MemoryContext con información relevante
        """
        # Obtener turnos más relevantes
        relevant_turns = self._find_relevant_turns(current_query, max_turns)
        
        # Extraer entidades de la consulta actual
        current_entities = self._extract_entities_from_text(current_query)
        
        # Obtener temas actuales
        current_topics = self._extract_topics_from_text(current_query)
        
        # Generar resumen de conversación
        conversation_summary = self._generate_conversation_summary(relevant_turns)
        
        # Calcular confianza de la memoria
        memory_confidence = self._calculate_memory_confidence(relevant_turns, current_query)
        
        return MemoryContext(
            relevant_history=relevant_turns,
            extracted_entities=current_entities,
            current_topics=current_topics,
            conversation_summary=conversation_summary,
            memory_confidence=memory_confidence
        )
    
    def _extract_entities_and_topics(self, turn: ConversationTurn) -> None:
        """Extrae entidades y temas de un turno de conversación"""
        
        # Combinar mensaje del usuario y respuesta del asistente
        full_text = f"{turn.user_message} {turn.assistant_response}"
        
        # Extraer entidades
        turn.entities = self._extract_entities_from_text(full_text)
        
        # Extraer temas
        turn.topics = self._extract_topics_from_text(full_text)
        
        # Actualizar contadores acumulados
        for entity in turn.entities:
            self.accumulated_entities[entity] = self.accumulated_entities.get(entity, 0) + 1
        
        for topic in turn.topics:
            self.recurring_topics[topic] = self.recurring_topics.get(topic, 0) + 1
    
    def _extract_entities_from_text(self, text: str) -> List[str]:
        """Extrae entidades del texto usando patrones simples"""
        
        entities = []
        
        # Patrones para documentos del sistema
        import re
        
        # Documentos del sistema
        df_pattern = r'DF_\w+\d+\.\d+[^,\.\s]*'
        entities.extend(re.findall(df_pattern, text, re.IGNORECASE))
        
        # Códigos numéricos
        code_pattern = r'\b\d{6,}\b'
        entities.extend(re.findall(code_pattern, text))
        
        # Nombres de módulos ECOFI
        ecofi_pattern = r'ECOFI[^,\.\s]*'
        entities.extend(re.findall(ecofi_pattern, text, re.IGNORECASE))
        
        # Procesos de negocio
        business_patterns = [
            r'\b(?:compras?|aprovisionamiento|tesorería|contabilidad|activos?)\b',
            r'\b(?:proveedores?|clientes?|facturas?|pagos?)\b',
            r'\b(?:CFIN|SMART|SAP|DARWIN)\b'
        ]
        
        for pattern in business_patterns:
            entities.extend(re.findall(pattern, text, re.IGNORECASE))
        
        # Limpiar y deduplicar
        entities = list(set([e.strip() for e in entities if len(e.strip()) > 2]))
        
        return entities[:10]  # Limitar a 10 entidades más relevantes
    
    def _extract_topics_from_text(self, text: str) -> List[str]:
        """Extrae temas del texto"""
        
        topics = []
        text_lower = text.lower()
        
        # Temas de negocio
        business_topics = {
            'finanzas': ['finanzas', 'financiero', 'contabilidad', 'tesorería', 'pagos', 'cobros'],
            'compras': ['compras', 'aprovisionamiento', 'proveedores', 'sourcing'],
            'almacenes': ['almacenes', 'inventario', 'stock', 'gestión almacenes'],
            'activos': ['activos fijos', 'activos', 'alquileres', 'ifrs16'],
            'impuestos': ['impuestos', 'fiscal', 'tributario'],
            'proyectos': ['proyectos', 'inversiones', 'gestión proyectos'],
            'documentación': ['documentos', 'documentación', 'archivos']
        }
        
        for topic, keywords in business_topics.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _find_relevant_turns(self, query: str, max_turns: int) -> List[ConversationTurn]:
        """Encuentra turnos relevantes para la consulta actual"""
        
        # Extraer entidades y temas de la consulta
        query_entities = set(self._extract_entities_from_text(query))
        query_topics = set(self._extract_topics_from_text(query))
        
        # Calcular relevancia de cada turno
        turn_scores = []
        
        # Combinar memoria a corto y largo plazo
        all_turns = self.short_term_memory + self.long_term_memory
        
        for turn in all_turns:
            score = 0.0
            
            # Score por entidades compartidas
            shared_entities = query_entities.intersection(set(turn.entities))
            score += len(shared_entities) * 2.0
            
            # Score por temas compartidos
            shared_topics = query_topics.intersection(set(turn.topics))
            score += len(shared_topics) * 1.5
            
            # Score por recencia (más reciente = mayor score)
            hours_ago = (datetime.now() - turn.timestamp).total_seconds() / 3600
            recency_score = max(0, 1.0 - (hours_ago / 24))  # Decae en 24 horas
            score += recency_score * 0.5
            
            # Score por confianza de la respuesta
            if turn.confidence_score:
                score += turn.confidence_score * 0.3
            
            # Penalizar si no hay overlap
            if score == 0 and not any(word in turn.user_message.lower() + turn.assistant_response.lower() 
                                    for word in query.lower().split()):
                score = -1.0
            
            turn_scores.append((turn, score))
        
        # Ordenar por score y tomar los mejores
        turn_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Filtrar scores negativos y tomar top max_turns
        relevant_turns = [turn for turn, score in turn_scores if score > 0][:max_turns]
        
        return relevant_turns
    
    def _generate_conversation_summary(self, turns: List[ConversationTurn]) -> str:
        """Genera un resumen de los turnos de conversación"""
        
        if not turns:
            return "No hay historial de conversación relevante."
        
        if len(turns) == 1:
            return f"Conversación previa sobre: {turns[0].user_message[:100]}..."
        
        # Extraer temas principales
        all_topics = []
        all_entities = []
        
        for turn in turns:
            all_topics.extend(turn.topics)
            all_entities.extend(turn.entities)
        
        # Contar frecuencias
        topic_counts = {}
        entity_counts = {}
        
        for topic in all_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        for entity in all_entities:
            entity_counts[entity] = entity_counts.get(entity, 0) + 1
        
        # Obtener los más frecuentes
        top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        summary_parts = []
        
        if top_topics:
            topics_str = ", ".join([topic for topic, _ in top_topics])
            summary_parts.append(f"Temas principales: {topics_str}")
        
        if top_entities:
            entities_str = ", ".join([entity for entity, _ in top_entities])
            summary_parts.append(f"Entidades mencionadas: {entities_str}")
        
        summary_parts.append(f"Basado en {len(turns)} intercambios recientes")
        
        return ". ".join(summary_parts) + "."
    
    def _calculate_memory_confidence(self, turns: List[ConversationTurn], query: str) -> float:
        """Calcula la confianza en la memoria para la consulta actual"""
        
        if not turns:
            return 0.0
        
        # Factores de confianza
        confidence_factors = []
        
        # Factor 1: Número de turnos relevantes
        turn_factor = min(len(turns) / 5.0, 1.0)  # Máximo con 5 turnos
        confidence_factors.append(turn_factor)
        
        # Factor 2: Recencia promedio
        avg_hours_ago = sum((datetime.now() - turn.timestamp).total_seconds() / 3600 for turn in turns) / len(turns)
        recency_factor = max(0, 1.0 - (avg_hours_ago / 48))  # Decae en 48 horas
        confidence_factors.append(recency_factor)
        
        # Factor 3: Confianza promedio de las respuestas
        confidence_scores = [turn.confidence_score for turn in turns if turn.confidence_score is not None]
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            confidence_factors.append(avg_confidence)
        
        # Factor 4: Overlap de entidades/temas
        query_entities = set(self._extract_entities_from_text(query))
        query_topics = set(self._extract_topics_from_text(query))
        
        all_turn_entities = set()
        all_turn_topics = set()
        
        for turn in turns:
            all_turn_entities.update(turn.entities)
            all_turn_topics.update(turn.topics)
        
        entity_overlap = len(query_entities.intersection(all_turn_entities)) / max(len(query_entities), 1)
        topic_overlap = len(query_topics.intersection(all_turn_topics)) / max(len(query_topics), 1)
        
        overlap_factor = (entity_overlap + topic_overlap) / 2
        confidence_factors.append(overlap_factor)
        
        # Calcular confianza final como promedio ponderado
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        else:
            return 0.5  # Confianza neutral por defecto
    
    def _manage_memory_limits(self) -> None:
        """Gestiona los límites de memoria moviendo turnos antiguos a largo plazo"""
        
        # Si la memoria a corto plazo excede el límite
        if len(self.short_term_memory) > self.max_short_term_turns:
            # Mover turnos más antiguos a memoria a largo plazo
            excess_turns = len(self.short_term_memory) - self.max_short_term_turns
            
            # Seleccionar turnos a mover (los más antiguos, pero preservar los importantes)
            turns_to_move = self._select_turns_for_long_term(excess_turns)
            
            # Mover a memoria a largo plazo
            for turn in turns_to_move:
                self.short_term_memory.remove(turn)
                self.long_term_memory.append(turn)
            
            logger.debug(f"Moved {len(turns_to_move)} turns to long-term memory")
        
        # Si la memoria a largo plazo excede el límite
        if len(self.long_term_memory) > self.max_long_term_turns:
            # Generar resumen de los turnos más antiguos
            excess_turns = len(self.long_term_memory) - self.max_long_term_turns
            oldest_turns = sorted(self.long_term_memory, key=lambda x: x.timestamp)[:excess_turns]
            
            # Crear resumen
            if oldest_turns:
                summary = self._create_memory_summary(oldest_turns)
                self.conversation_summaries.append(summary)
                
                # Remover turnos resumidos
                for turn in oldest_turns:
                    self.long_term_memory.remove(turn)
                
                logger.debug(f"Summarized and removed {len(oldest_turns)} old turns")
    
    def _select_turns_for_long_term(self, num_turns: int) -> List[ConversationTurn]:
        """Selecciona turnos para mover a memoria a largo plazo"""
        
        # Ordenar por timestamp (más antiguos primero)
        sorted_turns = sorted(self.short_term_memory, key=lambda x: x.timestamp)
        
        # Seleccionar los más antiguos, pero preservar turnos importantes
        selected_turns = []
        
        for turn in sorted_turns:
            if len(selected_turns) >= num_turns:
                break
            
            # Preservar turnos con alta confianza o muchas entidades
            is_important = (
                (turn.confidence_score and turn.confidence_score > 0.8) or
                len(turn.entities) > 3 or
                len(turn.topics) > 2
            )
            
            if not is_important:
                selected_turns.append(turn)
        
        # Si no hay suficientes turnos no importantes, tomar los más antiguos
        if len(selected_turns) < num_turns:
            remaining_needed = num_turns - len(selected_turns)
            remaining_turns = [t for t in sorted_turns if t not in selected_turns]
            selected_turns.extend(remaining_turns[:remaining_needed])
        
        return selected_turns
    
    def _create_memory_summary(self, turns: List[ConversationTurn]) -> str:
        """Crea un resumen de turnos para archivar"""
        
        if not turns:
            return ""
        
        # Extraer información clave
        all_entities = []
        all_topics = []
        key_questions = []
        
        for turn in turns:
            all_entities.extend(turn.entities)
            all_topics.extend(turn.topics)
            
            # Extraer preguntas clave (primeras palabras del mensaje del usuario)
            question_start = turn.user_message[:50].strip()
            if question_start:
                key_questions.append(question_start)
        
        # Crear resumen estructurado
        summary_parts = []
        
        # Período de tiempo
        start_time = min(turn.timestamp for turn in turns)
        end_time = max(turn.timestamp for turn in turns)
        summary_parts.append(f"Período: {start_time.strftime('%Y-%m-%d %H:%M')} - {end_time.strftime('%Y-%m-%d %H:%M')}")
        
        # Número de intercambios
        summary_parts.append(f"Intercambios: {len(turns)}")
        
        # Entidades principales
        entity_counts = {}
        for entity in all_entities:
            entity_counts[entity] = entity_counts.get(entity, 0) + 1
        
        if entity_counts:
            top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            entities_str = ", ".join([f"{entity} ({count})" for entity, count in top_entities])
            summary_parts.append(f"Entidades: {entities_str}")
        
        # Temas principales
        topic_counts = {}
        for topic in all_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        if topic_counts:
            top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            topics_str = ", ".join([topic for topic, _ in top_topics])
            summary_parts.append(f"Temas: {topics_str}")
        
        # Preguntas clave
        if key_questions:
            questions_sample = key_questions[:3]  # Primeras 3 preguntas
            summary_parts.append(f"Preguntas ejemplo: {'; '.join(questions_sample)}")
        
        return " | ".join(summary_parts)
    
    def _save_memory(self) -> None:
        """Guarda la memoria en archivo"""
        
        if not self.memory_file:
            return
        
        try:
            memory_data = {
                'short_term_memory': [self._turn_to_dict(turn) for turn in self.short_term_memory],
                'long_term_memory': [self._turn_to_dict(turn) for turn in self.long_term_memory],
                'conversation_summaries': self.conversation_summaries,
                'accumulated_entities': self.accumulated_entities,
                'recurring_topics': self.recurring_topics,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Memory saved to {self.memory_file}")
            
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
    
    def _load_memory(self) -> None:
        """Carga la memoria desde archivo"""
        
        if not self.memory_file or not Path(self.memory_file).exists():
            return
        
        try:
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)
            
            # Cargar turnos de memoria
            self.short_term_memory = [self._dict_to_turn(turn_dict) for turn_dict in memory_data.get('short_term_memory', [])]
            self.long_term_memory = [self._dict_to_turn(turn_dict) for turn_dict in memory_data.get('long_term_memory', [])]
            
            # Cargar otros datos
            self.conversation_summaries = memory_data.get('conversation_summaries', [])
            self.accumulated_entities = memory_data.get('accumulated_entities', {})
            self.recurring_topics = memory_data.get('recurring_topics', {})
            
            logger.info(f"Memory loaded from {self.memory_file}: {len(self.short_term_memory)} short-term, {len(self.long_term_memory)} long-term turns")
            
        except Exception as e:
            logger.error(f"Error loading memory: {e}")
    
    def _turn_to_dict(self, turn: ConversationTurn) -> Dict[str, Any]:
        """Convierte un ConversationTurn a diccionario para serialización"""
        
        return {
            'user_message': turn.user_message,
            'assistant_response': turn.assistant_response,
            'timestamp': turn.timestamp.isoformat(),
            'entities': turn.entities,
            'topics': turn.topics,
            'confidence_score': turn.confidence_score,
            'metadata': turn.metadata
        }
    
    def _dict_to_turn(self, turn_dict: Dict[str, Any]) -> ConversationTurn:
        """Convierte un diccionario a ConversationTurn"""
        
        return ConversationTurn(
            user_message=turn_dict['user_message'],
            assistant_response=turn_dict['assistant_response'],
            timestamp=datetime.fromisoformat(turn_dict['timestamp']),
            entities=turn_dict.get('entities', []),
            topics=turn_dict.get('topics', []),
            confidence_score=turn_dict.get('confidence_score'),
            metadata=turn_dict.get('metadata', {})
        )
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de la memoria"""
        
        return {
            'short_term_turns': len(self.short_term_memory),
            'long_term_turns': len(self.long_term_memory),
            'conversation_summaries': len(self.conversation_summaries),
            'total_entities': len(self.accumulated_entities),
            'total_topics': len(self.recurring_topics),
            'top_entities': sorted(self.accumulated_entities.items(), key=lambda x: x[1], reverse=True)[:5],
            'top_topics': sorted(self.recurring_topics.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def clear_memory(self) -> None:
        """Limpia toda la memoria"""
        
        self.short_term_memory.clear()
        self.long_term_memory.clear()
        self.conversation_summaries.clear()
        self.accumulated_entities.clear()
        self.recurring_topics.clear()
        
        if self.memory_file:
            self._save_memory()
        
        logger.info("Memory cleared")
    
    def _resolve_memory_file(self, memory_file: Optional[str], session_id: Optional[str]) -> Optional[str]:
        """
        Resuelve la ruta del archivo de memoria considerando sesiones.
        
        Args:
            memory_file: Archivo de memoria base
            session_id: ID de sesión (opcional)
            
        Returns:
            Ruta del archivo de memoria o None
        """
        if not memory_file:
            return None
        
        # Si hay session_id, usar el gestor de sesiones
        if session_id:
            try:
                from .session_manager import get_session_manager
                session_manager = get_session_manager()
                return session_manager.get_memory_file_path(session_id)
            except ImportError:
                logger.warning("SessionManager not available, using legacy memory file")
                return memory_file
        
        # Sin sesión, usar archivo tradicional
        return memory_file
