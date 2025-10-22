"""
Fixed ReasoningAgent - Modificado para priorizar hybrid_search para soporte de imágenes

Este módulo contiene una versión modificada del ReasoningAgent que prioriza
el uso de hybrid_search sobre keyword_search para asegurar que las imágenes
sean procesadas correctamente por el sistema multimodal.

Cambios principales:
1. Reemplaza keyword_search con hybrid_search en las decisiones
2. Mantiene metadata_search para consultas específicas de documentos
3. Asegura que las imágenes sean procesadas en todas las búsquedas
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
import re

logger = logging.getLogger(__name__)


@dataclass
class ReasoningResult:
    """Resultado del análisis de razonamiento"""
    intentions: List[str]  # Intenciones detectadas (puede ser múltiple)
    search_strategy: str  # Estrategia de búsqueda a usar
    tools_to_use: List[str]  # Herramientas a utilizar en orden
    extracted_identifiers: List[str]  # Códigos, IDs, números específicos extraídos
    search_queries: List[str]  # Consultas optimizadas para búsqueda
    missing_info: List[str]  # Información faltante que necesita
    detail_level: str  # Nivel de detalle: "low", "medium", "high"
    requires_multi_stage: bool  # Si requiere búsqueda en múltiples etapas
    confidence: float  # Confianza en el análisis (0-1)
    reasoning_trace: str  # Explicación del razonamiento


class FixedReasoningAgent:
    """
    Versión modificada del ReasoningAgent que prioriza hybrid_search
    para asegurar el procesamiento correcto de imágenes.
    """

    # Intenciones posibles
    POSSIBLE_INTENTIONS = [
        "BÚSQUEDA_INFORMACIÓN",
        "COMPARACIÓN",
        "ANÁLISIS_DOCUMENTO",
        "EXPLORACIÓN",
        "RESOLUCIÓN_PROBLEMA",
        "VERIFICACIÓN",
        "RESUMEN"
    ]

    # Herramientas disponibles - MODIFICADO: keyword_search reemplazado por hybrid_search
    AVAILABLE_TOOLS = [
        "semantic_search",    # Búsqueda por similitud semántica
        "hybrid_search",      # Combinación de semántica y palabras clave (PRIORIZADO)
        "metadata_search",    # Búsqueda por metadatos (nombre de documento)
        "graph_search",       # Búsqueda por relaciones
    ]

    def __init__(self, llm_client, acronym_dict_path: Optional[str] = None):
        """
        Inicializa el agente de razonamiento.

        Args:
            llm_client: Cliente LLM para generación
            acronym_dict_path: Ruta al diccionario de acrónimos (opcional)
        """
        self.llm_client = llm_client
        self.acronym_dict = self._load_acronym_dictionary(acronym_dict_path)
        logger.debug(f"FixedReasoningAgent initialized with {len(self.acronym_dict)} acronyms")

    def _load_acronym_dictionary(self, path: Optional[str]) -> Dict[str, str]:
        """Carga el diccionario de acrónimos"""
        if not path:
            return {}
        
        try:
            import json
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load acronym dictionary from {path}: {e}")
            return {}

    def analyze_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> ReasoningResult:
        """
        Analiza una consulta y determina la estrategia de búsqueda óptima.
        MODIFICADO: Prioriza hybrid_search para soporte de imágenes.

        Args:
            query: Consulta del usuario
            conversation_history: Historial conversacional (opcional)

        Returns:
            ReasoningResult con la estrategia determinada
        """
        try:
            # Construir prompt para análisis
            analysis_prompt = self._build_analysis_prompt(query, conversation_history)
            
            # Obtener análisis del LLM
            response = self.llm_client.generate(
                query=analysis_prompt,
                context="",
                system_prompt=self._get_system_prompt(),
                max_tokens=800,
                temperature=0.1
            )

            # Parsear respuesta JSON
            result = self._parse_llm_response(response['answer'])
            
            # MODIFICACIÓN CRÍTICA: Reemplazar keyword_search con hybrid_search
            result = self._fix_tool_selection(result)
            
            logger.info(f"Query analysis complete: {result.search_strategy} with tools {result.tools_to_use}")
            return result

        except Exception as e:
            logger.error(f"Error in query analysis: {e}")
            return self._get_default_strategy(query)

    def _fix_tool_selection(self, result: ReasoningResult) -> ReasoningResult:
        """
        MODIFICACIÓN CRÍTICA: Reemplaza keyword_search con hybrid_search
        para asegurar el procesamiento de imágenes.
        """
        # Reemplazar keyword_search con hybrid_search
        fixed_tools = []
        for tool in result.tools_to_use:
            if tool == "keyword_search":
                fixed_tools.append("hybrid_search")
                logger.info("Replaced keyword_search with hybrid_search for image support")
            else:
                fixed_tools.append(tool)
        
        # Crear nuevo resultado con herramientas corregidas
        return ReasoningResult(
            intentions=result.intentions,
            search_strategy=result.search_strategy,
            tools_to_use=fixed_tools,
            extracted_identifiers=result.extracted_identifiers,
            search_queries=result.search_queries,
            missing_info=result.missing_info,
            detail_level=result.detail_level,
            requires_multi_stage=result.requires_multi_stage,
            confidence=result.confidence,
            reasoning_trace=result.reasoning_trace + " [FIXED: keyword_search → hybrid_search for image support]"
        )

    def _build_analysis_prompt(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Construye el prompt para análisis de consulta"""

        prompt_parts = [
            f"CONSULTA DEL USUARIO: {query}",
            "",
            "HERRAMIENTAS DISPONIBLES:",
            "- semantic_search: Búsqueda por similitud semántica",
            "- hybrid_search: Combinación de semántica y palabras clave (RECOMENDADO para soporte multimodal)",
            "- metadata_search: Búsqueda por metadatos (nombre de documento)",
            "- graph_search: Búsqueda por relaciones",
            "",
            "CONTEXTO CONVERSACIONAL:"
        ]

        if conversation_history:
            for i, turn in enumerate(conversation_history[-3:], 1):  # Últimos 3 turnos
                prompt_parts.append(f"{i}. Usuario: {turn.get('user', '')}")
                prompt_parts.append(f"   Asistente: {turn.get('assistant', '')[:100]}...")
        else:
            prompt_parts.append("(Sin historial conversacional)")

        prompt_parts.extend([
            "",
            "Analiza y responde en formato JSON con la siguiente estructura:",
            """{
    "intentions": ["intención1", "intención2"],
    "search_strategy": "single_stage|multi_stage|comparative|exploratory",
    "tools_to_use": ["tool1", "tool2"],
    "extracted_identifiers": ["id1", "id2"],
    "search_queries": ["query1", "query2"],
    "missing_info": ["info1", "info2"],
    "detail_level": "low|medium|high",
    "requires_multi_stage": true|false,
    "confidence": 0.0-1.0,
    "reasoning_trace": "Explicación breve del razonamiento"
}""",
            "",
            "CRITERIOS DE DECISIÓN MODIFICADOS:",
            "0. **PRIORIDAD ALTA: Si es una consulta factual directa (ej: 'cuáles son los X procedimientos', 'qué es X', 'cómo funciona X') → usar single_stage con hybrid_search**",
            "1. Si la consulta menciona términos técnicos específicos → usar hybrid_search (MODIFICADO)",
            "2. Si busca conceptos o ideas → usar semantic_search",
            "3. Si compara múltiples cosas EXPLÍCITAMENTE (ej: 'diferencias entre A y B', 'ventajas de A vs B') → usar multi_stage con múltiples herramientas",
            "4. Si continúa conversación previa → considerar contexto del historial",
            "5. Si es exploratoria o amplia → usar hybrid_search",
            "6. **PRIORIDAD MÁXIMA: Si menciona un documento específico por nombre → usar metadata_search**",
            "",
            "DETECCIÓN DE CONSULTAS SOBRE DOCUMENTOS ESPECÍFICOS (PRIORIDAD MÁXIMA):",
            "- Patrones que indican consulta sobre documento específico:",
            '  * "resume/resumen/resumir el documento X"',
            '  * "qué contiene/dice el documento X"',
            '  * "información del documento X"',
            '  * "busca en el documento X"',
            '  * "contenido del documento X"',
            '  * Cualquier mención de nombres de documentos específicos del sistema',
            "",
            "- Cuando detectes estos patrones:",
            '  1. Extraer el nombre del documento completo en "extracted_identifiers"',
            '  2. Usar "metadata_search" como ÚNICA herramienta (no combinar con otras)',
            '  3. Crear search_queries con el nombre exacto del documento',
            '  4. Usar "single_stage" como estrategia',
            '  5. IMPORTANTE: La herramienta se llama "metadata_search"',
            "",
            "EXTRACCIÓN DE IDENTIFICADORES:",
            '- Si la consulta contiene códigos numéricos (ej: "código 01823994", "ID 12345") → extraerlos en "extracted_identifiers"',
            "- Si hay identificadores específicos → crear consultas de búsqueda optimizadas que incluyan SOLO el identificador",
            "- Para búsquedas de códigos/IDs → usar hybrid_search con el identificador extraído (MODIFICADO)",
            '- Ejemplo: "cuál es la entidad con código 01823994" → extracted_identifiers: ["01823994"], search_queries: ["01823994"]',
            "",
            "NOTA CRÍTICA: Se ha eliminado keyword_search del sistema. Usa hybrid_search para búsquedas que requieran palabras clave exactas.",
            "",
            "Responde SOLO con el JSON, sin texto adicional."
        ])

        return "\n".join(prompt_parts)

    def _get_system_prompt(self) -> str:
        """Retorna el system prompt para el agente de razonamiento"""
        return """Eres un agente de razonamiento experto en análisis de consultas y estrategias de búsqueda.
Tu tarea es analizar consultas de usuarios y determinar la mejor estrategia para recuperar información relevante.

IMPORTANTE: El sistema ha sido modificado para soportar contenido multimodal (texto e imágenes).
Por esta razón, se debe priorizar hybrid_search sobre keyword_search para asegurar que las imágenes
sean procesadas correctamente.

Debes ser:
- Preciso en la identificación de intenciones
- Estratégico en la selección de herramientas
- Consciente del contexto conversacional
- Eficiente en la extracción de identificadores
- Priorizar hybrid_search para soporte multimodal

Responde siempre en formato JSON válido."""

    def _parse_llm_response(self, response: str) -> ReasoningResult:
        """Parsea la respuesta JSON del LLM"""

        try:
            # Limpiar respuesta
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()

            # Parsear JSON
            data = json.loads(response)

            return ReasoningResult(
                intentions=data.get('intentions', ['BÚSQUEDA_INFORMACIÓN']),
                search_strategy=data.get('search_strategy', 'single_stage'),
                tools_to_use=data.get('tools_to_use', ['hybrid_search']),  # MODIFICADO: default a hybrid_search
                extracted_identifiers=data.get('extracted_identifiers', []),
                search_queries=data.get('search_queries', []),
                missing_info=data.get('missing_info', []),
                detail_level=data.get('detail_level', 'medium'),
                requires_multi_stage=data.get('requires_multi_stage', False),
                confidence=data.get('confidence', 0.7),
                reasoning_trace=data.get('reasoning_trace', 'Análisis automático')
            )

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {e}")
            logger.debug(f"Response was: {response}")
            # Fallback
            return ReasoningResult(
                intentions=['BÚSQUEDA_INFORMACIÓN'],
                search_strategy='single_stage',
                tools_to_use=['hybrid_search'],  # MODIFICADO: default a hybrid_search
                extracted_identifiers=[],
                search_queries=[],
                missing_info=[],
                detail_level='medium',
                requires_multi_stage=False,
                confidence=0.5,
                reasoning_trace='Fallback debido a error de parsing'
            )

    def _get_default_strategy(self, query: str) -> ReasoningResult:
        """
        Retorna una estrategia por defecto en caso de error.
        MODIFICADO: Default a hybrid_search para soporte de imágenes.

        Args:
            query: Consulta del usuario

        Returns:
            ReasoningResult con estrategia por defecto
        """
        logger.warning("Using default reasoning strategy")

        return ReasoningResult(
            intentions=['BÚSQUEDA_INFORMACIÓN'],
            search_strategy='single_stage',
            tools_to_use=['hybrid_search'],  # MODIFICADO: default a hybrid_search
            extracted_identifiers=[],
            search_queries=[],
            missing_info=[],
            detail_level='medium',
            requires_multi_stage=False,
            confidence=0.5,
            reasoning_trace='Estrategia por defecto debido a error en análisis [FIXED: using hybrid_search for image support]'
        )

    def should_ask_clarification(self, reasoning: ReasoningResult) -> bool:
        """
        Determina si se debe pedir clarificación al usuario.

        Args:
            reasoning: Resultado del análisis

        Returns:
            bool: True si se debe pedir clarificación
        """
        # NO pedir clarificación para consultas exploratorias
        if "EXPLORACIÓN" in reasoning.intentions:
            logger.info("Skipping clarification for exploratory query")
            return False

        # NO pedir clarificación si hay herramientas disponibles para usar
        if reasoning.tools_to_use and len(reasoning.tools_to_use) > 0:
            logger.info("Skipping clarification - tools available to use")
            return False

        # Pedir clarificación solo si:
        # 1. Confianza MUY baja (< 0.3)
        # 2. Falta información crítica Y no es exploratoria
        # 3. No hay herramientas identificadas

        if reasoning.confidence < 0.3:
            logger.info(f"Requesting clarification due to low confidence: {reasoning.confidence}")
            return True

        if len(reasoning.missing_info) > 3:
            logger.info(f"Requesting clarification due to missing info: {reasoning.missing_info}")
            return True

        return False

    def generate_clarification_question(
        self,
        reasoning: ReasoningResult,
        original_query: str
    ) -> str:
        """
        Genera una pregunta de clarificación basada en el análisis.

        Args:
            reasoning: Resultado del análisis
            original_query: Consulta original

        Returns:
            str: Pregunta de clarificación
        """
        if reasoning.missing_info:
            missing_items = ", ".join(reasoning.missing_info[:2])  # Primeros 2 elementos
            return f"Para ayudarte mejor, necesito más información sobre: {missing_items}. ¿Podrías proporcionar más detalles?"

        if reasoning.confidence < 0.3:
            return f"No estoy seguro de haber entendido completamente tu consulta '{original_query}'. ¿Podrías reformularla o proporcionar más contexto?"

        return "¿Podrías proporcionar más detalles sobre lo que estás buscando?"

    def get_reasoning_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del agente de razonamiento"""
        return {
            "available_tools": self.AVAILABLE_TOOLS,
            "possible_intentions": self.POSSIBLE_INTENTIONS,
            "acronym_dict_size": len(self.acronym_dict),
            "modifications": [
                "keyword_search replaced with hybrid_search",
                "Default fallback uses hybrid_search",
                "Prioritizes multimodal content support"
            ]
        }
