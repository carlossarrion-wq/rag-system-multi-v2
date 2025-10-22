"""
ToolOrchestrator - Orquestación inteligente de herramientas de búsqueda

Este módulo coordina el uso de múltiples herramientas de búsqueda
basándose en las decisiones del ReasoningAgent.

Maneja la ejecución secuencial y paralela de herramientas,
fusión de resultados y optimización de consultas.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from ..retrieval.specialized_retrievers import RetrieverFactory
from .reasoning_agent_fixed import ReasoningResult

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Resultado de la ejecución de una herramienta"""
    tool_name: str
    query: str
    results: List[Dict[str, Any]]
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class OrchestrationResult:
    """Resultado completo de la orquestación"""
    final_results: List[Dict[str, Any]]
    tool_results: List[ToolResult]
    total_execution_time: float
    strategy_used: str
    fusion_method: str
    success: bool
    reasoning_trace: str


class ToolOrchestrator:
    """
    Orquestador que coordina múltiples herramientas de búsqueda
    basándose en las decisiones del ReasoningAgent.
    """
    
    def __init__(
        self,
        retriever_factory: RetrieverFactory,
        config_path: str = "config/aws_config_production.yaml",
        max_workers: int = 3,
        timeout_seconds: int = 30
    ):
        """
        Inicializa el orquestador.
        
        Args:
            retriever_factory: Factory para crear retrievers
            max_workers: Máximo número de workers paralelos
            timeout_seconds: Timeout para ejecución de herramientas
        """
        self.retriever_factory = retriever_factory
        self.config_path = config_path
        self.max_workers = max_workers
        self.timeout_seconds = timeout_seconds
        
        # Cache de retrievers para reutilización
        self._retriever_cache = {}
        
        logger.info(f"ToolOrchestrator initialized with {max_workers} workers")
    
    def execute_strategy(
        self,
        reasoning: ReasoningResult,
        original_query: str,
        top_k: int = 10
    ) -> OrchestrationResult:
        """
        Ejecuta la estrategia determinada por el ReasoningAgent.
        
        Args:
            reasoning: Resultado del análisis de razonamiento
            original_query: Consulta original del usuario
            top_k: Número máximo de resultados por herramienta
        
        Returns:
            OrchestrationResult con los resultados fusionados
        """
        start_time = time.time()
        logger.info(f"Executing strategy: {reasoning.search_strategy} with tools: {reasoning.tools_to_use}")
        
        try:
            # NUEVA FUNCIONALIDAD: Manejar consultas conversacionales sin herramientas
            if reasoning.search_strategy == "conversational" or not reasoning.tools_to_use:
                logger.info("Handling conversational query - no document search needed")
                result = OrchestrationResult(
                    final_results=[],
                    tool_results=[],
                    total_execution_time=time.time() - start_time,
                    strategy_used="conversational",
                    fusion_method="none",
                    success=True,
                    reasoning_trace="Conversational query - no document retrieval required"
                )
                return result
            
            if reasoning.search_strategy == "single_stage":
                result = self._execute_single_stage(reasoning, original_query, top_k)
            elif reasoning.search_strategy == "multi_stage":
                result = self._execute_multi_stage(reasoning, original_query, top_k)
            elif reasoning.search_strategy == "comparative":
                result = self._execute_comparative(reasoning, original_query, top_k)
            elif reasoning.search_strategy == "exploratory":
                result = self._execute_exploratory(reasoning, original_query, top_k)
            else:
                # Fallback a single stage
                logger.warning(f"Unknown strategy {reasoning.search_strategy}, using single_stage")
                result = self._execute_single_stage(reasoning, original_query, top_k)
            
            result.total_execution_time = time.time() - start_time
            logger.info(f"Strategy execution completed in {result.total_execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing strategy: {e}")
            return OrchestrationResult(
                final_results=[],
                tool_results=[],
                total_execution_time=time.time() - start_time,
                strategy_used=reasoning.search_strategy,
                fusion_method="error",
                success=False,
                reasoning_trace=f"Error en ejecución: {str(e)}"
            )
    
    def _execute_single_stage(
        self,
        reasoning: ReasoningResult,
        query: str,
        top_k: int
    ) -> OrchestrationResult:
        """Ejecuta estrategia de una sola etapa"""
        
        # Usar las consultas optimizadas si están disponibles
        queries_to_use = reasoning.search_queries if reasoning.search_queries else [query]
        
        # Ejecutar herramientas en paralelo
        tool_results = self._execute_tools_parallel(
            reasoning.tools_to_use,
            queries_to_use,
            top_k
        )
        
        # Fusionar resultados
        if len(reasoning.tools_to_use) == 1:
            # Una sola herramienta, no necesita fusión
            final_results = tool_results[0].results if tool_results else []
            fusion_method = "single_tool"
        else:
            # Múltiples herramientas, usar RRF
            final_results = self._fuse_results_rrf(tool_results, top_k)
            fusion_method = "rrf"
        
        return OrchestrationResult(
            final_results=final_results,
            tool_results=tool_results,
            total_execution_time=0,  # Se calculará en execute_strategy
            strategy_used="single_stage",
            fusion_method=fusion_method,
            success=len(final_results) > 0,
            reasoning_trace=f"Ejecutada estrategia single_stage con {len(reasoning.tools_to_use)} herramientas"
        )
    
    def _execute_multi_stage(
        self,
        reasoning: ReasoningResult,
        query: str,
        top_k: int
    ) -> OrchestrationResult:
        """Ejecuta estrategia multi-etapa"""
        
        all_tool_results = []
        stage_results = []
        
        # Ejecutar herramientas en secuencia
        for i, tool in enumerate(reasoning.tools_to_use):
            logger.info(f"Executing stage {i+1}: {tool}")
            
            # Usar consulta específica si está disponible
            stage_query = reasoning.search_queries[i] if i < len(reasoning.search_queries) else query
            
            # Ejecutar herramienta individual
            tool_result = self._execute_single_tool(tool, stage_query, top_k)
            all_tool_results.append(tool_result)
            
            if tool_result.success and tool_result.results:
                stage_results.extend(tool_result.results)
        
        # Fusionar todos los resultados de todas las etapas
        final_results = self._deduplicate_and_rank(stage_results, top_k)
        
        return OrchestrationResult(
            final_results=final_results,
            tool_results=all_tool_results,
            total_execution_time=0,
            strategy_used="multi_stage",
            fusion_method="sequential",
            success=len(final_results) > 0,
            reasoning_trace=f"Ejecutada estrategia multi_stage con {len(reasoning.tools_to_use)} etapas"
        )
    
    def _execute_comparative(
        self,
        reasoning: ReasoningResult,
        query: str,
        top_k: int
    ) -> OrchestrationResult:
        """Ejecuta estrategia comparativa"""
        
        # Para comparaciones, ejecutar múltiples consultas en paralelo
        queries = reasoning.search_queries if reasoning.search_queries else [query]
        
        all_tool_results = []
        
        # Ejecutar cada consulta con todas las herramientas
        for i, comp_query in enumerate(queries):
            logger.info(f"Executing comparative query {i+1}: {comp_query[:50]}...")
            
            # Ejecutar herramientas para esta consulta
            query_results = self._execute_tools_parallel(
                reasoning.tools_to_use,
                [comp_query],
                top_k // len(queries)  # Dividir top_k entre consultas
            )
            
            all_tool_results.extend(query_results)
        
        # Fusionar todos los resultados
        final_results = self._fuse_results_rrf(all_tool_results, top_k)
        
        return OrchestrationResult(
            final_results=final_results,
            tool_results=all_tool_results,
            total_execution_time=0,
            strategy_used="comparative",
            fusion_method="rrf_comparative",
            success=len(final_results) > 0,
            reasoning_trace=f"Ejecutada estrategia comparativa con {len(queries)} consultas"
        )
    
    def _execute_exploratory(
        self,
        reasoning: ReasoningResult,
        query: str,
        top_k: int
    ) -> OrchestrationResult:
        """Ejecuta estrategia exploratoria"""
        
        # Para exploración, usar híbrido con múltiples variaciones de consulta
        tools_to_use = ["hybrid_search", "semantic_search"]
        
        # Generar variaciones de la consulta para exploración
        query_variations = self._generate_exploratory_queries(query, reasoning)
        
        # Ejecutar con múltiples variaciones
        all_tool_results = []
        
        for tool in tools_to_use:
            for variation in query_variations:
                tool_result = self._execute_single_tool(tool, variation, top_k // len(query_variations))
                all_tool_results.append(tool_result)
        
        # Fusionar con diversidad
        final_results = self._fuse_results_diverse(all_tool_results, top_k)
        
        return OrchestrationResult(
            final_results=final_results,
            tool_results=all_tool_results,
            total_execution_time=0,
            strategy_used="exploratory",
            fusion_method="diverse",
            success=len(final_results) > 0,
            reasoning_trace=f"Ejecutada estrategia exploratoria con {len(query_variations)} variaciones"
        )
    
    def _execute_tools_parallel(
        self,
        tools: List[str],
        queries: List[str],
        top_k: int
    ) -> List[ToolResult]:
        """Ejecuta múltiples herramientas en paralelo"""
        
        tool_results = []
        
        # Crear tareas para cada combinación herramienta-consulta
        tasks = []
        for tool in tools:
            for query in queries:
                tasks.append((tool, query, top_k))
        
        # Ejecutar en paralelo con ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Enviar todas las tareas
            future_to_task = {
                executor.submit(self._execute_single_tool, tool, query, top_k): (tool, query)
                for tool, query, top_k in tasks
            }
            
            # Recoger resultados conforme se completan
            for future in as_completed(future_to_task, timeout=self.timeout_seconds):
                tool, query = future_to_task[future]
                try:
                    result = future.result()
                    tool_results.append(result)
                except Exception as e:
                    logger.error(f"Error executing {tool} with query '{query[:50]}...': {e}")
                    # Crear resultado de error
                    error_result = ToolResult(
                        tool_name=tool,
                        query=query,
                        results=[],
                        execution_time=0.0,
                        success=False,
                        error_message=str(e)
                    )
                    tool_results.append(error_result)
        
        return tool_results
    
    def _execute_single_tool(
        self,
        tool_name: str,
        query: str,
        top_k: int
    ) -> ToolResult:
        """Ejecuta una sola herramienta"""
        
        start_time = time.time()
        
        try:
            # Obtener retriever del cache o crear nuevo
            if tool_name not in self._retriever_cache:
                self._retriever_cache[tool_name] = self._create_retriever(tool_name)
            
            retriever = self._retriever_cache[tool_name]
            
            # Ejecutar búsqueda
            results = retriever.search(query, top_k=top_k)
            
            execution_time = time.time() - start_time
            
            return ToolResult(
                tool_name=tool_name,
                query=query,
                results=results,
                execution_time=execution_time,
                success=True,
                metadata={"top_k": top_k}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error executing {tool_name}: {e}")
            
            return ToolResult(
                tool_name=tool_name,
                query=query,
                results=[],
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
    
    def _create_retriever(self, tool_name: str):
        """Crea un retriever basado en el nombre de la herramienta"""
        
        if tool_name == "semantic_search":
            return self.retriever_factory.create_semantic_retriever(self.config_path)
        elif tool_name == "keyword_search":
            return self.retriever_factory.create_keyword_retriever(self.config_path)
        elif tool_name == "hybrid_search":
            return self.retriever_factory.create_hybrid_retriever(self.config_path)
        elif tool_name == "metadata_search":
            return self.retriever_factory.create_metadata_retriever(self.config_path)
        elif tool_name == "graph_search":
            return self.retriever_factory.create_graph_retriever(self.config_path)
        else:
            logger.warning(f"Unknown tool {tool_name}, using hybrid_search")
            return self.retriever_factory.create_hybrid_retriever(self.config_path)
    
    def _fuse_results_rrf(
        self,
        tool_results: List[ToolResult],
        top_k: int,
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Fusiona resultados usando Reciprocal Rank Fusion (RRF).
        
        Args:
            tool_results: Resultados de las herramientas
            top_k: Número de resultados finales
            k: Parámetro RRF (default 60)
        
        Returns:
            Lista de resultados fusionados
        """
        if not tool_results:
            return []
        
        # Recopilar todos los documentos únicos
        doc_scores = {}
        
        for tool_result in tool_results:
            if not tool_result.success or not tool_result.results:
                continue
            
            for rank, doc in enumerate(tool_result.results):
                doc_id = doc.get('id', f"{doc.get('source', 'unknown')}_{rank}")
                
                # Calcular score RRF
                rrf_score = 1.0 / (k + rank + 1)
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        'document': doc,
                        'rrf_score': 0.0,
                        'tool_count': 0,
                        'tools': []
                    }
                
                doc_scores[doc_id]['rrf_score'] += rrf_score
                doc_scores[doc_id]['tool_count'] += 1
                doc_scores[doc_id]['tools'].append(tool_result.tool_name)
        
        # Ordenar por score RRF y devolver top_k
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x['rrf_score'],
            reverse=True
        )
        
        # Preparar resultados finales
        final_results = []
        for item in sorted_docs[:top_k]:
            doc = item['document'].copy()
            doc['fusion_score'] = item['rrf_score']
            doc['fusion_method'] = 'rrf'
            doc['contributing_tools'] = item['tools']
            final_results.append(doc)
        
        return final_results
    
    def _fuse_results_diverse(
        self,
        tool_results: List[ToolResult],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Fusiona resultados priorizando diversidad"""
        
        if not tool_results:
            return []
        
        all_results = []
        
        # Recopilar todos los resultados
        for tool_result in tool_results:
            if tool_result.success and tool_result.results:
                for doc in tool_result.results:
                    doc_copy = doc.copy()
                    doc_copy['source_tool'] = tool_result.tool_name
                    all_results.append(doc_copy)
        
        # Deduplicar manteniendo diversidad
        seen_sources = set()
        diverse_results = []
        
        for doc in all_results:
            source = doc.get('source', 'unknown')
            if source not in seen_sources or len(diverse_results) < top_k // 2:
                diverse_results.append(doc)
                seen_sources.add(source)
                
                if len(diverse_results) >= top_k:
                    break
        
        return diverse_results[:top_k]
    
    def _deduplicate_and_rank(
        self,
        results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Deduplica y rankea resultados"""
        
        if not results:
            return []
        
        # Deduplicar por ID o contenido
        seen_ids = set()
        unique_results = []
        
        for doc in results:
            doc_id = doc.get('id', doc.get('content', '')[:100])
            
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_results.append(doc)
        
        # Ordenar por score si está disponible
        if unique_results and 'score' in unique_results[0]:
            unique_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return unique_results[:top_k]
    
    def _generate_exploratory_queries(
        self,
        query: str,
        reasoning: ReasoningResult
    ) -> List[str]:
        """Genera variaciones de consulta para exploración"""
        
        variations = [query]  # Incluir consulta original
        
        # Agregar consultas más generales
        words = query.split()
        if len(words) > 2:
            # Consulta con palabras clave principales
            key_words = words[:2] if len(words) > 2 else words
            variations.append(" ".join(key_words))
        
        # Agregar consultas de los identificadores extraídos
        for identifier in reasoning.extracted_identifiers:
            if identifier.lower() not in query.lower():
                variations.append(identifier)
        
        # Limitar número de variaciones
        return variations[:3]
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de ejecución"""
        
        return {
            "cached_retrievers": len(self._retriever_cache),
            "retriever_types": list(self._retriever_cache.keys()),
            "max_workers": self.max_workers,
            "timeout_seconds": self.timeout_seconds
        }
