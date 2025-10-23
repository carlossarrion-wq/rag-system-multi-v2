"""
Specialized Retrievers - Complete version for EC2 with OpenSearch
Herramientas de búsqueda especializadas adaptadas para OpenSearch
"""

import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """Clase base para todos los retrievers especializados"""
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Recupera documentos relevantes.
        
        Args:
            query: Consulta del usuario
            top_k: Número de documentos a retornar
            **kwargs: Parámetros adicionales específicos del retriever
        
        Returns:
            Lista de documentos con metadata
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Retorna el nombre del retriever"""
        pass


class SemanticRetriever(BaseRetriever):
    """
    Retriever basado únicamente en búsqueda semántica (vectorial).
    Adaptado para OpenSearch en EC2.
    """
    
    def __init__(self, hybrid_retriever):
        """
        Inicializa el retriever semántico.
        
        Args:
            hybrid_retriever: Instancia de HybridRetriever (OpenSearch)
        """
        self.hybrid_retriever = hybrid_retriever
    
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Búsqueda puramente semántica usando OpenSearch.
        
        Args:
            query: Consulta del usuario
            top_k: Número de documentos a retornar
            **kwargs: Parámetros adicionales (threshold, etc.)
        
        Returns:
            Lista de documentos ordenados por similitud semántica
        """
        # Usar HybridRetriever con énfasis en búsqueda vectorial
        results = self.hybrid_retriever.search(query, top_k=top_k * 2)
        
        # Filtrar por threshold si se proporciona
        threshold = kwargs.get('threshold', 0.0)
        if threshold > 0:
            results = [r for r in results if r.get('score', 0) >= threshold]
        
        # Limitar a top_k
        results = results[:top_k]
        return results
    
    def search(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Alias para compatibilidad con AdvancedConversationalAgent"""
        return self.retrieve(query, top_k, **kwargs)
    
    def get_name(self) -> str:
        return "semantic_search"


class KeywordRetriever(BaseRetriever):
    """
    Retriever basado únicamente en búsqueda léxica (BM25).
    Adaptado para OpenSearch en EC2.
    """
    
    def __init__(self, hybrid_retriever):
        """
        Inicializa el retriever de palabras clave.
        
        Args:
            hybrid_retriever: Instancia de HybridRetriever (OpenSearch)
        """
        self.hybrid_retriever = hybrid_retriever
    
    def retrieve(self, query: str, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        Búsqueda puramente léxica usando OpenSearch BM25.
        
        Args:
            query: Consulta del usuario
            top_k: Número de documentos a retornar
            **kwargs: Parámetros adicionales
        
        Returns:
            Lista de documentos ordenados por relevancia BM25
        """
        # Usar HybridRetriever con énfasis en búsqueda léxica
        results = self.hybrid_retriever.search(query, top_k=top_k)
        return results
    
    def search(self, query: str, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """Alias para compatibilidad con AdvancedConversationalAgent"""
        return self.retrieve(query, top_k, **kwargs)
    
    def get_name(self) -> str:
        return "keyword_search"


class MetadataRetriever(BaseRetriever):
    """
    Retriever que filtra por metadatos (fecha, tipo, autor, etc.).
    Adaptado para OpenSearch en EC2.
    """
    
    def __init__(self, hybrid_retriever):
        """
        Inicializa el retriever de metadatos.
        
        Args:
            hybrid_retriever: Instancia de HybridRetriever (OpenSearch)
        """
        self.hybrid_retriever = hybrid_retriever
    
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Búsqueda con filtros de metadatos usando OpenSearch.
        
        Args:
            query: Consulta del usuario
            top_k: Número de documentos a retornar
            **kwargs: Filtros de metadatos (date_from, date_to, doc_type, etc.)
        
        Returns:
            Lista de documentos filtrados por metadatos
        """
        logger.debug(f"MetadataRetriever: Searching with filters: {kwargs}")
        
        # Si hay filtro de documento específico, usar estrategia diferente
        has_doc_filter = 'doc_title' in kwargs or 'doc_title_contains' in kwargs
        
        if has_doc_filter:
            return self._retrieve_from_specific_document(query, top_k, kwargs)
        else:
            return self._retrieve_with_general_filters(query, top_k, kwargs)
    
    def _retrieve_from_specific_document(
        self, 
        query: str, 
        top_k: int, 
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Recupera chunks de un documento específico usando OpenSearch.
        
        Args:
            query: Consulta del usuario
            top_k: Número de documentos a retornar
            filters: Filtros que incluyen doc_title o doc_title_contains
        
        Returns:
            Lista de chunks del documento rankeados por relevancia
        """
        doc_filter = filters.get('doc_title_contains') or filters.get('doc_title', '')
        
        # Usar HybridRetriever con filtros de documento
        results = self.hybrid_retriever.search(query, top_k=top_k * 3)
        
        # Filtrar por documento específico
        filtered_results = []
        for result in results:
            doc_title = result.get('metadata', {}).get('doc_title', '')
            if doc_filter.lower() in doc_title.lower():
                filtered_results.append(result)
        
        # Limitar a top_k
        filtered_results = filtered_results[:top_k]
        
        logger.debug(f"MetadataRetriever: Found {len(filtered_results)} chunks from document '{doc_filter}'")
        return filtered_results
    
    def _retrieve_with_general_filters(
        self,
        query: str,
        top_k: int,
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Búsqueda con filtros generales usando OpenSearch.
        
        Args:
            query: Consulta del usuario
            top_k: Número de documentos a retornar
            filters: Filtros de metadatos
        
        Returns:
            Lista de documentos filtrados
        """
        # Hacer búsqueda híbrida
        results = self.hybrid_retriever.search(query, top_k=top_k * 3)
        
        # Aplicar filtros de metadatos
        filtered_results = []
        for doc in results:
            if self._matches_filters(doc, filters):
                filtered_results.append(doc)
        
        # Limitar a top_k
        filtered_results = filtered_results[:top_k]
        
        logger.debug(f"MetadataRetriever: Found {len(filtered_results)} documents after filtering")
        return filtered_results
    
    def _matches_filters(self, doc: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """
        Verifica si un documento cumple con los filtros.
        
        Args:
            doc: Documento a verificar
            filters: Diccionario de filtros
        
        Returns:
            True si el documento cumple todos los filtros
        """
        metadata = doc.get('metadata', {})
        
        # Filtro por tipo de documento
        if 'doc_type' in filters:
            doc_type = metadata.get('type', '')
            if doc_type != filters['doc_type']:
                return False
        
        # Filtro por fecha (desde)
        if 'date_from' in filters:
            doc_date = metadata.get('date', '')
            if doc_date < filters['date_from']:
                return False
        
        # Filtro por fecha (hasta)
        if 'date_to' in filters:
            doc_date = metadata.get('date', '')
            if doc_date > filters['date_to']:
                return False
        
        # Filtro por autor
        if 'author' in filters:
            doc_author = metadata.get('author', '')
            if doc_author != filters['author']:
                return False
        
        # Filtro por tags
        if 'tags' in filters:
            doc_tags = metadata.get('tags', [])
            required_tags = filters['tags']
            if not all(tag in doc_tags for tag in required_tags):
                return False
        
        # Filtro por título de documento (match exacto)
        if 'doc_title' in filters:
            doc_title = metadata.get('doc_title', '')
            if doc_title != filters['doc_title']:
                return False
        
        # Filtro por título de documento (contiene)
        if 'doc_title_contains' in filters:
            doc_title = metadata.get('doc_title', '')
            search_term = filters['doc_title_contains']
            if search_term.lower() not in doc_title.lower():
                return False
        
        return True
    
    def search(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Alias para compatibilidad con AdvancedConversationalAgent"""
        return self.retrieve(query, top_k, **kwargs)
    
    def get_name(self) -> str:
        return "metadata_filter"


class HybridRetrieverWrapper(BaseRetriever):
    """
    Wrapper para el HybridRetriever existente.
    Mantiene compatibilidad con OpenSearch en EC2.
    """
    
    def __init__(self, hybrid_retriever):
        """
        Inicializa el wrapper.
        
        Args:
            hybrid_retriever: Instancia de HybridRetriever (OpenSearch)
        """
        self.hybrid_retriever = hybrid_retriever
    
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Búsqueda híbrida (vector + lexical + RRF) usando OpenSearch.
        
        Args:
            query: Consulta del usuario
            top_k: Número de documentos a retornar
            **kwargs: Parámetros adicionales
        
        Returns:
            Lista de documentos con fusión RRF
        """
        results = self.hybrid_retriever.search(query, top_k=top_k)
        return results
    
    def search(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Alias para compatibilidad con AdvancedConversationalAgent"""
        return self.retrieve(query, top_k, **kwargs)
    
    def get_name(self) -> str:
        return "hybrid_search"


class GraphRetriever(BaseRetriever):
    """
    Retriever basado en relaciones entre documentos.
    Adaptado para OpenSearch en EC2 (con fallback semántico).
    """
    
    def __init__(self, hybrid_retriever):
        """
        Inicializa el retriever de grafos.
        
        Args:
            hybrid_retriever: Instancia de HybridRetriever (para fallback)
        """
        self.hybrid_retriever = hybrid_retriever
    
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Búsqueda basada en relaciones (con fallback a semántica).
        
        Args:
            query: Consulta del usuario
            top_k: Número de documentos a retornar
            **kwargs: Parámetros adicionales
        
        Returns:
            Lista de documentos relacionados
        """
        logger.debug("GraphRetriever: Using fallback to hybrid search (graph not implemented)")
        
        # Fallback a búsqueda híbrida
        results = self.hybrid_retriever.search(query, top_k=top_k)
        
        return results
    
    def search(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Alias para compatibilidad con AdvancedConversationalAgent"""
        return self.retrieve(query, top_k, **kwargs)
    
    def get_name(self) -> str:
        return "graph_search"


class RetrieverFactory:
    """Factory para crear retrievers especializados adaptados para OpenSearch EC2"""
    
    @staticmethod
    def create_all_retrievers(
        vector_indexer,
        lexical_indexer,
        hybrid_retriever
    ) -> Dict[str, BaseRetriever]:
        """
        Crea todas las instancias de retrievers para OpenSearch EC2.
        
        Args:
            vector_indexer: No usado en EC2 (OpenSearch maneja vectores)
            lexical_indexer: No usado en EC2 (OpenSearch maneja BM25)
            hybrid_retriever: Instancia de HybridRetriever (OpenSearch)
        
        Returns:
            Diccionario con todos los retrievers disponibles
        """
        retrievers = {
            'semantic_search': SemanticRetriever(hybrid_retriever),
            'keyword_search': KeywordRetriever(hybrid_retriever),
            'metadata_search': MetadataRetriever(hybrid_retriever),
            'hybrid_search': HybridRetrieverWrapper(hybrid_retriever),
            'graph_search': GraphRetriever(hybrid_retriever),
            'document_search': MetadataRetriever(hybrid_retriever),  # Alias
            'section_search': MetadataRetriever(hybrid_retriever),   # Alias
        }
        return retrievers


    @staticmethod
    def create_semantic_retriever(config_path: str = "config/multi_app_config.yaml", application: str = "darwin"):
        """Crea un retriever semántico usando OpenSearch"""
        from ..retrieval.hybrid_retriever_fixed import HybridRetrieverFixed as HybridRetriever
        
        hr = HybridRetriever(config_path=config_path, application=application)
        return SemanticRetriever(hr)

    @staticmethod
    def create_hybrid_retriever(config_path: str = "config/multi_app_config.yaml", application: str = "darwin"):
        """Crea un retriever híbrido usando OpenSearch"""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"RetrieverFactory.create_hybrid_retriever called with:")
        logger.info(f"  - config_path: {config_path}")
        logger.info(f"  - application: {application}")
        
        try:
            logger.debug("Importing HybridRetrieverFixed...")
            from ..retrieval.hybrid_retriever_fixed import HybridRetrieverFixed as HybridRetriever
            
            logger.debug("Creating HybridRetrieverFixed instance...")
            hybrid_retriever = HybridRetriever(config_path=config_path, application=application)
            
            logger.debug("Creating HybridRetrieverWrapper...")
            wrapper = HybridRetrieverWrapper(hybrid_retriever)
            
            logger.info("Hybrid retriever created successfully")
            return wrapper
            
        except Exception as e:
            logger.error(f"Error in create_hybrid_retriever: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception args: {e.args}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    @staticmethod
    def create_keyword_retriever(config_path: str = "config/multi_app_config.yaml", application: str = "darwin"):
        """Crea un retriever de palabras clave usando OpenSearch"""
        from ..retrieval.hybrid_retriever_fixed import HybridRetrieverFixed as HybridRetriever
        
        hr = HybridRetriever(config_path=config_path, application=application)
        return KeywordRetriever(hr)

    @staticmethod
    def create_metadata_retriever(config_path: str = "config/multi_app_config.yaml", application: str = "darwin"):
        """Crea un retriever de metadatos usando OpenSearch"""
        from ..retrieval.hybrid_retriever_fixed import HybridRetrieverFixed as HybridRetriever
        
        hybrid_retriever = HybridRetriever(config_path=config_path, application=application)
        return MetadataRetriever(hybrid_retriever)

    @staticmethod
    def create_graph_retriever(config_path: str = "config/multi_app_config.yaml", application: str = "darwin"):
        """Crea un retriever de grafos usando OpenSearch"""
        from ..retrieval.hybrid_retriever_fixed import HybridRetrieverFixed as HybridRetriever
        
        hybrid_retriever = HybridRetriever(config_path=config_path, application=application)
        return GraphRetriever(hybrid_retriever)
