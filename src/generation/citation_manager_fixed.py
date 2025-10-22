"""
Fixed CitationManager - Preserva datos de imagen para procesamiento multimodal

Este módulo contiene una versión corregida del CitationManager que preserva
los datos de imagen (image_base64, metadata) cuando procesa las fuentes,
permitiendo que el LLM pueda acceder a las imágenes para análisis visual.

Cambio principal:
- process_sources() ahora preserva image_base64 y metadata de imágenes
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """Representa una cita con su fuente y contexto"""
    id: str
    source_title: str
    page_number: Optional[int] = None
    section: Optional[str] = None
    confidence: float = 1.0
    context: Optional[str] = None


class FixedCitationManager:
    """
    Gestor de citas y fuentes corregido que preserva datos de imagen.
    
    MODIFICACIÓN CRÍTICA: process_sources() preserva image_base64 y metadata
    para permitir el procesamiento visual por parte del LLM.
    """

    def __init__(self):
        """Inicializa el gestor de citas"""
        self.citations: List[Citation] = []
        self.sources: Dict[str, Dict[str, Any]] = {}
        logger.info("FixedCitationManager initialized - preserves image data")

    def process_sources(self, sources: List[Dict[str, Any]], answer: str = None) -> List[Dict[str, Any]]:
        """
        Procesa las fuentes y extrae información de citas.
        MODIFICADO: Preserva datos de imagen para procesamiento multimodal.
        
        Args:
            sources: Lista de fuentes de búsqueda
            answer: Respuesta generada (opcional)
            
        Returns:
            Lista de fuentes procesadas con datos de imagen preservados
        """
        processed_sources = []
        
        for i, source in enumerate(sources):
            # Estructura base (original)
            processed_source = {
                'id': f"[{i+1}]",
                'doc_id': source.get('doc_id', f'doc_{i}'),
                'title': self._extract_title(source),
                'score': source.get('rrf_score', source.get('score', 0.0)),
                'source': self._extract_title(source)
            }
            
            # MODIFICACIÓN CRÍTICA: Preservar datos de imagen
            if self._has_image_data(source):
                logger.debug(f"Preserving image data for source {i+1}: {processed_source['title']}")
                
                # Preservar image_base64 directamente
                if 'image_base64' in source:
                    processed_source['image_base64'] = source['image_base64']
                
                # Preservar metadata con datos de imagen
                metadata = source.get('metadata', {})
                if metadata:
                    # Preservar metadata completo si contiene datos de imagen
                    if 'image_base64' in metadata or 'has_image' in metadata:
                        processed_source['metadata'] = metadata
                        processed_source['has_image'] = metadata.get('has_image', True)
                        processed_source['image_context'] = metadata.get('image_context', f"Visual content from {processed_source['title']}")
                        processed_source['image_id'] = metadata.get('image_id', f'img_{i+1}')
                        processed_source['image_format'] = metadata.get('image_format', 'PNG')
                        
                        # Si image_base64 está en metadata, copiarlo al nivel superior
                        if 'image_base64' in metadata and 'image_base64' not in processed_source:
                            processed_source['image_base64'] = metadata['image_base64']
                
                # Marcar como fuente con imagen
                processed_source['content_type'] = 'multimodal'
                
                # Log para debugging
                img_data_length = len(processed_source.get('image_base64', ''))
                logger.info(f"Preserved image data for {processed_source['title']}: {img_data_length} characters")
            else:
                processed_source['content_type'] = 'text'
            
            processed_sources.append(processed_source)
        
        # Log resumen
        image_sources = sum(1 for s in processed_sources if s.get('has_image', False))
        logger.info(f"Processed {len(processed_sources)} sources, {image_sources} with image data preserved")
        
        return processed_sources

    def _has_image_data(self, source: Dict[str, Any]) -> bool:
        """
        Verifica si una fuente contiene datos de imagen.
        
        Args:
            source: Fuente a verificar
            
        Returns:
            True si la fuente contiene datos de imagen
        """
        # Verificar image_base64 directo
        if 'image_base64' in source and source['image_base64']:
            return True
        
        # Verificar en metadata
        metadata = source.get('metadata', {})
        if isinstance(metadata, dict):
            if 'image_base64' in metadata and metadata['image_base64']:
                return True
            if metadata.get('has_image', False):
                return True
            if metadata.get('is_multimodal', False):
                return True
        
        # Verificar por extensión de archivo
        title = self._extract_title(source)
        if title and any(ext in title.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']):
            return True
        
        return False

    def _extract_title(self, source: Dict[str, Any]) -> str:
        """Extrae el título de una fuente"""
        # Prioridad: title > source_file > file_name > source > doc_id
        for field in ['title', 'source_file', 'file_name', 'source', 'doc_id']:
            if field in source and source[field]:
                return str(source[field])
        
        return "Documento sin título"

    def add_citation(
        self,
        source_id: str,
        source_title: str,
        page_number: Optional[int] = None,
        section: Optional[str] = None,
        confidence: float = 1.0,
        context: Optional[str] = None
    ) -> Citation:
        """Añade una nueva cita"""
        citation = Citation(
            id=source_id,
            source_title=source_title,
            page_number=page_number,
            section=section,
            confidence=confidence,
            context=context
        )
        
        self.citations.append(citation)
        self.sources[source_id] = {
            'title': source_title,
            'page_number': page_number,
            'section': section,
            'confidence': confidence,
            'context': context
        }
        
        logger.debug(f"Added citation: {source_id} - {source_title}")
        return citation

    def get_citations(self) -> List[Citation]:
        """Retorna todas las citas"""
        return self.citations.copy()

    def get_sources(self) -> Dict[str, Dict[str, Any]]:
        """Retorna todas las fuentes"""
        return self.sources.copy()

    def format_citations(self, style: str = "apa") -> str:
        """Formatea las citas según el estilo especificado"""
        if not self.citations:
            return ""
        
        formatted_citations = []
        
        for citation in self.citations:
            if style.lower() == "apa":
                formatted = self._format_apa_citation(citation)
            elif style.lower() == "mla":
                formatted = self._format_mla_citation(citation)
            else:
                formatted = self._format_simple_citation(citation)
            
            formatted_citations.append(formatted)
        
        return "\n".join(formatted_citations)

    def _format_apa_citation(self, citation: Citation) -> str:
        """Formatea una cita en estilo APA"""
        parts = [f"{citation.id}. {citation.source_title}"]
        
        if citation.page_number:
            parts.append(f"p. {citation.page_number}")
        
        if citation.section:
            parts.append(f"Sección: {citation.section}")
        
        return ". ".join(parts)

    def _format_mla_citation(self, citation: Citation) -> str:
        """Formatea una cita en estilo MLA"""
        parts = [f"{citation.source_title}"]
        
        if citation.page_number:
            parts.append(f"{citation.page_number}")
        
        return f"{citation.id}. " + ". ".join(parts)

    def _format_simple_citation(self, citation: Citation) -> str:
        """Formatea una cita de forma simple"""
        return f"{citation.id}. {citation.source_title}"

    def extract_citations_from_answer(self, answer: str) -> List[str]:
        """Extrae referencias de citas del texto de respuesta"""
        # Buscar patrones como [1], [2], etc.
        citation_pattern = r'\[(\d+)\]'
        matches = re.findall(citation_pattern, answer)
        return list(set(matches))  # Eliminar duplicados

    def validate_citations(self, answer: str) -> Tuple[bool, List[str]]:
        """Valida que todas las citas en la respuesta tengan fuentes correspondientes"""
        cited_ids = self.extract_citations_from_answer(answer)
        available_ids = [citation.id.strip('[]') for citation in self.citations]
        
        missing_citations = []
        for cited_id in cited_ids:
            if cited_id not in available_ids:
                missing_citations.append(cited_id)
        
        is_valid = len(missing_citations) == 0
        return is_valid, missing_citations

    def get_citation_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de las citas"""
        total_citations = len(self.citations)
        unique_sources = len(set(citation.source_title for citation in self.citations))
        avg_confidence = sum(citation.confidence for citation in self.citations) / total_citations if total_citations > 0 else 0
        
        return {
            'total_citations': total_citations,
            'unique_sources': unique_sources,
            'average_confidence': avg_confidence,
            'sources_with_pages': sum(1 for c in self.citations if c.page_number),
            'sources_with_sections': sum(1 for c in self.citations if c.section)
        }

    def clear_citations(self):
        """Limpia todas las citaciones y fuentes"""
        self.citations = []
        self.sources = {}
        logger.info("Citations cleared")

    def merge_duplicate_sources(self) -> int:
        """Fusiona fuentes duplicadas basándose en el título"""
        original_count = len(self.citations)
        
        # Agrupar por título
        title_groups = {}
        for citation in self.citations:
            title = citation.source_title.lower().strip()
            if title not in title_groups:
                title_groups[title] = []
            title_groups[title].append(citation)
        
        # Mantener solo una cita por título (la de mayor confianza)
        self.citations = []
        for title, group in title_groups.items():
            best_citation = max(group, key=lambda c: c.confidence)
            self.citations.append(best_citation)
        
        # Actualizar sources
        self.sources = {}
        for citation in self.citations:
            self.sources[citation.id] = {
                'title': citation.source_title,
                'page_number': citation.page_number,
                'section': citation.section,
                'confidence': citation.confidence,
                'context': citation.context
            }
        
        merged_count = original_count - len(self.citations)
        if merged_count > 0:
            logger.info(f"Merged {merged_count} duplicate sources")
        
        return merged_count
