"""
Document Context Enhancer for Multi-Application RAG System
Provides document inventory and summaries to enhance LLM context for better caching (2048+ chars for Haiku 3)
"""

import json
import sys
import tempfile
import yaml
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.multi_app_config_manager import MultiAppConfigManager
from src.utils.connection_manager import ConnectionManager
from loguru import logger


class DocumentContextEnhancer:
    """
    Enhances conversational context with document inventory and summaries.
    Ensures context reaches 2048+ characters for optimal Haiku 3 caching.
    """
    
    def __init__(self, app_name: str, config_path: str = "config/multi_app_config.yaml"):
        """
        Initialize document context enhancer.
        
        Args:
            app_name: Application name
            config_path: Path to multi-application configuration
        """
        self.app_name = app_name
        self.config_manager = MultiAppConfigManager(config_path)
        self.app_config = self.config_manager.get_application_config(app_name)
        self.application_info = self.config_manager.get_application_info(app_name)
        
        # Initialize connection manager
        legacy_config = self.config_manager.create_legacy_config(app_name)
        temp_config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(legacy_config, temp_config_file, default_flow_style=False)
        temp_config_file.close()
        
        self.conn_manager = ConnectionManager(config_path=temp_config_file.name)
        self.s3_client = self.conn_manager.get_s3_client()
        self.opensearch_client = self.conn_manager.get_opensearch_client()
        
        # Clean up temp file
        os.unlink(temp_config_file.name)
        
        # Application-specific settings
        self.s3_bucket = self.app_config['services']['s3']['bucket']
        self.index_name = self.app_config['opensearch']['index_name']
        
        # Cache for document inventory
        self._inventory_cache = None
        self._cache_timestamp = None
        self._cache_ttl = 300  # 5 minutes cache TTL
        
        logger.info(f"[{app_name}] DocumentContextEnhancer initialized")
    
    def get_enhanced_context_from_chunks(self, query: str, retrieved_chunks: List[Dict[str, Any]], max_context_length: int = 4000) -> Dict[str, Any]:
        """
        Get enhanced context based ONLY on documents from retrieved chunks.
        This replaces the old method that sent ALL documents from the repository.
        
        Args:
            query: User query for context relevance
            retrieved_chunks: List of chunks retrieved from vector search
            max_context_length: Maximum context length to generate
            
        Returns:
            Dictionary with enhanced context information
        """
        try:
            # Extract unique document identifiers from retrieved chunks
            chunk_documents = self._extract_documents_from_chunks(retrieved_chunks)
            
            if not chunk_documents:
                return self._create_minimal_context()
            
            # Get document inventory (for metadata and summaries)
            inventory = self._get_document_inventory()
            
            # Filter inventory to only include documents from retrieved chunks
            filtered_inventory = self._filter_inventory_by_chunks(inventory, chunk_documents) if inventory else None
            
            # Build enhanced context using only chunk-related documents
            context_parts = []
            
            # 1. Application overview (reduced, focused on retrieved documents)
            app_overview = self._build_chunk_focused_overview(chunk_documents, filtered_inventory)
            context_parts.append(app_overview)
            
            # 2. Retrieved documents summary
            retrieved_docs_summary = self._build_retrieved_documents_summary(chunk_documents, filtered_inventory)
            context_parts.append(retrieved_docs_summary)
            
            # 3. Document summaries for retrieved documents only
            if filtered_inventory:
                relevant_summaries = self._get_summaries_for_retrieved_docs(chunk_documents, filtered_inventory)
                if relevant_summaries:
                    summaries_context = self._build_summaries_context(relevant_summaries)
                    context_parts.append(summaries_context)
            
            # 4. Key terms and topics from retrieved documents only
            terms_context = self._build_terms_context_from_chunks(chunk_documents, filtered_inventory)
            context_parts.append(terms_context)
            
            # Combine all parts
            full_context = "\n\n".join(context_parts)
            
            # Ensure minimum length for caching
            if len(full_context) < 2048:
                full_context = self._pad_context_for_caching_chunks(full_context, chunk_documents, filtered_inventory)
            
            # Truncate if too long
            if len(full_context) > max_context_length:
                full_context = full_context[:max_context_length - 100] + "\n\n[Context truncated for length...]"
            
            return {
                'enhanced_context': full_context,
                'context_length': len(full_context),
                'document_count': len(chunk_documents),
                'chunk_count': len(retrieved_chunks),
                'application_name': self.application_info['name'],
                'cache_optimized': len(full_context) >= 2048,
                'generated_at': datetime.now().isoformat(),
                'query_processed': query[:100] + "..." if len(query) > 100 else query,
                'context_source': 'retrieved_chunks_only'
            }
            
        except Exception as e:
            logger.error(f"[{self.app_name}] Error generating chunk-based enhanced context: {e}")
            return self._create_minimal_context()

    
    def _get_document_inventory(self) -> Optional[Dict[str, Any]]:
        """Get document inventory from S3 with caching."""
        try:
            # Check cache
            current_time = datetime.now().timestamp()
            if (self._inventory_cache and self._cache_timestamp and 
                (current_time - self._cache_timestamp) < self._cache_ttl):
                return self._inventory_cache
            
            # Fetch from S3
            inventory_key = f"applications/{self.app_name}/inventory/document_inventory_latest.json"
            
            response = self.s3_client.get_object(
                Bucket=self.s3_bucket,
                Key=inventory_key
            )
            
            inventory = json.loads(response['Body'].read())
            
            # Update cache
            self._inventory_cache = inventory
            self._cache_timestamp = current_time
            
            logger.debug(f"[{self.app_name}] Loaded document inventory: {inventory.get('total_documents', 0)} documents")
            return inventory
            
        except Exception as e:
            logger.warning(f"[{self.app_name}] Could not load document inventory: {e}")
            return None
    
    
    
    
    def _build_summaries_context(self, relevant_summaries: List[Dict[str, Any]]) -> str:
        """Build context from relevant document summaries."""
        if not relevant_summaries:
            return ""
        
        context = "=== RELEVANT DOCUMENT SUMMARIES ===\n\n"
        
        for i, doc in enumerate(relevant_summaries, 1):
            context += f"{i}. {doc.get('file_name', 'Unknown Document')}\n"
            context += f"   Type: {doc.get('document_type', 'unknown').upper()}\n"
            context += f"   Relevance: {doc.get('relevance_score', 0):.1f}/10\n"
            context += f"   Summary: {doc.get('summary', 'No summary available')}\n"
            
            # Add key terms if available
            key_terms = doc.get('key_terms', [])
            if key_terms:
                context += f"   Key Terms: {', '.join(key_terms[:5])}\n"
            
            context += "\n"
        
        return context
    
    
    
    def _create_minimal_context(self) -> Dict[str, Any]:
        """Create minimal context when inventory is not available."""
        minimal_context = f"""=== {self.application_info['name']} - KNOWLEDGE BASE ===

Application: {self.application_info['name']}
Description: {self.application_info['description']}
Status: Document inventory not available

This is a specialized AI assistant for {self.application_info['name']}. 
While the full document inventory is not currently accessible, I can still 
help with general questions and provide assistance based on my training data.

For the most accurate and up-to-date information specific to {self.application_info['name']}, 
please ensure the document inventory system is properly configured and accessible.

System Context:
- Application domain: {self.application_info['name']}
- Response mode: General assistance with domain awareness
- Knowledge base: Limited to training data
- Recommendation: Verify document inventory system status

This context provides the minimum 2048 characters required for optimal 
Haiku 3 model caching while maintaining useful system information for responses."""
        
        # Pad to ensure minimum length
        while len(minimal_context) < 2048:
            minimal_context += f"\n\nAdditional context padding for {self.application_info['name']} domain optimization and caching performance."
        
        return {
            'enhanced_context': minimal_context,
            'context_length': len(minimal_context),
            'document_count': 0,
            'application_name': self.application_info['name'],
            'cache_optimized': len(minimal_context) >= 2048,
            'generated_at': datetime.now().isoformat(),
            'query_processed': 'N/A - Minimal context mode',
            'status': 'minimal_context'
        }
    
    
    def reduce_context_for_images(self, context: str, max_tokens: int = 30000) -> str:
        """
        Reduce context size when images are present to stay within token limits
        
        Args:
            context: The original context string
            max_tokens: Maximum tokens to allow for context (default 30K to leave room for images and system prompt)
        
        Returns:
            Reduced context string
        """
        # Rough estimation: 4 characters per token
        max_chars = max_tokens * 4
        
        if len(context) <= max_chars:
            return context
        
        # Split context into sections
        lines = context.split('\n')
        
        # Priority order for keeping content:
        # 1. Headers and titles (lines starting with #, ##, etc.)
        # 2. Short lines (likely important summaries)
        # 3. Lines with keywords related to diagrams
        # 4. Other content
        
        priority_lines = []
        diagram_lines = []
        short_lines = []
        other_lines = []
        
        diagram_keywords = ['diagrama', 'flujo', 'proceso', 'usuario', '700', '900', 'nif', 'workflow']
        
        for line in lines:
            line_lower = line.lower()
            
            if line.startswith('#') or line.startswith('**') or line.startswith('Title:'):
                priority_lines.append(line)
            elif any(keyword in line_lower for keyword in diagram_keywords):
                diagram_lines.append(line)
            elif len(line.strip()) < 100 and line.strip():
                short_lines.append(line)
            else:
                other_lines.append(line)
        
        # Reconstruct context with priorities
        reduced_lines = []
        current_chars = 0
        
        # Add priority content first
        for line_group in [priority_lines, diagram_lines, short_lines, other_lines]:
            for line in line_group:
                if current_chars + len(line) + 1 < max_chars:
                    reduced_lines.append(line)
                    current_chars += len(line) + 1
                else:
                    break
            if current_chars >= max_chars * 0.9:  # Stop at 90% to be safe
                break
        
        reduced_context = '\n'.join(reduced_lines)
        
        # Add truncation notice
        if len(reduced_context) < len(context):
            reduced_context += "\n\n[Context truncated to fit within token limits when processing images]"
        
        return reduced_context
    
    def reduce_context_for_text_only(self, context: str, max_tokens: int = 40000) -> str:
        """
        Reduce context size for text-only mode to stay within token limits
        
        Args:
            context: The original context string
            max_tokens: Maximum tokens to allow for context in text-only mode
        
        Returns:
            Reduced context string
        """
        # Rough estimation: 4 characters per token
        max_chars = max_tokens * 4
        
        if len(context) <= max_chars:
            return context
        
        # Split context into sections
        lines = context.split('\n')
        
        # Priority order for keeping content:
        # 1. Headers and titles (lines starting with #, ##, etc.)
        # 2. Short lines (likely important summaries)
        # 3. Lines with keywords related to diagrams
        # 4. Other content
        
        priority_lines = []
        diagram_lines = []
        short_lines = []
        other_lines = []
        
        diagram_keywords = ['diagrama', 'flujo', 'proceso', 'usuario', '700', '900', 'nif', 'workflow']
        
        for line in lines:
            line_lower = line.lower()
            
            if line.startswith('#') or line.startswith('**') or line.startswith('Title:'):
                priority_lines.append(line)
            elif any(keyword in line_lower for keyword in diagram_keywords):
                diagram_lines.append(line)
            elif len(line.strip()) < 80 and line.strip():
                short_lines.append(line)
            else:
                other_lines.append(line)
        
        # Reconstruct context with priorities
        reduced_lines = []
        current_chars = 0
        
        # Add priority content first
        for line_group in [priority_lines, diagram_lines, short_lines, other_lines]:
            for line in line_group:
                if current_chars + len(line) + 1 < max_chars:
                    reduced_lines.append(line)
                    current_chars += len(line) + 1
                else:
                    break
            if current_chars >= max_chars * 0.8:  # Stop at 80% to be safe
                break
        
        reduced_context = '\n'.join(reduced_lines)
        
        # Add truncation notice
        if len(reduced_context) < len(context):
            reduced_context += "\n\n[Context truncated to fit within token limits for text-only processing]"
        
        return reduced_context
    
    def get_opensearch_document_stats(self) -> Dict[str, Any]:
        """Get additional document statistics from OpenSearch."""
        try:
            # Get document count from OpenSearch
            count_response = self.opensearch_client.count(
                index=self.index_name,
                body={
                    "query": {
                        "term": {
                            "application_id": self.app_name
                        }
                    }
                }
            )
            
            # Get sample documents for analysis
            sample_response = self.opensearch_client.search(
                index=self.index_name,
                body={
                    "size": 50,
                    "query": {
                        "term": {
                            "application_id": self.app_name
                        }
                    },
                    "_source": ["document_type", "has_images", "file_name", "document_summary"]
                }
            )
            
            # Analyze results
            total_chunks = count_response['count']
            documents_with_summaries = 0
            documents_with_images = 0
            
            for hit in sample_response['hits']['hits']:
                source = hit['_source']
                if source.get('document_summary'):
                    documents_with_summaries += 1
                if source.get('has_images'):
                    documents_with_images += 1
            
            return {
                'total_chunks_in_opensearch': total_chunks,
                'sample_documents_analyzed': len(sample_response['hits']['hits']),
                'documents_with_summaries': documents_with_summaries,
                'documents_with_images': documents_with_images,
                'opensearch_index': self.index_name
            }
            
        except Exception as e:
            logger.warning(f"[{self.app_name}] Could not get OpenSearch stats: {e}")
            return {}

    # NEW METHODS FOR CHUNK-BASED FILTERING
    
    def _extract_documents_from_chunks(self, retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract unique document identifiers from retrieved chunks.
        
        Args:
            retrieved_chunks: List of chunks retrieved from vector search
            
        Returns:
            List of unique document information extracted from chunks
        """
        unique_docs = {}
        
        for chunk in retrieved_chunks:
            # Extract document identifier from chunk metadata
            metadata = chunk.get('metadata', {})
            
            # Try different possible field names for document identification
            doc_id = (
                metadata.get('source_file') or 
                metadata.get('file_name') or 
                metadata.get('document_id') or 
                chunk.get('source') or
                chunk.get('doc_id') or
                'unknown_document'
            )
            
            if doc_id not in unique_docs:
                unique_docs[doc_id] = {
                    'file_name': doc_id,
                    'document_type': metadata.get('document_type', 'unknown'),
                    'chunk_count': 0,
                    'total_score': 0.0,
                    'max_score': 0.0,
                    'has_images': metadata.get('has_images', False),
                    'chunks': []
                }
            
            # Update document statistics
            score = chunk.get('score', chunk.get('rrf_score', 0.0))
            unique_docs[doc_id]['chunk_count'] += 1
            unique_docs[doc_id]['total_score'] += score
            unique_docs[doc_id]['max_score'] = max(unique_docs[doc_id]['max_score'], score)
            unique_docs[doc_id]['chunks'].append({
                'text': chunk.get('text', chunk.get('content', '')),
                'score': score,
                'chunk_id': chunk.get('chunk_id', '')
            })
        
        # Convert to list and add average scores
        document_list = []
        for doc_id, doc_info in unique_docs.items():
            doc_info['avg_score'] = doc_info['total_score'] / doc_info['chunk_count'] if doc_info['chunk_count'] > 0 else 0.0
            document_list.append(doc_info)
        
        # Sort by relevance (max score, then chunk count)
        document_list.sort(key=lambda x: (x['max_score'], x['chunk_count']), reverse=True)
        
        logger.info(f"[{self.app_name}] Extracted {len(document_list)} unique documents from {len(retrieved_chunks)} chunks")
        
        return document_list
    
    def _filter_inventory_by_chunks(self, inventory: Dict[str, Any], chunk_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Filter document inventory to only include documents from retrieved chunks.
        
        Args:
            inventory: Full document inventory from S3
            chunk_documents: Documents extracted from retrieved chunks
            
        Returns:
            Filtered inventory containing only relevant documents
        """
        if not inventory or not chunk_documents:
            return inventory
        
        # Create set of document names from chunks for fast lookup
        chunk_doc_names = {doc['file_name'].lower() for doc in chunk_documents}
        
        # Filter document summary
        original_docs = inventory.get('document_summary', [])
        filtered_docs = []
        
        for doc in original_docs:
            doc_name = doc.get('file_name', '').lower()
            if doc_name in chunk_doc_names:
                filtered_docs.append(doc)
        
        # Create filtered inventory
        filtered_inventory = {
            'total_documents': len(filtered_docs),
            'document_summary': filtered_docs,
            'generated_at': inventory.get('generated_at'),
            'statistics': self._calculate_filtered_statistics(filtered_docs),
            'filter_info': {
                'original_document_count': inventory.get('total_documents', 0),
                'filtered_document_count': len(filtered_docs),
                'chunk_document_count': len(chunk_documents),
                'filter_type': 'retrieved_chunks_only'
            }
        }
        
        logger.info(f"[{self.app_name}] Filtered inventory: {len(filtered_docs)} documents (from {inventory.get('total_documents', 0)} total)")
        
        return filtered_inventory
    
    def _calculate_filtered_statistics(self, filtered_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics for filtered document set."""
        if not filtered_docs:
            return {}
        
        doc_types = set()
        all_topics = set()
        all_terms = set()
        relevance_scores = []
        
        for doc in filtered_docs:
            doc_types.add(doc.get('document_type', 'unknown'))
            all_topics.update(doc.get('topics', []))
            all_terms.update(doc.get('key_terms', []))
            
            relevance = doc.get('relevance_score', 0)
            if relevance > 0:
                relevance_scores.append(relevance)
        
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        
        return {
            'document_types': list(doc_types),
            'total_topics': len(all_topics),
            'total_key_terms': len(all_terms),
            'avg_relevance_score': avg_relevance,
            'documents_with_relevance': len(relevance_scores)
        }
    
    def _build_chunk_focused_overview(self, chunk_documents: List[Dict[str, Any]], filtered_inventory: Optional[Dict[str, Any]]) -> str:
        """Build application overview focused on retrieved documents."""
        doc_count = len(chunk_documents)
        chunk_count = sum(doc['chunk_count'] for doc in chunk_documents)
        
        overview = f"""=== {self.application_info['name']} - RETRIEVED DOCUMENTS CONTEXT ===

Application: {self.application_info['name']}
Description: {self.application_info['description']}
Retrieved Documents: {doc_count} documents
Retrieved Chunks: {chunk_count} chunks
Context Source: Vector search results only

This context contains ONLY information from documents that were retrieved based on your query.
This ensures maximum relevance and reduces noise from unrelated documentation."""
        
        if filtered_inventory:
            stats = filtered_inventory.get('statistics', {})
            overview += f"""
Document Types in Results: {', '.join(stats.get('document_types', []))}
Average Relevance Score: {stats.get('avg_relevance_score', 0.0):.1f}/10.0"""
        
        return overview
    
    def _build_retrieved_documents_summary(self, chunk_documents: List[Dict[str, Any]], filtered_inventory: Optional[Dict[str, Any]]) -> str:
        """Build summary of retrieved documents."""
        summary = "=== RETRIEVED DOCUMENTS SUMMARY ===\n\n"
        
        # Group by document type
        doc_types = {}
        for doc in chunk_documents:
            doc_type = doc.get('document_type', 'unknown')
            if doc_type not in doc_types:
                doc_types[doc_type] = []
            doc_types[doc_type].append(doc)
        
        for doc_type, docs in doc_types.items():
            summary += f"{doc_type.upper()} Documents ({len(docs)}):\n"
            
            # Sort by relevance (max score)
            sorted_docs = sorted(docs, key=lambda x: x['max_score'], reverse=True)
            
            for doc in sorted_docs:
                summary += f"  • {doc['file_name']} (Score: {doc['max_score']:.2f}, Chunks: {doc['chunk_count']})\n"
            summary += "\n"
        
        return summary
    
    def _get_summaries_for_retrieved_docs(self, chunk_documents: List[Dict[str, Any]], filtered_inventory: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get document summaries for retrieved documents only."""
        if not filtered_inventory:
            return []
        
        # Get summaries from filtered inventory
        summaries = filtered_inventory.get('document_summary', [])
        
        # Sort by relevance score
        summaries.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return summaries
    
    def _build_terms_context_from_chunks(self, chunk_documents: List[Dict[str, Any]], filtered_inventory: Optional[Dict[str, Any]]) -> str:
        """Build context from key terms and topics of retrieved documents only."""
        if not filtered_inventory:
            return ""
        
        documents = filtered_inventory.get('document_summary', [])
        if not documents:
            return ""
        
        # Collect terms and topics from retrieved documents only
        all_terms = {}
        all_topics = {}
        
        for doc in documents:
            # Count key terms
            for term in doc.get('key_terms', []):
                all_terms[term] = all_terms.get(term, 0) + 1
            
            # Count topics
            for topic in doc.get('topics', []):
                all_topics[topic] = all_topics.get(topic, 0) + 1
        
        context = "=== KEY KNOWLEDGE AREAS (Retrieved Documents Only) ===\n\n"
        
        # Top terms from retrieved documents
        if all_terms:
            sorted_terms = sorted(all_terms.items(), key=lambda x: x[1], reverse=True)
            context += "Key Terms from Retrieved Documents:\n"
            for term, count in sorted_terms[:8]:  # Fewer terms since we have fewer documents
                context += f"  • {term} (appears in {count} retrieved documents)\n"
            context += "\n"
        
        # Top topics from retrieved documents
        if all_topics:
            sorted_topics = sorted(all_topics.items(), key=lambda x: x[1], reverse=True)
            context += "Topics from Retrieved Documents:\n"
            for topic, count in sorted_topics[:6]:  # Fewer topics since we have fewer documents
                context += f"  • {topic} (covered in {count} retrieved documents)\n"
        
        return context
    
    def _pad_context_for_caching_chunks(self, context: str, chunk_documents: List[Dict[str, Any]], filtered_inventory: Optional[Dict[str, Any]]) -> str:
        """Pad context to reach minimum 2048 characters for Haiku 3 caching (chunk-based version)."""
        if len(context) >= 2048:
            return context
        
        # Add additional context to reach minimum length
        padding_parts = []
        
        # Add detailed chunk statistics
        total_chunks = sum(doc['chunk_count'] for doc in chunk_documents)
        avg_score = sum(doc['avg_score'] for doc in chunk_documents) / len(chunk_documents) if chunk_documents else 0.0
        max_score = max(doc['max_score'] for doc in chunk_documents) if chunk_documents else 0.0
        
        padding_parts.append(f"""
=== RETRIEVED DOCUMENTS DETAILED STATISTICS ===

Chunk-Based Context Metrics:
- Retrieved documents: {len(chunk_documents)}
- Total chunks analyzed: {total_chunks}
- Average chunk score: {avg_score:.3f}
- Maximum chunk score: {max_score:.3f}
- Context optimization: Focused on query-relevant documents only
- Knowledge base scope: {self.application_info['name']} domain-specific information
- Context generation: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Relevance Optimization:
This enhanced context contains ONLY information from documents that were retrieved 
based on your specific query. This approach ensures maximum relevance and accuracy 
while reducing noise from unrelated documentation in the {self.application_info['name']} 
knowledge base.""")
        
        # Add document type breakdown for retrieved documents
        if chunk_documents:
            doc_types = {}
            for doc in chunk_documents:
                doc_type = doc.get('document_type', 'unknown')
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            padding_parts.append(f"""
=== RETRIEVED DOCUMENT TYPE BREAKDOWN ===

Document Categories in Results:""")
            
            for doc_type, count in sorted(doc_types.items()):
                padding_parts.append(f"- {doc_type.upper()}: {count} documents")
        
        # Add system context for chunk-based responses
        padding_parts.append(f"""
=== OPTIMIZED RESPONSE CONTEXT ===

Response Generation Guidelines:
1. Prioritize information from the retrieved documents shown above
2. Reference specific documents when citing information
3. Indicate confidence level based on retrieved document relevance scores
4. Focus on query-relevant content from {self.application_info['name']} domain
5. Maintain context awareness of document retrieval scores and relevance

This chunk-based context ensures optimal caching performance for Haiku 3 model 
while providing highly relevant domain knowledge for accurate responses. The context 
is dynamically filtered based on vector search results to maximize relevance.""")
        
        # Combine original context with padding
        padded_context = context + "\n\n" + "\n".join(padding_parts)
        
        return padded_context
