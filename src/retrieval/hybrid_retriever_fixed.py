"""
Fix for Hybrid Retriever - Corrects kNN query structure for OpenSearch
The issue is that kNN queries need to be structured differently in OpenSearch
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ..indexing.opensearch_indexer import OpenSearchIndexer
from ..utils.connection_manager import ConnectionManager
from ..utils.stop_words_manager import get_stop_words_manager
import json
from typing import List, Dict, Any
from loguru import logger

class HybridRetrieverFixed:
    def __init__(self, config_path: str = "config/aws_config_production.yaml"):
        self.config_path = config_path
        self.connection_manager = ConnectionManager(config_path)
        self.opensearch_client = self.connection_manager.get_opensearch_client()
        self.bedrock_client = self.connection_manager.get_bedrock_client()

        # Load config
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.index_name = self.config['services']['opensearch']['index_name']
        self.embedding_model = self.config['bedrock']['embedding_model']
        
        # Initialize stop words manager
        self.stop_words_manager = get_stop_words_manager()

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Bedrock"""
        try:
            body = json.dumps({
                'inputText': text,
                'embeddingConfig': {'outputEmbeddingLength': 1024}
            })

            response = self.bedrock_client.invoke_model(
                modelId=self.embedding_model,
                body=body,
                contentType='application/json',
                accept='application/json'
            )

            response_body = json.loads(response['body'].read())
            return response_body['embedding']

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def _filter_search_query(self, query: str) -> str:
        """
        Filter stop words from search query to improve search precision
        """
        try:
            # Get filtered words using stop words manager
            filtered_words = self.stop_words_manager.filter_words(query)
            
            # Join the filtered words back into a query string
            filtered_query = ' '.join(filtered_words)
            
            # Log the filtering for debugging
            if filtered_query != query:
                logger.debug(f"Query filtered: '{query}' -> '{filtered_query}'")
            
            # Return original query if filtering results in empty string
            if not filtered_query.strip():
                logger.debug(f"Filtered query is empty, using original: '{query}'")
                return query
                
            return filtered_query
            
        except Exception as e:
            logger.warning(f"Error filtering query, using original: {e}")
            return query

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector and text search
        FIXED: Corrected kNN query structure for OpenSearch
        """
        try:
            # Filter stop words from query for better search precision
            filtered_query = self._filter_search_query(query)
            
            # Generate query embedding using filtered query
            query_embedding = self._get_embedding(filtered_query)

            # FIXED: Use proper hybrid search structure for OpenSearch
            # First, get vector results
            vector_search_body = {
                "size": top_k,
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "_score",
                            "params": {"query_vector": query_embedding}
                        }
                    }
                },
                "_source": ["content", "title", "file_name", "chunk_id", "metadata", 
                           "image_base64", "document_type", "has_images"]
            }

            # Execute vector search
            vector_response = self.opensearch_client.search(
                index=self.index_name,
                body=vector_search_body
            )

            # Then, get text results using filtered query
            text_search_body = {
                "size": top_k,
                "query": {
                    "multi_match": {
                        "query": filtered_query,
                        "fields": ["content^2", "title", "file_name"],
                        "type": "best_fields"
                    }
                },
                "_source": ["content", "title", "file_name", "chunk_id", "metadata", 
                           "image_base64", "document_type", "has_images"]
            }

            # Execute text search
            text_response = self.opensearch_client.search(
                index=self.index_name,
                body=text_search_body
            )

            # Combine and deduplicate results using RRF (Reciprocal Rank Fusion)
            results = self._combine_results_rrf(
                vector_response['hits']['hits'],
                text_response['hits']['hits'],
                top_k
            )

            logger.debug(f"Hybrid search returned {len(results)} results")
            
            # Log image results only if found
            image_results = [r for r in results if r['metadata'].get('has_image')]
            if image_results:
                logger.debug(f"Found {len(image_results)} results with image data")
            
            return results

        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            # Fallback to simple text search with filtered query
            try:
                filtered_query = self._filter_search_query(query)
                return self._fallback_text_search(filtered_query, top_k)
            except Exception as fallback_error:
                logger.error(f"Fallback search also failed: {fallback_error}")
                return []

    def _combine_results_rrf(self, vector_hits: List, text_hits: List, top_k: int, k: int = 60) -> List[Dict[str, Any]]:
        """
        Combine vector and text search results using Reciprocal Rank Fusion (RRF)
        """
        # Create score maps
        vector_scores = {}
        text_scores = {}
        all_docs = {}

        # Process vector results
        for rank, hit in enumerate(vector_hits):
            doc_id = hit['_id']
            vector_scores[doc_id] = 1.0 / (k + rank + 1)
            all_docs[doc_id] = hit

        # Process text results
        for rank, hit in enumerate(text_hits):
            doc_id = hit['_id']
            text_scores[doc_id] = 1.0 / (k + rank + 1)
            if doc_id not in all_docs:
                all_docs[doc_id] = hit

        # Calculate RRF scores
        rrf_scores = {}
        for doc_id in all_docs:
            rrf_scores[doc_id] = vector_scores.get(doc_id, 0) + text_scores.get(doc_id, 0)

        # Sort by RRF score and take top_k
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Build final results
        results = []
        for doc_id, rrf_score in sorted_docs:
            hit = all_docs[doc_id]
            source = hit['_source']
            
            # Get base metadata
            metadata = source.get('metadata', {})
            
            # Add image data to metadata if available
            if source.get('image_base64'):
                metadata['has_image'] = True
                metadata['image_base64'] = source['image_base64']
                metadata['image_id'] = source.get('chunk_id', 'unknown')
                metadata['image_context'] = f"Visual content from {source.get('file_name', 'unknown file')}"
            else:
                metadata['has_image'] = False

            # FIXED: Extract title and filename from available data with fallback chain
            # Try to get filename from multiple sources
            filename = (
                source.get('file_name') or
                source.get('title') or
                metadata.get('file_name') or
                metadata.get('title') or
                metadata.get('source_file') or
                f"document_{doc_id}"  # Last resort
            )
            
            # Try to get title from multiple sources
            title = (
                source.get('title') or
                source.get('file_name') or
                metadata.get('title') or
                metadata.get('file_name') or
                metadata.get('source_file') or
                filename  # Use filename as title if nothing else
            )


            result = {
                'chunk_id': source.get('chunk_id', ''),
                'text': source.get('content', ''),
                'metadata': metadata,
                'score': float(hit['_score']),
                'rrf_score': float(rrf_score),
                'retrieval_method': 'hybrid_rrf',
                'source_file': filename,  # Use extracted filename
                'title': title,          # Use extracted title
                'file_name': filename,   # Add explicit file_name field
                'doc_id': doc_id
            }
            results.append(result)

        return results

    def _fallback_text_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback to simple text search if hybrid fails"""
        search_body = {
            "size": top_k,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["content^2", "title", "file_name"],
                    "type": "best_fields"
                }
            },
            "_source": ["content", "title", "file_name", "chunk_id", "metadata", 
                       "image_base64", "document_type", "has_images"]
        }

        response = self.opensearch_client.search(
            index=self.index_name,
            body=search_body
        )

        results = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            
            # Get base metadata
            metadata = source.get('metadata', {})
            
            # Add image data to metadata if available
            if source.get('image_base64'):
                metadata['has_image'] = True
                metadata['image_base64'] = source['image_base64']
                metadata['image_id'] = source.get('chunk_id', 'unknown')
                metadata['image_context'] = f"Visual content from {source.get('file_name', 'unknown file')}"
            else:
                metadata['has_image'] = False
                
            # FIXED: Extract title and filename from available data with fallback chain (same as hybrid search)
            # Try to get filename from multiple sources
            filename = (
                source.get('file_name') or
                source.get('title') or
                metadata.get('file_name') or
                metadata.get('title') or
                metadata.get('source_file') or
                f"document_{hit['_id']}"  # Last resort
            )
            
            # Try to get title from multiple sources
            title = (
                source.get('title') or
                source.get('file_name') or
                metadata.get('title') or
                metadata.get('file_name') or
                metadata.get('source_file') or
                filename  # Use filename as title if nothing else
            )

            result = {
                'chunk_id': source.get('chunk_id', ''),
                'text': source.get('content', ''),
                'metadata': metadata,
                'score': float(hit['_score']),
                'retrieval_method': 'text_fallback',
                'source_file': filename,  # Use extracted filename
                'title': title,          # Use extracted title
                'file_name': filename,   # Add explicit file_name field
                'doc_id': hit['_id']
            }
            results.append(result)

        return results
