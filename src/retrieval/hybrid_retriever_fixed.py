"""
Fix for Hybrid Retriever - Corrects kNN query structure for OpenSearch
The issue is that kNN queries need to be structured differently in OpenSearch
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
from typing import List, Dict, Any
from loguru import logger

try:
    from ..indexing.opensearch_indexer import OpenSearchIndexer
    from ..utils.connection_manager import ConnectionManager
except ImportError as e:
    logger.error(f"Import error in hybrid_retriever_fixed: {e}")
    raise

class HybridRetrieverFixed:
    def __init__(self, config_path: str = "config/multi_app_config.yaml", application: str = "darwin"):
        logger.info(f"HybridRetrieverFixed.__init__ called with:")
        logger.info(f"  - config_path: {config_path}")
        logger.info(f"  - application: {application}")
        
        self.config_path = config_path
        self.application = application
        
        try:
            # Load config
            logger.debug("Loading YAML configuration...")
            import yaml
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.debug("YAML configuration loaded successfully")
            
            # Validate config structure
            logger.debug("Validating configuration structure...")
            if 'opensearch' not in self.config:
                raise KeyError("'opensearch' key not found in configuration")
            if 'applications' not in self.config:
                raise KeyError("'applications' key not found in configuration")
            if application not in self.config['applications']:
                raise KeyError(f"Application '{application}' not found in configuration")
            
            logger.debug("Configuration structure validated")
            
            # Create a compatible config structure for ConnectionManager
            # The ConnectionManager expects the endpoint WITHOUT https:// prefix
            logger.debug("Processing OpenSearch endpoint...")
            opensearch_endpoint = self.config['opensearch']['endpoint']
            logger.debug(f"Original endpoint: {opensearch_endpoint}")
            
            if opensearch_endpoint.startswith('https://'):
                opensearch_endpoint = opensearch_endpoint.replace('https://', '')
            elif opensearch_endpoint.startswith('http://'):
                opensearch_endpoint = opensearch_endpoint.replace('http://', '')
            
            logger.debug(f"Processed endpoint: {opensearch_endpoint}")
            
            logger.debug("Building compatible configuration...")
            compatible_config = {
                'aws': self.config['aws'],
                'bedrock': self.config['bedrock'],
                'services': {
                    'opensearch': {
                        'endpoint': f"https://{opensearch_endpoint}",
                        'use_ssl': self.config['opensearch'].get('use_ssl', True),
                        'verify_certs': self.config['opensearch'].get('verify_certs', True),
                        'connection_class': self.config['opensearch'].get('connection_class', 'RequestsHttpConnection'),
                        'vpc_access': self.config['opensearch'].get('vpc_access', True),
                        'timeout': self.config['opensearch'].get('timeout', 30)
                    },
                    'postgresql': self.config.get('postgresql', {}),
                    's3': {
                        'bucket': self.config['applications'][application]['s3']['bucket']
                    }
                }
            }
            
            # Debug logging to help troubleshoot
            logger.info(f"Compatible config created:")
            logger.info(f"  - OpenSearch endpoint: {compatible_config['services']['opensearch']['endpoint']}")
            logger.info(f"  - AWS region: {compatible_config['aws']['region']}")
            logger.info(f"  - S3 bucket: {compatible_config['services']['s3']['bucket']}")
            
            logger.debug("Creating ConnectionManager...")
            self.connection_manager = ConnectionManager(config_dict=compatible_config)
            logger.debug("ConnectionManager created successfully")
            
            logger.debug("Getting OpenSearch client...")
            self.opensearch_client = self.connection_manager.get_opensearch_client()
            logger.debug("OpenSearch client obtained successfully")
            
            logger.debug("Getting Bedrock client...")
            self.bedrock_client = self.connection_manager.get_bedrock_client()
            logger.debug("Bedrock client obtained successfully")

            logger.debug("Setting index name and embedding model...")
            self.index_name = self.config['applications'][application]['opensearch']['index_name']
            self.embedding_model = self.config['bedrock']['embedding_model']
            logger.info(f"Index name: {self.index_name}")
            logger.info(f"Embedding model: {self.embedding_model}")
            
            
            logger.info("HybridRetrieverFixed initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Error in HybridRetrieverFixed.__init__: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception args: {e.args}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

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


    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector and text search
        FIXED: Corrected kNN query structure for OpenSearch
        """
        try:
            # Generate query embedding using original query
            query_embedding = self._get_embedding(query)

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

            # Then, get text results using original query
            text_search_body = {
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
            # Fallback to simple text search with original query
            try:
                return self._fallback_text_search(query, top_k)
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
