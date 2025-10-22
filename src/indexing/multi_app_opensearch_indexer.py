"""
Multi-Application OpenSearch Indexer
Supports indexing documents for multiple applications with separate indices
"""

import json
import boto3
from typing import List, Dict, Any, Optional
from loguru import logger
import hashlib
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.multi_app_config_manager import MultiAppConfigManager
from src.utils.connection_manager import ConnectionManager
from src.indexing.semantic_chunker import SemanticChunker


class MultiAppOpenSearchIndexer:
    """
    Multi-application OpenSearch indexer that supports separate indices per application.
    
    Features:
    - Application-specific index management
    - Application-aware document indexing
    - Configurable chunking per application
    - Multimodal support (text + images)
    - Application metadata tagging
    """
    
    def __init__(self, app_name: Optional[str] = None, 
                 config_path: str = "config/multi_app_config.yaml"):
        """
        Initialize multi-application OpenSearch indexer.
        
        Args:
            app_name: Name of the application to index for
            config_path: Path to multi-application configuration
        """
        self.config_manager = MultiAppConfigManager(config_path)
        self.app_name = app_name or self.config_manager.default_app
        self.application_info = self.config_manager.get_application_info(self.app_name)
        
        # Validate application
        if not self.config_manager.validate_application(self.app_name):
            available_apps = ', '.join(self.config_manager.get_available_applications())
            raise ValueError(f"Application '{self.app_name}' not found. Available: {available_apps}")
        
        # Get application-specific configuration
        self.app_config = self.config_manager.get_application_config(self.app_name)
        self.rag_config = self.config_manager.get_rag_config(self.app_name)
        
        # Initialize connection manager with legacy config
        legacy_config = self.config_manager.create_legacy_config(self.app_name)

        # Create temporary config file for ConnectionManager
        import tempfile
        import yaml
        import os
        temp_config_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(legacy_config, temp_config_file, default_flow_style=False)
        temp_config_file.close()
        self.conn_manager = ConnectionManager(config_path=temp_config_file.name)
        self.opensearch_client = self.conn_manager.get_opensearch_client()
        self.bedrock_client = self.conn_manager.get_bedrock_client()

        # Clean up temp file
        os.unlink(temp_config_file.name)
        
        # Application-specific settings
        self.index_name = self.config_manager.get_opensearch_index_name(self.app_name)
        self.s3_config = self.config_manager.get_s3_config(self.app_name)
        
        # Chunking configuration
        chunking_config = self.rag_config.get('chunking', {})
        self.chunk_size = chunking_config.get('chunk_size', 1500)
        self.chunk_overlap = chunking_config.get('chunk_overlap', 225)
        self.separators = chunking_config.get('separators', ["\n\n", "\n", " ", ""])
        
        # Initialize Semantic Chunker with table preservation
        self.semantic_chunker = SemanticChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        logger.info(f"MultiAppOpenSearchIndexer initialized for {self.application_info['name']} - Index: {self.index_name}")
        
    def create_index(self) -> bool:
        """Create the application-specific OpenSearch index with proper mapping"""
        try:
            if self.opensearch_client.indices.exists(index=self.index_name):
                logger.info(f"Index {self.index_name} already exists for application {self.app_name}")
                return True
            
            # Index mapping for multimodal RAG with application metadata
            mapping = {
                "mappings": {
                    "properties": {
                        "content": {
                            "type": "text",
                            "analyzer": "standard"
                        },
                        "title": {
                            "type": "text"
                        },
                        "file_path": {
                            "type": "keyword"
                        },
                        "file_name": {
                            "type": "keyword"
                        },
                        "chunk_id": {
                            "type": "keyword"
                        },
                        "chunk_index": {
                            "type": "integer"
                        },
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": 1024
                        },
                        "image_base64": {
                            "type": "text",
                            "index": False
                        },
                        "metadata": {
                            "type": "object"
                        },
                        "timestamp": {
                            "type": "date"
                        },
                        "document_type": {
                            "type": "keyword"
                        },
                        "has_images": {
                            "type": "boolean"
                        },
                        "image_count": {
                            "type": "integer"
                        },
                        # Application-specific fields
                        "application_id": {
                            "type": "keyword"
                        },
                        "application_name": {
                            "type": "keyword"
                        },
                        "s3_bucket": {
                            "type": "keyword"
                        },
                        "s3_prefix": {
                            "type": "keyword"
                        }
                    }
                },
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 1,
                    "index": {
                        "knn": True,
                        "knn.algo_param.ef_search": 100
                    }
                }
            }
            
            response = self.opensearch_client.indices.create(
                index=self.index_name,
                body=mapping
            )
            
            logger.info(f"Created index {self.index_name} for application {self.app_name}: {response}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating index {self.index_name}: {e}")
            return False
    
    def generate_embedding(self, text: str, image_base64: Optional[str] = None) -> List[float]:
        """Generate embedding using Bedrock Titan (multimodal if image provided)"""
        try:
            # Prepare the request for Titan Embeddings
            if image_base64:
                # Multimodal embedding (text + image)
                body = json.dumps({
                    "inputText": text,
                    "inputImage": image_base64
                })
            else:
                # Text-only embedding
                body = json.dumps({
                    "inputText": text
                })
            
            response = self.bedrock_client.invoke_model(
                modelId=self.app_config['bedrock']['embedding_model'],
                body=body,
                contentType='application/json',
                accept='application/json'
            )
            
            response_body = json.loads(response['body'].read())
            embedding = response_body.get('embedding', [])
            
            if len(embedding) != 1024:
                logger.warning(f"Unexpected embedding dimension: {len(embedding)}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding for {self.app_name}: {e}")
            return []
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks using application-specific configuration"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at separator boundaries
            if end < len(text):
                # Look for the best separator
                best_break = end
                for separator in self.separators:
                    if separator == "":
                        continue
                    
                    # Look for separator within reasonable distance
                    search_start = max(start + self.chunk_size - 200, start)
                    search_end = min(end + 100, len(text))
                    
                    separator_pos = text.rfind(separator, search_start, search_end)
                    if separator_pos != -1:
                        best_break = separator_pos + len(separator)
                        break
                
                end = best_break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def index_document(self, document: Dict[str, Any]) -> bool:
        """Index a single document with application-specific configuration"""
        try:
            if not document.get('content'):
                logger.warning(f"No content to index for {document.get('file_name', 'unknown')} in {self.app_name}")
                return False
            
            # Check if document has images
            has_images = len(document.get('images', [])) > 0
            is_image_document = document.get('file_extension', '').lower() in ['.png', '.jpg', '.jpeg']
            
            # Add application metadata to document
            document_metadata = {
                **document.get('metadata', {}),
                'application_id': self.app_name,
                'application_name': self.application_info['name'],
                'indexed_at': datetime.now().isoformat(),
                's3_bucket': self.s3_config['bucket'],
                's3_prefix': self.s3_config.get('documents_prefix', ''),
                'chunk_config': {
                    'chunk_size': self.chunk_size,
                    'chunk_overlap': self.chunk_overlap
                }
            }
            
            if is_image_document and has_images:
                # For image documents, create a single chunk with multimodal embedding
                
                chunk_id = f"{self.app_name}_{document['file_hash']}_img_0"
                image_data = document['images'][0]['data']  # Get base64 image data
                
                # Generate multimodal embedding (text description + image)
                embedding = self.generate_embedding(document['content'], image_data)
                if not embedding:
                    logger.error(f"Failed to generate multimodal embedding for {document['file_name']} in {self.app_name}")
                    return False
                
                # Prepare document for indexing
                doc_to_index = {
                    "content": document['content'],
                    "title": document.get('file_name', ''),
                    "file_path": document.get('file_path', ''),
                    "file_name": document.get('file_name', ''),
                    "chunk_id": chunk_id,
                    "chunk_index": 0,
                    "embedding": embedding,
                    "image_base64": image_data,
                    "metadata": {
                        **document_metadata,
                        "file_size": document.get('file_size', 0),
                        "file_extension": document.get('file_extension', ''),
                        "total_chunks": 1,
                        "is_multimodal": True,
                        "image_format": document['images'][0].get('format', 'unknown'),
                        "image_size": document['images'][0].get('size', (0, 0)),
                        "has_image": True,
                        "image_id": chunk_id,
                        "image_context": f"Visual content from {document.get('file_name', 'unknown file')}",
                        "image_base64": image_data
                    },
                    "timestamp": datetime.now().isoformat(),
                    "document_type": "image",
                    "has_images": True,
                    "image_count": len(document.get('images', [])),
                    # Application fields
                    "application_id": self.app_name,
                    "application_name": self.application_info['name'],
                    "s3_bucket": self.s3_config['bucket'],
                    "s3_prefix": self.s3_config.get('documents_prefix', '')
                }
                
                # Index the image chunk
                response = self.opensearch_client.index(
                    index=self.index_name,
                    id=chunk_id,
                    body=doc_to_index
                )
                
                logger.info(f"Successfully indexed multimodal chunk for {document['file_name']} in {self.app_name}: {response['result']}")
                return True
                
            else:
                # Regular text document processing with semantic chunking
                semantic_chunks = self.semantic_chunker.chunk_with_table_preservation(
                    document['content'], 
                    document_metadata
                )
                
                # Index each semantic chunk
                for i, semantic_chunk in enumerate(semantic_chunks):
                    chunk_id = f"{self.app_name}_{document['file_hash']}_{i}"
                    chunk_text = semantic_chunk['text']
                    chunk_metadata = semantic_chunk['metadata']
                    
                    # Generate text embedding
                    embedding = self.generate_embedding(chunk_text)
                    if not embedding:
                        logger.error(f"Failed to generate embedding for semantic chunk {i} in {self.app_name}")
                        continue
                    
                    # Merge document metadata with chunk-specific metadata
                    merged_metadata = {
                        **document_metadata,
                        **chunk_metadata,
                        "file_size": document.get('file_size', 0),
                        "file_extension": document.get('file_extension', ''),
                        "total_chunks": len(semantic_chunks),
                        "is_multimodal": False,
                        "semantic_chunk_index": semantic_chunk.get('chunk_index', i),
                        "chunk_position": semantic_chunk.get('position', 0),
                        "chunk_length": semantic_chunk.get('length', len(chunk_text)),
                        "chunk_type": semantic_chunk.get('chunk_type', 'text')
                    }
                    
                    # Prepare document for indexing
                    doc_to_index = {
                        "content": chunk_text,
                        "title": document.get('file_name', ''),
                        "file_path": document.get('file_path', ''),
                        "file_name": document.get('file_name', ''),
                        "chunk_id": chunk_id,
                        "chunk_index": i,
                        "embedding": embedding,
                        "metadata": merged_metadata,
                        "timestamp": datetime.now().isoformat(),
                        "document_type": document.get('file_extension', '').replace('.', ''),
                        "has_images": has_images,
                        "image_count": len(document.get('images', [])),
                        # Application fields
                        "application_id": self.app_name,
                        "application_name": self.application_info['name'],
                        "s3_bucket": self.s3_config['bucket'],
                        "s3_prefix": self.s3_config.get('documents_prefix', '')
                    }
                    
                    # Index the chunk
                    response = self.opensearch_client.index(
                        index=self.index_name,
                        id=chunk_id,
                        body=doc_to_index
                    )
                    
                    # Log table chunks with more detail
                    if semantic_chunk.get('chunk_type') == 'table':
                        table_info = chunk_metadata.get('table_rows_count', 0)
                        codes_count = len(chunk_metadata.get('technical_codes', []))
                        logger.debug(f"Indexed table chunk {i}: {table_info} rows, {codes_count} codes")
                
                # Log summary of chunk types
                table_chunks = sum(1 for chunk in semantic_chunks if chunk.get('chunk_type') == 'table')
                text_chunks = len(semantic_chunks) - table_chunks
                total_codes = sum(len(chunk['metadata'].get('technical_codes', [])) for chunk in semantic_chunks)
                
                logger.info(f"Indexed {document['file_name']}: {len(semantic_chunks)} chunks ({table_chunks} tables, {total_codes} codes)")
                return True
            
        except Exception as e:
            logger.error(f"Error indexing document {document.get('file_name', 'unknown')} in {self.app_name}: {e}")
            return False
    
    def search_documents(self, query: str, size: int = 5, similarity_threshold: float = 0.7) -> List[Dict]:
        """Search documents using hybrid search (text + vector) within application scope"""
        try:
            # Get application-specific search configuration
            search_config = self.rag_config.get('search', {})
            similarity_threshold = search_config.get('similarity_threshold', similarity_threshold)
            
            # Generate query embedding
            query_embedding = self.generate_embedding(query)
            if not query_embedding:
                logger.error(f"Failed to generate query embedding for {self.app_name}")
                return []
            
            # Hybrid search query with application filter
            search_body = {
                "size": size,
                "query": {
                    "bool": {
                        "must": [
                            # Application filter - CRITICAL for multi-app separation
                            {
                                "term": {
                                    "application_id": self.app_name
                                }
                            }
                        ],
                        "should": [
                            # Text search
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["content^2", "title^1.5", "file_name"],
                                    "type": "best_fields",
                                    "boost": 1.0
                                }
                            },
                            # Vector search
                            {
                                "script_score": {
                                    "query": {"match_all": {}},
                                    "script": {
                                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                        "params": {"query_vector": query_embedding}
                                    },
                                    "boost": 2.0
                                }
                            }
                        ],
                        "minimum_should_match": 1
                    }
                },
                "_source": {
                    "excludes": ["embedding"]  # Don't return embeddings in results
                }
            }
            
            response = self.opensearch_client.search(
                index=self.index_name,
                body=search_body
            )
            
            results = []
            for hit in response['hits']['hits']:
                if hit['_score'] >= similarity_threshold:
                    result = {
                        'content': hit['_source']['content'],
                        'file_name': hit['_source']['file_name'],
                        'file_path': hit['_source']['file_path'],
                        'chunk_index': hit['_source']['chunk_index'],
                        'score': hit['_score'],
                        'metadata': hit['_source'].get('metadata', {}),
                        'application_id': hit['_source'].get('application_id'),
                        'application_name': hit['_source'].get('application_name'),
                        # Include image data if present
                        'image_base64': hit['_source'].get('image_base64'),
                        'has_images': hit['_source'].get('has_images', False)
                    }
                    results.append(result)
            
            logger.info(f"Found {len(results)} relevant documents for query in {self.app_name}: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents in {self.app_name}: {e}")
            return []
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the application-specific index"""
        try:
            stats = self.opensearch_client.indices.stats(index=self.index_name)
            
            # Get document count with application filter
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
            
            return {
                'application_id': self.app_name,
                'application_name': self.application_info['name'],
                'index_name': self.index_name,
                'total_documents': count_response['count'],
                'index_size': stats['indices'][self.index_name]['total']['store']['size_in_bytes'],
                's3_bucket': self.s3_config['bucket'],
                's3_prefix': self.s3_config.get('documents_prefix', ''),
                'status': 'healthy'
            }
        except Exception as e:
            logger.error(f"Error getting index stats for {self.app_name}: {e}")
            return {
                'application_id': self.app_name,
                'status': 'error', 
                'error': str(e)
            }
    
    def delete_application_documents(self) -> bool:
        """Delete all documents for this application from the index"""
        try:
            # Delete by query - only documents for this application
            delete_query = {
                "query": {
                    "term": {
                        "application_id": self.app_name
                    }
                }
            }
            
            response = self.opensearch_client.delete_by_query(
                index=self.index_name,
                body=delete_query
            )
            
            deleted_count = response.get('deleted', 0)
            logger.info(f"Deleted {deleted_count} documents for application {self.app_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents for {self.app_name}: {e}")
            return False
    
    def delete_index(self) -> bool:
        """Delete the entire index (use with caution - affects all applications using this index)"""
        try:
            if self.opensearch_client.indices.exists(index=self.index_name):
                response = self.opensearch_client.indices.delete(index=self.index_name)
                logger.warning(f"Deleted entire index {self.index_name} (affects all applications): {response}")
                return True
            else:
                logger.info(f"Index {self.index_name} does not exist")
                return True
        except Exception as e:
            logger.error(f"Error deleting index {self.index_name}: {e}")
            return False


def create_indexer_for_application(app_name: str, 
                                 config_path: str = "config/multi_app_config.yaml") -> MultiAppOpenSearchIndexer:
    """
    Convenience function to create an indexer for a specific application.
    
    Args:
        app_name: Name of the application
        config_path: Path to multi-application configuration
        
    Returns:
        MultiAppOpenSearchIndexer instance
    """
    return MultiAppOpenSearchIndexer(app_name=app_name, config_path=config_path)
