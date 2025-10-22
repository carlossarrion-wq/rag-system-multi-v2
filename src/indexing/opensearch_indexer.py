import json
import boto3
from typing import List, Dict, Any, Optional
from loguru import logger
import hashlib
from datetime import datetime

class OpenSearchIndexer:
    def __init__(self, connection_manager):
        self.conn_manager = connection_manager
        self.opensearch_client = connection_manager.get_opensearch_client()
        self.bedrock_client = connection_manager.get_bedrock_client()
        self.config = connection_manager.config
        self.index_name = self.config['services']['opensearch']['index_name']
        
    def create_index(self):
        """Create the OpenSearch index with proper mapping"""
        try:
            if self.opensearch_client.indices.exists(index=self.index_name):
                logger.info(f"Index {self.index_name} already exists")
                return True
            
            # Index mapping for multimodal RAG
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
                        }
                    }
                },
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 1
                }
            }
            
            response = self.opensearch_client.indices.create(
                index=self.index_name,
                body=mapping
            )
            
            logger.info(f"Created index {self.index_name}: {response}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating index: {e}")
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
                logger.debug(f"Generating multimodal embedding (text + image)")
            else:
                # Text-only embedding
                body = json.dumps({
                    "inputText": text
                })
                logger.debug(f"Generating text-only embedding")
            
            response = self.bedrock_client.invoke_model(
                modelId=self.config['bedrock']['embedding_model'],
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
            logger.error(f"Error generating embedding: {e}")
            return []
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start + chunk_size - 100, start), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def index_document(self, document: Dict[str, Any]) -> bool:
        """Index a single document with chunking and multimodal support"""
        try:
            if not document.get('content'):
                logger.warning(f"No content to index for {document.get('file_name', 'unknown')}")
                return False
            
            # Check if document has images
            has_images = len(document.get('images', [])) > 0
            is_image_document = document.get('file_extension', '').lower() in ['.png', '.jpg', '.jpeg']
            
            if is_image_document and has_images:
                # For image documents, create a single chunk with multimodal embedding
                logger.info(f"Processing image document: {document['file_name']}")
                
                chunk_id = f"{document['file_hash']}_img_0"
                image_data = document['images'][0]['data']  # Get base64 image data
                
                # Generate multimodal embedding (text description + image)
                embedding = self.generate_embedding(document['content'], image_data)
                if not embedding:
                    logger.error(f"Failed to generate multimodal embedding for {document['file_name']}")
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
                    "image_base64": image_data,  # ✅ FIXED: Store image data for LLM access
                    "metadata": {
                        "file_size": document.get('file_size', 0),
                        "file_extension": document.get('file_extension', ''),
                        "total_chunks": 1,
                        "is_multimodal": True,
                        "image_format": document['images'][0].get('format', 'unknown'),
                        "image_size": document['images'][0].get('size', (0, 0)),
                        "has_image": True,
                        "image_id": chunk_id,
                        "image_context": f"Visual content from {document.get('file_name', 'unknown file')}",
                        "image_base64": image_data,  # ✅ DUPLICATE: Also store in metadata for retrieval
                        **document.get('metadata', {})
                    },
                    "timestamp": datetime.now().isoformat(),
                    "document_type": "image",
                    "has_images": True,
                    "image_count": len(document.get('images', []))
                }
                
                # Index the image chunk
                response = self.opensearch_client.index(
                    index=self.index_name,
                    id=chunk_id,
                    body=doc_to_index
                )
                
                logger.info(f"Successfully indexed multimodal chunk for {document['file_name']}: {response['result']}")
                return True
                
            else:
                # Regular text document processing
                chunks = self.chunk_text(document['content'])
                logger.info(f"Created {len(chunks)} chunks for {document['file_name']}")
                
                # Index each chunk
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{document['file_hash']}_{i}"
                    
                    # Generate text embedding
                    embedding = self.generate_embedding(chunk)
                    if not embedding:
                        logger.error(f"Failed to generate embedding for chunk {i}")
                        continue
                    
                    # Prepare document for indexing
                    doc_to_index = {
                        "content": chunk,
                        "title": document.get('file_name', ''),
                        "file_path": document.get('file_path', ''),
                        "file_name": document.get('file_name', ''),
                        "chunk_id": chunk_id,
                        "chunk_index": i,
                        "embedding": embedding,
                        "metadata": {
                            "file_size": document.get('file_size', 0),
                            "file_extension": document.get('file_extension', ''),
                            "total_chunks": len(chunks),
                            "is_multimodal": False,
                            **document.get('metadata', {})
                        },
                        "timestamp": datetime.now().isoformat(),
                        "document_type": document.get('file_extension', '').replace('.', ''),
                        "has_images": has_images,
                        "image_count": len(document.get('images', []))
                    }
                    
                    # Index the chunk
                    response = self.opensearch_client.index(
                        index=self.index_name,
                        id=chunk_id,
                        body=doc_to_index
                    )
                    
                    logger.debug(f"Indexed chunk {i} for {document['file_name']}: {response['result']}")
                
                logger.info(f"Successfully indexed {len(chunks)} chunks for {document['file_name']}")
                return True
            
        except Exception as e:
            logger.error(f"Error indexing document {document.get('file_name', 'unknown')}: {e}")
            return False
    
    def search_documents(self, query: str, size: int = 5, similarity_threshold: float = 0.7) -> List[Dict]:
        """Search documents using hybrid search (text + vector)"""
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query)
            if not query_embedding:
                logger.error("Failed to generate query embedding")
                return []
            
            # Hybrid search query
            search_body = {
                "size": size,
                "query": {
                    "bool": {
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
                        'metadata': hit['_source'].get('metadata', {})
                    }
                    results.append(result)
            
            logger.info(f"Found {len(results)} relevant documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the index"""
        try:
            stats = self.opensearch_client.indices.stats(index=self.index_name)
            count_response = self.opensearch_client.count(index=self.index_name)
            
            return {
                'total_documents': count_response['count'],
                'index_size': stats['indices'][self.index_name]['total']['store']['size_in_bytes'],
                'status': 'healthy'
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def delete_index(self):
        """Delete the index (use with caution)"""
        try:
            if self.opensearch_client.indices.exists(index=self.index_name):
                response = self.opensearch_client.indices.delete(index=self.index_name)
                logger.info(f"Deleted index {self.index_name}: {response}")
                return True
            else:
                logger.info(f"Index {self.index_name} does not exist")
                return True
        except Exception as e:
            logger.error(f"Error deleting index: {e}")
            return False
