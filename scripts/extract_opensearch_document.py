#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para extraer contenido específico de documentos desde OpenSearch
Busca y muestra el contenido completo de un documento específico por nombre de archivo
"""

import sys
import os
import json
import argparse
import tempfile
import yaml
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.multi_app_config_manager import MultiAppConfigManager
from src.utils.connection_manager import ConnectionManager

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OpenSearchDocumentExtractor:
    """
    Extractor de documentos específicos desde OpenSearch.
    Permite buscar y extraer el contenido completo de documentos por nombre de archivo.
    """
    
    def __init__(self, app_name: str, config_path: str = "config/multi_app_config.yaml"):
        """
        Initialize document extractor.
        
        Args:
            app_name: Application name
            config_path: Path to multi-app configuration file
        """
        self.app_name = app_name
        self.config_path = config_path
        
        # Initialize multi-app configuration
        logger.info(f"Initializing configuration for {app_name}...")
        self.config_manager = MultiAppConfigManager(config_path)
        
        # Validate application
        if not self.config_manager.validate_application(app_name):
            available_apps = ', '.join(self.config_manager.get_available_applications())
            raise ValueError(f"Application '{app_name}' not found. Available: {available_apps}")
        
        self.app_config = self.config_manager.get_application_config(app_name)
        
        # Initialize AWS connections using legacy config
        logger.info(f"[{app_name}] Initializing AWS connections...")
        legacy_config = self.config_manager.create_legacy_config(app_name)
        
        # Create temporary config file for ConnectionManager
        temp_config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(legacy_config, temp_config_file, default_flow_style=False)
        temp_config_file.close()
        
        self.conn_manager = ConnectionManager(config_path=temp_config_file.name)
        self.opensearch_client = self.conn_manager.get_opensearch_client()
        
        # Clean up temp file
        os.unlink(temp_config_file.name)
        
        # Get index name
        self.index_name = self.config_manager.get_opensearch_index_name(app_name)
        
        logger.info(f"[{app_name}] Document extractor initialized - Index: {self.index_name}")
    
    def extract_document_by_filename(self, filename: str, include_metadata: bool = True, 
                                   include_image_data: bool = False) -> dict:
        """
        Extract document content by filename.
        
        Args:
            filename: Name of the file to search for
            include_metadata: Whether to include metadata in results
            include_image_data: Whether to include base64 image data
            
        Returns:
            Dictionary with extraction results
        """
        logger.info(f"[{self.app_name}] Searching for document: {filename}")
        
        try:
            # Prepare source exclusions
            source_excludes = ['embedding']  # Always exclude embeddings (too large)
            if not include_image_data:
                source_excludes.append('image_base64')
            
            # Search for the specific document
            search_body = {
                'query': {
                    'bool': {
                        'must': [
                            {'term': {'application_id': self.app_name}},
                            {'wildcard': {'file_name': f'*{filename}*'}}
                        ]
                    }
                },
                'size': 50,  # Allow multiple chunks
                'sort': [
                    {'chunk_index': {'order': 'asc'}}  # Sort by chunk index
                ],
                '_source': {
                    'excludes': source_excludes
                }
            }
            
            response = self.opensearch_client.search(index=self.index_name, body=search_body)
            hits = response['hits']['hits']
            
            if not hits:
                logger.warning(f"[{self.app_name}] Document '{filename}' not found")
                return {
                    'found': False,
                    'filename': filename,
                    'app_name': self.app_name,
                    'message': f"Document '{filename}' not found in OpenSearch"
                }
            
            logger.info(f"[{self.app_name}] Found {len(hits)} chunk(s) for '{filename}'")
            
            # Process results
            chunks = []
            document_info = None
            
            for hit in hits:
                source = hit['_source']
                
                # Extract document info from first chunk
                if document_info is None:
                    document_info = {
                        'file_name': source.get('file_name', 'N/A'),
                        'file_path': source.get('file_path', 'N/A'),
                        'document_type': source.get('document_type', 'N/A'),
                        'has_images': source.get('has_images', False),
                        'image_count': source.get('image_count', 0),
                        'application_id': source.get('application_id', 'N/A'),
                        'application_name': source.get('application_name', 'N/A'),
                        's3_bucket': source.get('s3_bucket', 'N/A'),
                        's3_prefix': source.get('s3_prefix', 'N/A'),
                        'timestamp': source.get('timestamp', 'N/A')
                    }
                
                # Extract chunk info
                chunk_data = {
                    'chunk_id': hit['_id'],
                    'chunk_index': source.get('chunk_index', 0),
                    'score': hit['_score'],
                    'content': source.get('content', ''),
                    'content_length': len(source.get('content', ''))
                }
                
                # Add metadata if requested
                if include_metadata:
                    chunk_data['metadata'] = source.get('metadata', {})
                
                # Add image data if requested and available
                if include_image_data and 'image_base64' in source:
                    chunk_data['has_image_data'] = True
                    chunk_data['image_data_length'] = len(source.get('image_base64', ''))
                    chunk_data['image_base64'] = source.get('image_base64', '')
                
                chunks.append(chunk_data)
            
            # Combine all content
            combined_content = '\n\n'.join([chunk['content'] for chunk in chunks])
            
            result = {
                'found': True,
                'filename': filename,
                'app_name': self.app_name,
                'index_name': self.index_name,
                'document_info': document_info,
                'total_chunks': len(chunks),
                'total_content_length': len(combined_content),
                'combined_content': combined_content,
                'chunks': chunks,
                'extracted_at': datetime.now().isoformat()
            }
            
            logger.info(f"[{self.app_name}] Successfully extracted '{filename}': {len(chunks)} chunks, {len(combined_content)} characters")
            return result
            
        except Exception as e:
            logger.error(f"[{self.app_name}] Error extracting document '{filename}': {e}")
            return {
                'found': False,
                'filename': filename,
                'app_name': self.app_name,
                'error': str(e),
                'message': f"Error extracting document: {str(e)}"
            }
    
    def search_documents_by_pattern(self, pattern: str, max_results: int = 10) -> list:
        """
        Search for documents matching a pattern.
        
        Args:
            pattern: Search pattern (supports wildcards)
            max_results: Maximum number of results to return
            
        Returns:
            List of matching documents
        """
        logger.info(f"[{self.app_name}] Searching for documents matching pattern: {pattern}")
        
        try:
            search_body = {
                'query': {
                    'bool': {
                        'must': [
                            {'term': {'application_id': self.app_name}},
                            {'wildcard': {'file_name': f'*{pattern}*'}}
                        ]
                    }
                },
                'size': max_results,
                '_source': ['file_name', 'file_path', 'document_type', 'has_images', 'image_count', 'timestamp'],
                'collapse': {
                    'field': 'file_name'  # Collapse by filename to avoid duplicates
                }
            }
            
            response = self.opensearch_client.search(index=self.index_name, body=search_body)
            hits = response['hits']['hits']
            
            results = []
            for hit in hits:
                source = hit['_source']
                results.append({
                    'file_name': source.get('file_name', 'N/A'),
                    'file_path': source.get('file_path', 'N/A'),
                    'document_type': source.get('document_type', 'N/A'),
                    'has_images': source.get('has_images', False),
                    'image_count': source.get('image_count', 0),
                    'timestamp': source.get('timestamp', 'N/A'),
                    'score': hit['_score']
                })
            
            logger.info(f"[{self.app_name}] Found {len(results)} documents matching pattern '{pattern}'")
            return results
            
        except Exception as e:
            logger.error(f"[{self.app_name}] Error searching documents: {e}")
            return []
    
    def save_extraction_to_file(self, extraction_result: dict, output_file: str = None) -> str:
        """
        Save extraction result to a file.
        
        Args:
            extraction_result: Result from extract_document_by_filename
            output_file: Output file path (optional)
            
        Returns:
            Path to saved file
        """
        if not extraction_result.get('found', False):
            logger.error("Cannot save extraction - document not found")
            return ""
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = extraction_result['filename'].replace('.', '_')
            output_file = f"extraction_{self.app_name}_{filename}_{timestamp}.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(extraction_result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[{self.app_name}] Extraction saved to: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"[{self.app_name}] Error saving extraction: {e}")
            return ""


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Extract specific documents from OpenSearch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

EXTRAER DOCUMENTO ESPECÍFICO:
  # Extraer DiagramasDeFlujo.png de mulesoft
  python extract_opensearch_document.py --app mulesoft --filename DiagramasDeFlujo.png
  
  # Extraer con metadatos completos
  python extract_opensearch_document.py --app mulesoft --filename DiagramasDeFlujo.png --include-metadata
  
  # Extraer incluyendo datos de imagen (base64)
  python extract_opensearch_document.py --app mulesoft --filename DiagramasDeFlujo.png --include-image-data
  
  # Guardar resultado en archivo
  python extract_opensearch_document.py --app mulesoft --filename DiagramasDeFlujo.png --save-to-file

BUSCAR DOCUMENTOS:
  # Buscar documentos que contengan "Diagrama"
  python extract_opensearch_document.py --app mulesoft --search Diagrama
  
  # Buscar con más resultados
  python extract_opensearch_document.py --app mulesoft --search Diagrama --max-results 20

MOSTRAR SOLO CONTENIDO:
  # Mostrar solo el contenido combinado (sin metadatos)
  python extract_opensearch_document.py --app mulesoft --filename DiagramasDeFlujo.png --content-only
        """
    )
    
    # Global options
    parser.add_argument(
        '--config',
        default='config/multi_app_config.yaml',
        help='Multi-app configuration file path'
    )
    
    parser.add_argument(
        '--app',
        required=True,
        help='Application name (required)'
    )
    
    # Main actions
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--filename',
        help='Extract specific document by filename'
    )
    group.add_argument(
        '--search',
        help='Search documents by pattern'
    )
    
    # Options for extraction
    parser.add_argument(
        '--include-metadata',
        action='store_true',
        help='Include metadata in extraction results'
    )
    
    parser.add_argument(
        '--include-image-data',
        action='store_true',
        help='Include base64 image data in results'
    )
    
    parser.add_argument(
        '--save-to-file',
        action='store_true',
        help='Save extraction results to JSON file'
    )
    
    parser.add_argument(
        '--output-file',
        help='Specific output file path (used with --save-to-file)'
    )
    
    parser.add_argument(
        '--content-only',
        action='store_true',
        help='Show only the combined content (no metadata or structure)'
    )
    
    # Options for search
    parser.add_argument(
        '--max-results',
        type=int,
        default=10,
        help='Maximum number of search results (default: 10)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize extractor
        extractor = OpenSearchDocumentExtractor(app_name=args.app, config_path=args.config)
        
        if args.filename:
            # Extract specific document
            logger.info(f"Extracting document '{args.filename}' from application '{args.app}'")
            
            result = extractor.extract_document_by_filename(
                filename=args.filename,
                include_metadata=args.include_metadata,
                include_image_data=args.include_image_data
            )
            
            if not result.get('found', False):
                print(f"❌ {result.get('message', 'Document not found')}")
                return 1
            
            # Save to file if requested
            if args.save_to_file:
                output_file = extractor.save_extraction_to_file(result, args.output_file)
                if output_file:
                    print(f"💾 Results saved to: {output_file}")
            
            # Display results
            if args.content_only:
                print("\n" + "="*80)
                print(f"CONTENIDO COMPLETO DE: {result['filename']}")
                print("="*80)
                print(result['combined_content'])
                print("="*80)
            else:
                print("\n" + "="*80)
                print(f"DOCUMENTO EXTRAÍDO: {result['filename']}")
                print("="*80)
                
                doc_info = result['document_info']
                print(f"📄 Archivo: {doc_info['file_name']}")
                print(f"📁 Ruta: {doc_info['file_path']}")
                print(f"📊 Tipo: {doc_info['document_type']}")
                print(f"🖼️  Tiene imágenes: {doc_info['has_images']}")
                print(f"🔢 Número de imágenes: {doc_info['image_count']}")
                print(f"🏷️  Aplicación: {doc_info['application_name']} ({doc_info['application_id']})")
                print(f"🪣 S3 Bucket: {doc_info['s3_bucket']}")
                print(f"📅 Timestamp: {doc_info['timestamp']}")
                print(f"🧩 Total chunks: {result['total_chunks']}")
                print(f"📏 Longitud total: {result['total_content_length']} caracteres")
                
                print("\n" + "-"*80)
                print("CONTENIDO COMBINADO:")
                print("-"*80)
                print(result['combined_content'])
                print("-"*80)
                
                if args.include_metadata and result['chunks']:
                    print("\n" + "-"*80)
                    print("INFORMACIÓN DE CHUNKS:")
                    print("-"*80)
                    for i, chunk in enumerate(result['chunks'], 1):
                        print(f"Chunk {i}:")
                        print(f"  ID: {chunk['chunk_id']}")
                        print(f"  Índice: {chunk['chunk_index']}")
                        print(f"  Score: {chunk['score']:.4f}")
                        print(f"  Longitud: {chunk['content_length']} caracteres")
                        if chunk.get('has_image_data'):
                            print(f"  Datos de imagen: {chunk['image_data_length']} bytes")
                        print()
        
        elif args.search:
            # Search documents
            logger.info(f"Searching for documents matching '{args.search}' in application '{args.app}'")
            
            results = extractor.search_documents_by_pattern(args.search, args.max_results)
            
            if not results:
                print(f"❌ No documents found matching pattern '{args.search}'")
                return 1
            
            print(f"\n📋 Found {len(results)} documents matching '{args.search}':")
            print("="*80)
            
            for i, doc in enumerate(results, 1):
                print(f"{i}. {doc['file_name']}")
                print(f"   📁 Path: {doc['file_path']}")
                print(f"   📊 Type: {doc['document_type']}")
                print(f"   🖼️  Images: {doc['image_count']} ({'Yes' if doc['has_images'] else 'No'})")
                print(f"   📅 Timestamp: {doc['timestamp']}")
                print(f"   🎯 Score: {doc['score']:.4f}")
                print()
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
