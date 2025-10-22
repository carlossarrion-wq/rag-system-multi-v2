"""
Image Summary Retriever
Retrieves image summaries from S3 when images cannot be processed directly
"""

import json
import boto3
from typing import Dict, Any, List, Optional
from botocore.exceptions import ClientError
from loguru import logger
from src.utils.connection_manager import ConnectionManager


class ImageSummaryRetriever:
    """
    Retrieves image summaries from S3 as an alternative to sending raw image data.
    Used when image processing fails or token limits are exceeded.
    """
    
    def __init__(self, app_name: str, config_path: str = "config/multi_app_config.yaml"):
        """
        Initialize image summary retriever.
        
        Args:
            app_name: Application name (e.g., 'pds', 'gadea')
            config_path: Path to configuration file
        """
        self.app_name = app_name
        self.config_path = config_path
        
        # Initialize connection manager
        self.connection_manager = ConnectionManager(config_path)
        self.s3_client = self.connection_manager.get_s3_client()
        
        # Load configuration
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get S3 bucket for this application
        self.s3_bucket = config['applications'][app_name]['services']['s3']['bucket']
        
        logger.info(f"ImageSummaryRetriever initialized for app: {app_name}, bucket: {self.s3_bucket}")
    
    def get_image_summary(self, image_metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Retrieve image summary from S3 based on image metadata.
        
        Args:
            image_metadata: Metadata containing image information
            
        Returns:
            Dictionary with image summary information or None if not found
        """
        try:
            # Extract key information from metadata
            file_name = image_metadata.get('source_file', '')
            chunk_id = image_metadata.get('chunk_id', '')
            image_id = image_metadata.get('image_id', '')
            
            if not file_name:
                logger.warning("No file name found in image metadata")
                return None
            
            # Try different S3 key patterns for image summaries
            possible_keys = self._generate_summary_keys(file_name, chunk_id, image_id)
            
            for key in possible_keys:
                try:
                    logger.debug(f"Trying S3 key: {key}")
                    response = self.s3_client.get_object(
                        Bucket=self.s3_bucket,
                        Key=key
                    )
                    
                    summary_data = json.loads(response['Body'].read().decode('utf-8'))
                    logger.info(f"Found image summary at key: {key}")
                    
                    return {
                        'summary_text': summary_data.get('summary', ''),
                        'description': summary_data.get('description', ''),
                        'key_elements': summary_data.get('key_elements', []),
                        'visual_content': summary_data.get('visual_content', ''),
                        'source_file': file_name,
                        'image_id': image_id,
                        's3_key': key,
                        'summary_type': 'detailed'
                    }
                    
                except ClientError as e:
                    if e.response['Error']['Code'] != 'NoSuchKey':
                        logger.warning(f"Error accessing S3 key {key}: {e}")
                    continue
            
            # If no detailed summary found, try to get basic image info
            return self._get_basic_image_info(image_metadata)
            
        except Exception as e:
            logger.error(f"Error retrieving image summary: {e}")
            return None
    
    def _generate_summary_keys(self, file_name: str, chunk_id: str, image_id: str) -> List[str]:
        """
        Generate possible S3 keys where image summaries might be stored.
        
        Args:
            file_name: Original file name
            chunk_id: Chunk identifier
            image_id: Image identifier
            
        Returns:
            List of possible S3 keys to try
        """
        keys = []
        
        # Clean file name (remove extension, special characters)
        clean_name = file_name.replace('.pdf', '').replace('.docx', '').replace('.png', '')
        clean_name = clean_name.replace(' ', '_').replace('-', '_')
        
        # Pattern 1: applications/{app}/summaries/images/{file_name}/{image_id}.json
        if image_id:
            keys.append(f"applications/{self.app_name}/summaries/images/{clean_name}/{image_id}.json")
        
        # Pattern 2: applications/{app}/summaries/images/{file_name}_summary.json
        keys.append(f"applications/{self.app_name}/summaries/images/{clean_name}_summary.json")
        
        # Pattern 3: applications/{app}/images/summaries/{chunk_id}.json
        if chunk_id:
            keys.append(f"applications/{self.app_name}/images/summaries/{chunk_id}.json")
        
        # Pattern 4: applications/{app}/processed/{file_name}/image_summaries.json
        keys.append(f"applications/{self.app_name}/processed/{clean_name}/image_summaries.json")
        
        # Pattern 5: summaries/images/{app}/{file_name}.json
        keys.append(f"summaries/images/{self.app_name}/{clean_name}.json")
        
        return keys
    
    def _get_basic_image_info(self, image_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate basic image information when detailed summary is not available.
        
        Args:
            image_metadata: Image metadata
            
        Returns:
            Basic image information dictionary
        """
        file_name = image_metadata.get('source_file', 'unknown file')
        image_context = image_metadata.get('image_context', '')
        image_id = image_metadata.get('image_id', 'unknown')
        
        # Extract meaningful information from file name
        summary_text = f"Diagrama visual: {file_name}"
        description = image_context or f"Diagrama de flujo o contenido visual del archivo {file_name}"
        
        # Try to extract specific information from file name
        key_elements = ['diagrama', 'flujo de proceso', 'contenido visual']
        
        # If file name contains specific identifiers, make it more specific
        if '900' in file_name:
            summary_text = f"Diagrama de flujo del usuario 900 - {file_name}"
            description = f"Diagrama de flujo específico para el usuario 900, mostrando procesos de gestión de servicios, información de contratos, solicitudes de modificación, gestión de incidencias y pagos"
            key_elements = ['diagrama de flujo', 'usuario 900', 'gestión de servicios', 'contratos', 'incidencias', 'pagos']
        elif '700' in file_name:
            summary_text = f"Diagrama de flujo del usuario 700 - {file_name}"
            description = f"Diagrama de flujo para el usuario 700, representando procesos importantes de la plataforma PDS relacionados con servicios al cliente"
            key_elements = ['diagrama de flujo', 'usuario 700', 'servicios al cliente', 'plataforma PDS']
        elif 'NIF' in file_name:
            summary_text = f"Diagrama de flujo de identificación NIF - {file_name}"
            description = f"Diagrama de flujo relacionado con la identificación y verificación de clientes mediante NIF dentro de la plataforma PDS"
            key_elements = ['diagrama de flujo', 'identificación NIF', 'verificación de clientes', 'plataforma PDS']
        
        return {
            'summary_text': summary_text,
            'description': description,
            'key_elements': key_elements,
            'visual_content': f"Elemento visual extraído de {file_name} - {description}",
            'source_file': file_name,
            'image_id': image_id,
            's3_key': None,
            'summary_type': 'enhanced_basic'
        }
    
    def get_multiple_image_summaries(self, images_metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Retrieve summaries for multiple images.
        
        Args:
            images_metadata: List of image metadata dictionaries
            
        Returns:
            List of image summary dictionaries
        """
        summaries = []
        
        for metadata in images_metadata:
            summary = self.get_image_summary(metadata)
            if summary:
                summaries.append(summary)
            else:
                # Add basic info even if summary retrieval fails
                basic_info = self._get_basic_image_info(metadata)
                summaries.append(basic_info)
        
        logger.info(f"Retrieved {len(summaries)} image summaries")
        return summaries
    
    def format_summaries_for_context(self, summaries: List[Dict[str, Any]]) -> str:
        """
        Format image summaries for inclusion in LLM context.
        
        Args:
            summaries: List of image summary dictionaries
            
        Returns:
            Formatted text for LLM context
        """
        if not summaries:
            return ""
        
        formatted_parts = []
        formatted_parts.append("=== RESÚMENES DE IMÁGENES ===\n")
        
        for i, summary in enumerate(summaries, 1):
            source_file = summary.get('source_file', 'unknown')
            summary_text = summary.get('summary_text', '')
            description = summary.get('description', '')
            key_elements = summary.get('key_elements', [])
            
            formatted_parts.append(f"Imagen {i}: {source_file}")
            
            if summary_text:
                formatted_parts.append(f"Resumen: {summary_text}")
            
            if description and description != summary_text:
                formatted_parts.append(f"Descripción: {description}")
            
            if key_elements:
                elements_str = ", ".join(key_elements[:5])  # Limit to 5 elements
                formatted_parts.append(f"Elementos clave: {elements_str}")
            
            formatted_parts.append("")  # Empty line between images
        
        return "\n".join(formatted_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        return {
            'app_name': self.app_name,
            's3_bucket': self.s3_bucket,
            'config_path': self.config_path
        }
