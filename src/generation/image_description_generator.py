"""
Image Description Generator
Generates detailed descriptions of images during ingestion using LLM vision capabilities
"""

import json
import base64
from typing import Dict, Any, Optional
from loguru import logger
from PIL import Image
from io import BytesIO
import boto3
from botocore.exceptions import ClientError


class ImageDescriptionGenerator:
    """
    Generates detailed descriptions of images during document ingestion.
    Uses Claude's vision capabilities to create comprehensive textual descriptions
    that will be stored and used instead of sending raw images to the LLM during queries.
    """
    
    def __init__(self, bedrock_client, model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"):
        """
        Initialize the image description generator.
        
        Args:
            bedrock_client: AWS Bedrock client instance
            model_id: Claude model ID with vision capabilities
        """
        self.bedrock_client = bedrock_client
        self.model_id = model_id
        
        logger.info(f"ImageDescriptionGenerator initialized with model: {model_id}")
    
    def generate_detailed_description(self, image_data: Dict[str, Any], document_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a detailed description of an image using Claude's vision capabilities.
        
        Args:
            image_data: Dictionary containing image information (base64 data, format, etc.)
            document_context: Optional context about the document containing the image
            
        Returns:
            Dictionary with detailed description and metadata
        """
        try:
            # Extract image information
            image_base64 = image_data.get('data', '')
            image_format = image_data.get('format', 'png').lower()
            source_file = document_context.get('file_name', 'unknown') if document_context else 'unknown'
            
            if not image_base64:
                logger.warning("No image data provided for description generation")
                return self._create_fallback_description(image_data, document_context)
            
            # Validate and get image dimensions
            try:
                image_bytes = base64.b64decode(image_base64)
                pil_image = Image.open(BytesIO(image_bytes))
                width, height = pil_image.size
                image_info = f"{width}x{height} píxeles"
            except Exception as e:
                logger.warning(f"Could not extract image dimensions: {e}")
                image_info = "dimensiones desconocidas"
            
            # Determine media type
            media_type_map = {
                'png': 'image/png',
                'jpg': 'image/jpeg',
                'jpeg': 'image/jpeg',
                'gif': 'image/gif',
                'webp': 'image/webp'
            }
            media_type = media_type_map.get(image_format, 'image/png')
            
            # Create detailed prompt for image description
            system_prompt = """Eres un experto analista de imágenes técnicas. Tu tarea es generar descripciones extremadamente detalladas y precisas de imágenes que contienen información técnica, diagramas, interfaces, flujos de proceso, arquitecturas de sistemas, etc.

INSTRUCCIONES CRÍTICAS:
1. Describe TODOS los elementos visibles en la imagen con máximo detalle
2. Identifica y explica TODOS los componentes, conexiones, flujos, textos, etiquetas
3. Describe la estructura, organización y relaciones entre elementos
4. Incluye TODOS los textos legibles, nombres, códigos, números
5. Explica el propósito y función de cada elemento identificado
6. Describe colores, formas, posiciones relativas
7. Si es un diagrama de flujo, describe CADA paso y conexión
8. Si es una interfaz, describe CADA elemento de UI
9. Si es una arquitectura, describe CADA componente y conexión

La descripción debe ser tan detallada que alguien pueda entender completamente el contenido sin ver la imagen."""
            
            # Create context-aware user prompt
            context_info = ""
            if document_context:
                context_info = f"""
CONTEXTO DEL DOCUMENTO:
- Archivo: {source_file}
- Tipo: {document_context.get('file_extension', 'desconocido')}
- Sistema: {document_context.get('application_context', 'desconocido')}
"""
            
            user_prompt = f"""Analiza esta imagen técnica y proporciona una descripción extremadamente detallada.

{context_info}

INFORMACIÓN DE LA IMAGEN:
- Formato: {image_format.upper()}
- Dimensiones: {image_info}
- Fuente: {source_file}

TAREA: Genera una descripción completa y detallada que incluya:

1. **DESCRIPCIÓN GENERAL**: Qué tipo de imagen es (diagrama, interfaz, arquitectura, etc.)

2. **ELEMENTOS PRINCIPALES**: Lista y describe todos los componentes principales visibles

3. **SISTEMAS Y COMPONENTES**: Identifica todos los sistemas, plataformas, servicios mencionados

4. **FLUJOS E INTEGRACIONES**: Describe todas las conexiones, flujos de datos, integraciones

5. **TEXTOS Y ETIQUETAS**: Transcribe TODOS los textos visibles, nombres, códigos, URLs

6. **DETALLES TÉCNICOS**: Operaciones, métodos, protocolos, configuraciones mencionadas

7. **ESTRUCTURA Y ORGANIZACIÓN**: Cómo están organizados los elementos (izquierda a derecha, arriba abajo, etc.)

8. **PROPÓSITO Y FUNCIÓN**: Explica qué representa la imagen y para qué sirve

Sé extremadamente detallado y preciso. Esta descripción reemplazará completamente a la imagen en futuras consultas."""
            
            # Build multimodal request
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4000,  # Increased for detailed descriptions
                "temperature": 0.1,  # Low temperature for consistent, factual descriptions
                "system": system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": user_prompt
                            }
                        ]
                    }
                ]
            }
            
            # Call Claude with vision
            logger.info(f"Generating detailed description for image from {source_file}")
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType='application/json',
                accept='application/json'
            )
            
            response_body = json.loads(response['body'].read())
            detailed_description = response_body['content'][0]['text']
            
            # Extract usage information
            usage = response_body.get('usage', {})
            
            # Create comprehensive result
            result = {
                'detailed_description': detailed_description,
                'image_info': {
                    'format': image_format,
                    'dimensions': image_info,
                    'media_type': media_type,
                    'source_file': source_file
                },
                'generation_metadata': {
                    'model_used': self.model_id,
                    'tokens_used': usage.get('output_tokens', 0),
                    'input_tokens': usage.get('input_tokens', 0),
                    'temperature': 0.1,
                    'generated_at': self._get_timestamp(),
                    'description_length': len(detailed_description),
                    'description_quality': 'detailed_vision_analysis'
                },
                'context': document_context or {},
                'success': True
            }
            
            logger.info(f"Generated detailed description for {source_file}: {len(detailed_description)} characters, {usage.get('output_tokens', 0)} tokens")
            return result
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            logger.error(f"Bedrock API error generating image description: {error_code} - {error_message}")
            return self._create_error_description(image_data, document_context, f"API Error: {error_message}")
            
        except Exception as e:
            logger.error(f"Error generating detailed image description: {e}")
            return self._create_error_description(image_data, document_context, str(e))
    
    def _create_fallback_description(self, image_data: Dict[str, Any], document_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a fallback description when image processing fails.
        
        Args:
            image_data: Image data dictionary
            document_context: Document context
            
        Returns:
            Fallback description dictionary
        """
        source_file = document_context.get('file_name', 'unknown') if document_context else 'unknown'
        image_format = image_data.get('format', 'unknown')
        
        fallback_description = f"""Esta imagen del archivo "{source_file}" contiene información visual importante que no pudo ser procesada automáticamente.

INFORMACIÓN DISPONIBLE:
- Archivo fuente: {source_file}
- Formato de imagen: {image_format.upper()}
- Tipo de contenido: Imagen técnica o diagrama

CONTENIDO PROBABLE:
Basándose en el contexto del documento, esta imagen probablemente contiene:
- Diagramas de flujo o procesos
- Arquitecturas de sistemas
- Interfaces de usuario
- Esquemas técnicos
- Capturas de pantalla
- Gráficos explicativos

NOTA: Para obtener información específica sobre el contenido visual de esta imagen, se recomienda consultar el documento original o solicitar una descripción manual."""
        
        return {
            'detailed_description': fallback_description,
            'image_info': {
                'format': image_format,
                'source_file': source_file,
                'dimensions': 'unknown'
            },
            'generation_metadata': {
                'model_used': 'fallback',
                'description_quality': 'fallback_generic',
                'generated_at': self._get_timestamp(),
                'description_length': len(fallback_description)
            },
            'context': document_context or {},
            'success': False,
            'fallback_reason': 'no_image_data'
        }
    
    def _create_error_description(self, image_data: Dict[str, Any], document_context: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        """
        Create an error description when image processing fails.
        
        Args:
            image_data: Image data dictionary
            document_context: Document context
            error_message: Error message
            
        Returns:
            Error description dictionary
        """
        source_file = document_context.get('file_name', 'unknown') if document_context else 'unknown'
        image_format = image_data.get('format', 'unknown')
        
        error_description = f"""Error al procesar la imagen del archivo "{source_file}".

INFORMACIÓN DE LA IMAGEN:
- Archivo fuente: {source_file}
- Formato: {image_format.upper()}
- Error: {error_message}

DESCRIPCIÓN GENÉRICA:
Esta imagen contiene información visual que no pudo ser analizada automáticamente debido a un error técnico. La imagen probablemente contiene diagramas, esquemas, interfaces o contenido gráfico relevante para el sistema.

Para obtener información específica sobre esta imagen, consulte el documento original."""
        
        return {
            'detailed_description': error_description,
            'image_info': {
                'format': image_format,
                'source_file': source_file,
                'dimensions': 'unknown'
            },
            'generation_metadata': {
                'model_used': 'error_fallback',
                'description_quality': 'error_fallback',
                'generated_at': self._get_timestamp(),
                'description_length': len(error_description),
                'error': error_message
            },
            'context': document_context or {},
            'success': False,
            'error': error_message
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def process_multiple_images(self, images_data: list, document_context: Dict[str, Any] = None) -> list:
        """
        Process multiple images and generate descriptions for each.
        
        Args:
            images_data: List of image data dictionaries
            document_context: Document context
            
        Returns:
            List of description results
        """
        results = []
        
        for i, image_data in enumerate(images_data):
            logger.info(f"Processing image {i+1}/{len(images_data)} from {document_context.get('file_name', 'unknown') if document_context else 'unknown'}")
            
            # Add image index to context
            image_context = document_context.copy() if document_context else {}
            image_context['image_index'] = i + 1
            image_context['total_images'] = len(images_data)
            
            result = self.generate_detailed_description(image_data, image_context)
            results.append(result)
        
        successful_descriptions = sum(1 for r in results if r.get('success', False))
        logger.info(f"Generated descriptions for {successful_descriptions}/{len(images_data)} images")
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generator statistics."""
        return {
            'model_id': self.model_id,
            'description_generator': 'image_description_generator',
            'version': '1.0'
        }
