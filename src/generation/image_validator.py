"""
Image Validator - Valida y corrige datos de imagen para procesamiento multimodal
Asegura que las imágenes cumplan con los requisitos de AWS Bedrock Claude
"""

import base64
import logging
from typing import Dict, Any, Optional, Tuple
import re
from PIL import Image
import io

logger = logging.getLogger(__name__)


class ImageValidator:
    """
    Validador de imágenes para procesamiento multimodal con Claude via Bedrock.
    
    Funcionalidades:
    - Validación de formato Base64
    - Detección automática de tipo de imagen
    - Redimensionamiento si excede límites
    - Limpieza de datos Base64
    """
    
    # Límites de AWS Bedrock para Claude
    MAX_IMAGE_SIZE_MB = 5  # 5MB máximo
    MAX_IMAGE_DIMENSION = 8000  # 8000px máximo por dimensión
    SUPPORTED_FORMATS = ['PNG', 'JPEG', 'JPG', 'GIF', 'WEBP']
    
    def __init__(self):
        """Inicializa el validador de imágenes"""
        self.stats = {
            'images_processed': 0,
            'images_validated': 0,
            'images_resized': 0,
            'images_rejected': 0,
            'format_corrections': 0
        }
        logger.debug("ImageValidator initialized")
    
    def validate_and_fix_image(self, image_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], str]:
        """
        Valida y corrige datos de imagen para procesamiento multimodal.
        
        Args:
            image_data: Diccionario con datos de imagen
            
        Returns:
            Tuple de (is_valid, corrected_data, error_message)
        """
        self.stats['images_processed'] += 1
        
        try:
            # Extraer datos base64
            base64_data = image_data.get('image_base64', '')
            if not base64_data:
                return False, {}, "No image_base64 data found"
            
            # Limpiar datos base64
            cleaned_base64 = self._clean_base64_data(base64_data)
            if not cleaned_base64:
                return False, {}, "Invalid base64 data after cleaning"
            
            # Validar formato base64
            try:
                image_bytes = base64.b64decode(cleaned_base64)
            except Exception as e:
                logger.error(f"Base64 decode error: {e}")
                return False, {}, f"Base64 decode failed: {str(e)}"
            
            # Validar tamaño
            if len(image_bytes) > self.MAX_IMAGE_SIZE_MB * 1024 * 1024:
                logger.warning(f"Image too large: {len(image_bytes)} bytes")
                # Intentar redimensionar
                resized_bytes, media_type = self._resize_image(image_bytes)
                if resized_bytes:
                    image_bytes = resized_bytes
                    cleaned_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    self.stats['images_resized'] += 1
                    logger.info(f"Image resized to {len(image_bytes)} bytes")
                else:
                    self.stats['images_rejected'] += 1
                    return False, {}, "Image too large and resize failed"
            
            # Detectar formato de imagen
            detected_format, media_type = self._detect_image_format(image_bytes)
            if not detected_format:
                self.stats['images_rejected'] += 1
                return False, {}, "Unsupported image format"
            
            # Corregir media_type si es necesario
            original_format = image_data.get('image_format', 'PNG')
            if original_format.upper() != detected_format:
                self.stats['format_corrections'] += 1
                logger.info(f"Format corrected: {original_format} -> {detected_format}")
            
            # Construir datos corregidos
            corrected_data = {
                'source_id': image_data.get('source_id', 'unknown'),
                'image_base64': cleaned_base64,
                'image_id': image_data.get('image_id', 'img_unknown'),
                'image_context': image_data.get('image_context', ''),
                'image_format': detected_format,
                'media_type': media_type,
                'size_bytes': len(image_bytes),
                'validated': True
            }
            
            self.stats['images_validated'] += 1
            logger.info(f"Image validated successfully: {detected_format}, {len(image_bytes)} bytes")
            
            return True, corrected_data, ""
            
        except Exception as e:
            logger.error(f"Image validation error: {e}")
            self.stats['images_rejected'] += 1
            return False, {}, f"Validation error: {str(e)}"
    
    def _clean_base64_data(self, base64_data: str) -> str:
        """
        Limpia datos base64 eliminando prefijos y caracteres no válidos.
        
        Args:
            base64_data: Datos base64 sin procesar
            
        Returns:
            Datos base64 limpios
        """
        if not base64_data:
            return ""
        
        # Eliminar prefijos de data URL si existen
        if base64_data.startswith('data:'):
            # Formato: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...
            if ',,' in base64_data:
                base64_data = base64_data.split(',,')[1]
            elif ',' in base64_data:
                base64_data = base64_data.split(',')[1]
        
        # Eliminar espacios en blanco y saltos de línea
        base64_data = re.sub(r'\s+', '', base64_data)
        
        # Validar que solo contenga caracteres base64 válidos
        if not re.match(r'^[A-Za-z0-9+/]*={0,2}$', base64_data):
            logger.warning("Base64 data contains invalid characters")
            # Intentar limpiar caracteres inválidos
            base64_data = re.sub(r'[^A-Za-z0-9+/=]', '', base64_data)
        
        return base64_data
    
    def _detect_image_format(self, image_bytes: bytes) -> Tuple[Optional[str], Optional[str]]:
        """
        Detecta el formato de imagen basándose en los bytes.
        
        Args:
            image_bytes: Bytes de la imagen
            
        Returns:
            Tuple de (format, media_type)
        """
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                format_name = img.format
                if format_name in self.SUPPORTED_FORMATS:
                    if format_name == 'JPEG':
                        return 'JPEG', 'image/jpeg'
                    elif format_name == 'PNG':
                        return 'PNG', 'image/png'
                    elif format_name == 'GIF':
                        return 'GIF', 'image/gif'
                    elif format_name == 'WEBP':
                        return 'WEBP', 'image/webp'
                    else:
                        return format_name, f'image/{format_name.lower()}'
                else:
                    logger.warning(f"Unsupported format detected: {format_name}")
                    return None, None
        except Exception as e:
            logger.error(f"Error detecting image format: {e}")
            return None, None
    
    def _resize_image(self, image_bytes: bytes) -> Tuple[Optional[bytes], Optional[str]]:
        """
        Redimensiona imagen si excede los límites.
        
        Args:
            image_bytes: Bytes de la imagen original
            
        Returns:
            Tuple de (resized_bytes, media_type)
        """
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                original_width, original_height = img.size
                
                # Calcular nuevo tamaño manteniendo aspect ratio
                if original_width > self.MAX_IMAGE_DIMENSION or original_height > self.MAX_IMAGE_DIMENSION:
                    ratio = min(
                        self.MAX_IMAGE_DIMENSION / original_width,
                        self.MAX_IMAGE_DIMENSION / original_height
                    )
                    new_width = int(original_width * ratio)
                    new_height = int(original_height * ratio)
                    
                    # Redimensionar
                    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    # Convertir a bytes
                    output_buffer = io.BytesIO()
                    format_name = img.format or 'PNG'
                    
                    if format_name == 'JPEG':
                        resized_img.save(output_buffer, format='JPEG', quality=85, optimize=True)
                        media_type = 'image/jpeg'
                    else:
                        resized_img.save(output_buffer, format='PNG', optimize=True)
                        media_type = 'image/png'
                    
                    resized_bytes = output_buffer.getvalue()
                    
                    logger.info(f"Image resized from {original_width}x{original_height} to {new_width}x{new_height}")
                    return resized_bytes, media_type
                
                # No necesita redimensionamiento
                return image_bytes, f'image/{img.format.lower()}'
                
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            return None, None
    
    def validate_image_list(self, images_data: list) -> Tuple[list, list]:
        """
        Valida una lista de imágenes.
        
        Args:
            images_data: Lista de datos de imagen
            
        Returns:
            Tuple de (valid_images, error_messages)
        """
        valid_images = []
        error_messages = []
        
        for i, image_data in enumerate(images_data):
            is_valid, corrected_data, error_msg = self.validate_and_fix_image(image_data)
            
            if is_valid:
                valid_images.append(corrected_data)
            else:
                error_messages.append(f"Image {i+1}: {error_msg}")
                logger.warning(f"Image {i+1} validation failed: {error_msg}")
        
        logger.info(f"Validated {len(valid_images)}/{len(images_data)} images successfully")
        return valid_images, error_messages
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del validador"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reinicia las estadísticas"""
        self.stats = {
            'images_processed': 0,
            'images_validated': 0,
            'images_resized': 0,
            'images_rejected': 0,
            'format_corrections': 0
        }
        logger.info("ImageValidator stats reset")
