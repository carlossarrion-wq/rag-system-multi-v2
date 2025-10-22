"""
ConfigLoader - Cargador de configuración para diferentes entornos
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
from loguru import logger


class ConfigLoader:
    """Cargador de configuración para diferentes entornos"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Carga configuración desde un archivo específico"""
        
        config_file = self.base_path / config_path
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def load_aws_config(self) -> Dict[str, Any]:
        """Carga configuración para AWS/production"""
        
        # Intentar cargar config específico de EC2 primero
        config_files = [
            "config/aws_config_ec2.yaml",
            "config/aws_config_production.yaml", 
            "config/aws_config.yaml"
        ]
        
        for config_file in config_files:
            config_path = self.base_path / config_file
            if config_path.exists():
                logger.debug(f"Loading AWS config from: {config_file}")
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
        
        raise FileNotFoundError(f"No AWS configuration file found. Tried: {config_files}")
    
    def load_local_config(self) -> Dict[str, Any]:
        """Carga configuración para entorno local"""
        
        return self.load_config("config/model_config.yaml")
    
    def get_environment(self) -> str:
        """Detecta el entorno actual"""
        
        return os.getenv('RAG_ENV', 'local')
