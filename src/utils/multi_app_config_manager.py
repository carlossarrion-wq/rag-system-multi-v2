"""
Multi-Application Configuration Manager
Manages configuration for multiple RAG applications with separate indices, S3 buckets, and system prompts
"""

import yaml
import os
import json
from typing import Dict, Any, Optional, List
from loguru import logger


class MultiAppConfigManager:
    """
    Configuration manager for multi-application RAG system.
    
    Features:
    - Load application-specific configurations
    - Merge global and application-specific settings
    - Validate application configurations
    - Provide application-aware connection parameters
    """
    
    def __init__(self, config_path: str = "config/multi_app_config.yaml"):
        """
        Initialize the multi-application configuration manager.
        
        Args:
            config_path: Path to the multi-application configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.applications = self.config.get('applications', {})
        self.default_app = self.config.get('default_application', 'sap')
        
        logger.info(f"MultiAppConfigManager initialized: {len(self.applications)} applications ({', '.join(self.applications.keys())})")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load the multi-application configuration file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def get_available_applications(self) -> List[str]:
        """Get list of available application names."""
        return list(self.applications.keys())
    
    def validate_application(self, app_name: str) -> bool:
        """
        Validate if an application exists in the configuration.
        
        Args:
            app_name: Name of the application to validate
            
        Returns:
            True if application exists, False otherwise
        """
        return app_name in self.applications
    
    def get_application_config(self, app_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get complete configuration for a specific application.
        
        Args:
            app_name: Name of the application. If None, uses default application.
            
        Returns:
            Complete configuration dictionary for the application
            
        Raises:
            ValueError: If application doesn't exist
        """
        if app_name is None:
            app_name = self.default_app
        
        if not self.validate_application(app_name):
            available_apps = ', '.join(self.get_available_applications())
            raise ValueError(f"Application '{app_name}' not found. Available applications: {available_apps}")
        
        app_config = self.applications[app_name].copy()
        
        # Load system prompt from external file if specified
        system_prompt = self.get_system_prompt(app_name)
        
        # Merge global configurations with application-specific ones
        merged_config = {
            'aws': self.config.get('aws', {}),
            'bedrock': self.config.get('bedrock', {}),
            'opensearch': {
                **self.config.get('opensearch', {}),
                **app_config.get('opensearch', {})
            },
            'postgresql': self.config.get('postgresql', {}),
            'services': {
                'opensearch': {
                    **self.config.get('opensearch', {}),
                    **app_config.get('opensearch', {})
                },
                'postgresql': self.config.get('postgresql', {}),
                's3': app_config.get('s3', {})
            },
            'rag_system': app_config.get('rag_system', {}),
            'logging': self.config.get('logging', {}),
            'environment': self.config.get('environment', {}),
            'application': {
                'name': app_config.get('name', app_name),
                'description': app_config.get('description', ''),
                'system_prompt': system_prompt,
                'app_id': app_name
            }
        }
        
        return merged_config
    
    def get_opensearch_index_name(self, app_name: Optional[str] = None) -> str:
        """
        Get OpenSearch index name for a specific application.
        
        Args:
            app_name: Name of the application
            
        Returns:
            OpenSearch index name for the application
        """
        app_config = self.get_application_config(app_name)
        return app_config['services']['opensearch']['index_name']
    
    def get_s3_config(self, app_name: Optional[str] = None) -> Dict[str, str]:
        """
        Get S3 configuration for a specific application.
        
        Args:
            app_name: Name of the application
            
        Returns:
            S3 configuration dictionary
        """
        app_config = self.get_application_config(app_name)
        return app_config['services']['s3']
    
    def get_system_prompt(self, app_name: Optional[str] = None) -> str:
        """
        Get system prompt for a specific application.
        Loads from external file if system_prompt_file is specified, otherwise uses inline system_prompt.
        
        Args:
            app_name: Name of the application
            
        Returns:
            System prompt string for the application
        """
        if app_name is None:
            app_name = self.default_app
        
        if not self.validate_application(app_name):
            available_apps = ', '.join(self.get_available_applications())
            raise ValueError(f"Application '{app_name}' not found. Available applications: {available_apps}")
        
        app_config = self.applications[app_name]
        
        # Check if external system prompt file is specified
        system_prompt_file = app_config.get('system_prompt_file')
        if system_prompt_file:
            return self._load_system_prompt_from_file(system_prompt_file, app_name)
        
        # Fallback to inline system prompt
        inline_prompt = app_config.get('system_prompt', '')
        if inline_prompt:
            return inline_prompt
        
        # No system prompt found
        return ""
    
    def _load_system_prompt_from_file(self, file_path: str, app_name: str) -> str:
        """
        Load system prompt from external file.
        Supports both JSON and text formats.
        
        Args:
            file_path: Path to the system prompt file
            app_name: Name of the application (for logging)
            
        Returns:
            System prompt string
        """
        try:
            # Make path relative to config file directory if not absolute
            if not os.path.isabs(file_path):
                config_dir = os.path.dirname(self.config_path)
                file_path = os.path.join(config_dir, file_path)
            
            if not os.path.exists(file_path):
                logger.error(f"System prompt file not found: {file_path}")
                raise FileNotFoundError(f"System prompt file not found: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Determine file format and process accordingly
            if file_path.endswith('.json'):
                try:
                    # Try to parse as JSON
                    json_data = json.loads(content)
                    
                    # If it's a structured JSON prompt (like SAP), convert to string representation
                    if isinstance(json_data, dict):
                        return json.dumps(json_data, indent=2, ensure_ascii=False)
                    else:
                        # If it's a simple JSON string, return the string value
                        return str(json_data)
                        
                except json.JSONDecodeError:
                    # If JSON parsing fails, treat as plain text
                    return content
            else:
                # Plain text file
                return content
                
        except Exception as e:
            logger.error(f"Error loading system prompt from {file_path} for application {app_name}: {e}")
            raise
    
    def get_rag_config(self, app_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get RAG system configuration for a specific application.
        
        Args:
            app_name: Name of the application
            
        Returns:
            RAG configuration dictionary
        """
        app_config = self.get_application_config(app_name)
        return app_config['rag_system']
    
    def get_bedrock_config(self, app_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get AWS Bedrock configuration for a specific application.
        
        Args:
            app_name: Name of the application
            
        Returns:
            Bedrock configuration dictionary
        """
        app_config = self.get_application_config(app_name)
        return app_config['bedrock']
    
    def get_application_info(self, app_name: Optional[str] = None) -> Dict[str, str]:
        """
        Get basic information about an application.
        
        Args:
            app_name: Name of the application
            
        Returns:
            Dictionary with application name, description, and ID
        """
        app_config = self.get_application_config(app_name)
        return app_config['application']
    
    def list_applications_info(self) -> List[Dict[str, str]]:
        """
        Get information about all available applications.
        
        Returns:
            List of dictionaries with application information
        """
        apps_info = []
        for app_name in self.get_available_applications():
            app_config = self.applications[app_name]
            apps_info.append({
                'id': app_name,
                'name': app_config.get('name', app_name),
                'description': app_config.get('description', ''),
                'index_name': app_config.get('opensearch', {}).get('index_name', ''),
                's3_bucket': app_config.get('s3', {}).get('bucket', ''),
                's3_prefix': app_config.get('s3', {}).get('documents_prefix', '')
            })
        return apps_info
    
    def create_legacy_config(self, app_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a legacy-compatible configuration for backward compatibility.
        This allows existing components to work without modification.
        
        Args:
            app_name: Name of the application
            
        Returns:
            Legacy-compatible configuration dictionary
        """
        app_config = self.get_application_config(app_name)
        
        # Create legacy format that matches the original aws_config_production.yaml structure
        legacy_config = {
            'aws': app_config['aws'],
            'bedrock': app_config['bedrock'],
            'services': {
                'opensearch': {
                    'enabled': True,
                    'endpoint': app_config['opensearch']['endpoint'],
                    'index_name': app_config['services']['opensearch']['index_name'],
                    'use_ssl': app_config['opensearch']['use_ssl'],
                    'verify_certs': app_config['opensearch']['verify_certs'],
                    'connection_class': app_config['opensearch']['connection_class'],
                    'vpc_access': app_config['opensearch']['vpc_access'],
                    'timeout': app_config['opensearch']['timeout']
                },
                'postgresql': {
                    **app_config['postgresql'],
                    'enabled': app_config['postgresql'].get('enabled', True)
                },
                's3': {
                    **app_config['services']['s3'],
                    'enabled': True
                }
            },
            'rag_system': app_config['rag_system'],
            'logging': app_config['logging'],
            'environment': app_config['environment']
        }
        
        return legacy_config
    
    def get_app_name_from_index(self, index_name: str) -> Optional[str]:
        """
        Get application name from OpenSearch index name.
        
        Args:
            index_name: OpenSearch index name
            
        Returns:
            Application name if found, None otherwise
        """
        for app_name, app_config in self.applications.items():
            if app_config.get('opensearch', {}).get('index_name') == index_name:
                return app_name
        return None
    
    def validate_configuration(self, app_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate configuration for an application.
        
        Args:
            app_name: Name of the application to validate
            
        Returns:
            Dictionary with validation results
        """
        if app_name is None:
            app_name = self.default_app
        
        validation_result = {
            'application': app_name,
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            app_config = self.get_application_config(app_name)
            
            # Check required fields
            required_fields = [
                ('services.opensearch.index_name', 'OpenSearch index name'),
                ('services.s3.bucket', 'S3 bucket'),
                ('application.system_prompt', 'System prompt')
            ]
            
            for field_path, field_name in required_fields:
                if not self._get_nested_value(app_config, field_path):
                    validation_result['errors'].append(f"Missing {field_name}")
                    validation_result['valid'] = False
            
            # Check optional but recommended fields
            recommended_fields = [
                ('application.description', 'Application description'),
                ('rag_system.chunking.chunk_size', 'Chunk size configuration')
            ]
            
            for field_path, field_name in recommended_fields:
                if not self._get_nested_value(app_config, field_path):
                    validation_result['warnings'].append(f"Missing {field_name}")
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Configuration error: {str(e)}")
        
        return validation_result
    
    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """
        Get nested value from configuration using dot notation.
        
        Args:
            config: Configuration dictionary
            path: Dot-separated path (e.g., 'services.opensearch.index_name')
            
        Returns:
            Value at the specified path, or None if not found
        """
        keys = path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value


# Convenience function for backward compatibility
def load_application_config(app_name: Optional[str] = None, 
                          config_path: str = "config/multi_app_config.yaml") -> Dict[str, Any]:
    """
    Load configuration for a specific application in legacy format.
    
    Args:
        app_name: Name of the application
        config_path: Path to the multi-application configuration file
        
    Returns:
        Legacy-compatible configuration dictionary
    """
    manager = MultiAppConfigManager(config_path)
    return manager.create_legacy_config(app_name)
