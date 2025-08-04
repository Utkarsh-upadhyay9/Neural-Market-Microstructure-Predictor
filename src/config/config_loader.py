"""
Configuration loader with environment variable support
"""
import os
import yaml
from pathlib import Path

class ConfigLoader:
    def __init__(self, config_path="config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self):
        """Load configuration with environment variable substitution"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as file:
            config_content = file.read()
        
        config_content = self._substitute_env_vars(config_content)
        return yaml.safe_load(config_content)
    
    def _substitute_env_vars(self, content):
        """Replace ${VAR_NAME} with environment variables"""
        import re
        
        def replace_var(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) else ""
            return os.getenv(var_name, default_value)
        
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        return re.sub(pattern, replace_var, content)
    
    def get(self, key_path, default=None):
        """Get configuration value using dot notation"""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_api_key(self, service):
        """Get API key for a specific service"""
        return self.get(f'api.{service}.key')

config = ConfigLoader()
