"""
YMERA AI Platform - Configuration Loader
Loads and validates configuration from environment variables and YAML files
"""

import os
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads and manages system configuration"""
    
    def __init__(self, env_file: str = ".env", config_file: str = "config.yaml"):
        """
        Initialize configuration loader
        
        Args:
            env_file: Path to .env file
            config_file: Path to YAML config file
        """
        self.env_file = env_file
        self.config_file = config_file
        self.config: Dict[str, Any] = {}
        
        # Load environment variables
        self._load_env()
        
        # Load YAML configuration
        self._load_yaml()
        
        # Validate configuration
        self._validate()
    
    def _load_env(self):
        """Load environment variables from .env file"""
        if os.path.exists(self.env_file):
            load_dotenv(self.env_file)
            logger.info(f"Loaded environment variables from {self.env_file}")
        else:
            logger.warning(f".env file not found at {self.env_file}. Using system environment variables only.")
    
    def _load_yaml(self):
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        
        with open(self.config_file, 'r') as f:
            raw_config = yaml.safe_load(f)
        
        # Substitute environment variables
        self.config = self._substitute_env_vars(raw_config)
        logger.info(f"Loaded configuration from {self.config_file}")
    
    def _substitute_env_vars(self, obj: Any) -> Any:
        """
        Recursively substitute environment variables in configuration
        Supports ${VAR_NAME} and ${VAR_NAME:default_value} syntax
        """
        if isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            # Check if string contains environment variable
            if obj.startswith("${") and obj.endswith("}"):
                var_spec = obj[2:-1]  # Remove ${ and }
                
                # Check for default value
                if ":" in var_spec:
                    var_name, default = var_spec.split(":", 1)
                    return os.getenv(var_name, default)
                else:
                    value = os.getenv(var_spec)
                    if value is None:
                        logger.warning(f"Environment variable {var_spec} not found")
                        return ""
                    return value
            return obj
        else:
            return obj
    
    def _validate(self):
        """Validate essential configuration"""
        # Check for at least one enabled AI provider
        providers = self.config.get("ai_providers", {})
        enabled_providers = [
            name for name, cfg in providers.items() 
            if cfg.get("enabled", False)
        ]
        
        if not enabled_providers:
            logger.warning("No AI providers are enabled in configuration")
        else:
            logger.info(f"Enabled AI providers: {', '.join(enabled_providers)}")
        
        # Validate API keys for enabled providers
        self._validate_provider_keys()
    
    def _validate_provider_keys(self):
        """Validate that enabled providers have API keys"""
        providers = self.config.get("ai_providers", {})
        
        for name, cfg in providers.items():
            if not cfg.get("enabled", False):
                continue
            
            # Check for API key(s)
            if name == "gemini":
                keys = cfg.get("api_keys", [])
                if not any(k for k in keys if k):
                    logger.warning(f"Gemini provider enabled but no API keys configured")
            elif name in ["anthropic", "replicate"]:
                # Optional providers
                continue
            else:
                key_field = "api_key" if "api_key" in cfg else "api_token"
                if not cfg.get(key_field):
                    logger.warning(f"{name.title()} provider enabled but no API key configured")
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated path
        
        Args:
            path: Dot-separated path (e.g., 'ai_providers.gemini.enabled')
            default: Default value if path not found
        
        Returns:
            Configuration value or default
        """
        keys = path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get configuration for specific AI provider"""
        return self.get(f"ai_providers.{provider}", {})
    
    def get_enabled_providers(self) -> List[str]:
        """Get list of enabled AI providers"""
        providers = self.config.get("ai_providers", {})
        return [
            name for name, cfg in providers.items()
            if cfg.get("enabled", False)
        ]
    
    def get_routing_config(self) -> Dict[str, Any]:
        """Get routing configuration"""
        return self.get("routing", {})
    
    def get_caching_config(self) -> Dict[str, Any]:
        """Get caching configuration"""
        return self.get("caching", {})
    
    def get_langchain_config(self) -> Dict[str, Any]:
        """Get LangChain configuration"""
        return self.get("langchain", {})
    
    def get_mcp_config(self) -> Dict[str, Any]:
        """Get MCP tools configuration"""
        return self.get("mcp_tools", {})
    
    def get_ml_config(self) -> Dict[str, Any]:
        """Get ML/Learning configuration"""
        return self.get("ml_learning", {})
    
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.get("system.environment", "development") == "development"
    
    def is_debug(self) -> bool:
        """Check if debug mode is enabled"""
        return self.get("system.debug", True)
    
    def reload(self):
        """Reload configuration from files"""
        self._load_env()
        self._load_yaml()
        self._validate()
        logger.info("Configuration reloaded")


# Global configuration instance
_config: Optional[ConfigLoader] = None


def get_config() -> ConfigLoader:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = ConfigLoader()
    return _config


def reload_config():
    """Reload global configuration"""
    global _config
    if _config is not None:
        _config.reload()
    else:
        _config = ConfigLoader()


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load configuration
    config = get_config()
    
    # Print enabled providers
    print("\nEnabled AI Providers:")
    for provider in config.get_enabled_providers():
        print(f"  - {provider}")
    
    # Print routing strategy
    routing = config.get_routing_config()
    print(f"\nRouting Strategy: {routing.get('strategy')}")
    
    # Print caching configuration
    caching = config.get_caching_config()
    print(f"\nCaching Enabled: {caching.get('enabled')}")
    print(f"Caching Strategy: {caching.get('strategy')}")
    
    # Print LangChain status
    langchain = config.get_langchain_config()
    print(f"\nLangChain Enabled: {langchain.get('enabled')}")
    print(f"RAG Enabled: {langchain.get('enabled')}")
    print(f"Agents Enabled: {langchain.get('agents', {}).get('enabled')}")
