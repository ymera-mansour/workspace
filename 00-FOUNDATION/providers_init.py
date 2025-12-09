"""
YMERA AI Platform - AI Providers Initialization
Initialize and manage all AI provider connections
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ProviderStatus(Enum):
    """Provider connection status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class ProviderInfo:
    """Information about an AI provider"""
    name: str
    enabled: bool
    status: ProviderStatus
    models: List[str]
    api_key_configured: bool
    rate_limits: Dict[str, int]
    error_message: Optional[str] = None


class AIProvidersManager:
    """Manages all AI provider connections"""
    
    def __init__(self, config):
        """
        Initialize AI providers manager
        
        Args:
            config: ConfigLoader instance
        """
        self.config = config
        self.providers: Dict[str, Any] = {}
        self.provider_info: Dict[str, ProviderInfo] = {}
        
        # Initialize all enabled providers
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all enabled AI providers"""
        enabled_providers = self.config.get_enabled_providers()
        
        for provider_name in enabled_providers:
            try:
                self._init_provider(provider_name)
            except Exception as e:
                logger.error(f"Failed to initialize {provider_name}: {str(e)}")
                self.provider_info[provider_name] = ProviderInfo(
                    name=provider_name,
                    enabled=True,
                    status=ProviderStatus.ERROR,
                    models=[],
                    api_key_configured=False,
                    rate_limits={},
                    error_message=str(e)
                )
    
    def _init_provider(self, provider_name: str):
        """Initialize a specific provider"""
        provider_config = self.config.get_provider_config(provider_name)
        
        if provider_name == "gemini":
            self._init_gemini(provider_config)
        elif provider_name == "mistral":
            self._init_mistral(provider_config)
        elif provider_name == "groq":
            self._init_groq(provider_config)
        elif provider_name == "openrouter":
            self._init_openrouter(provider_config)
        elif provider_name == "huggingface":
            self._init_huggingface(provider_config)
        elif provider_name == "cohere":
            self._init_cohere(provider_config)
        elif provider_name == "together":
            self._init_together(provider_config)
        elif provider_name == "anthropic":
            self._init_anthropic(provider_config)
        elif provider_name == "replicate":
            self._init_replicate(provider_config)
        else:
            logger.warning(f"Unknown provider: {provider_name}")
    
    def _init_gemini(self, config: Dict):
        """Initialize Gemini provider"""
        try:
            import google.generativeai as genai
            
            # Get API keys (supports multiple for rotation)
            api_keys = [k for k in config.get("api_keys", []) if k]
            if not api_keys:
                raise ValueError("No API keys configured for Gemini")
            
            # Configure with first key (rotation handled by routing layer)
            genai.configure(api_key=api_keys[0])
            
            # Get model names
            models = [m["name"] for m in config.get("models", [])]
            
            self.providers["gemini"] = {
                "client": genai,
                "api_keys": api_keys,
                "config": config
            }
            
            self.provider_info["gemini"] = ProviderInfo(
                name="gemini",
                enabled=True,
                status=ProviderStatus.ACTIVE,
                models=models,
                api_key_configured=True,
                rate_limits=config.get("rate_limits", {})
            )
            
            logger.info(f"Gemini initialized with {len(api_keys)} API key(s) and {len(models)} models")
            
        except ImportError:
            logger.error("google-generativeai not installed. Run: pip install google-generativeai")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {str(e)}")
            raise
    
    def _init_mistral(self, config: Dict):
        """Initialize Mistral provider"""
        try:
            # Mistral client would be initialized here
            # from mistralai.client import MistralClient
            
            api_key = config.get("api_key")
            if not api_key:
                raise ValueError("No API key configured for Mistral")
            
            models = [m["name"] for m in config.get("models", [])]
            
            self.providers["mistral"] = {
                "api_key": api_key,
                "config": config
            }
            
            self.provider_info["mistral"] = ProviderInfo(
                name="mistral",
                enabled=True,
                status=ProviderStatus.ACTIVE,
                models=models,
                api_key_configured=True,
                rate_limits=config.get("rate_limits", {})
            )
            
            logger.info(f"Mistral initialized with {len(models)} models")
            
        except Exception as e:
            logger.error(f"Failed to initialize Mistral: {str(e)}")
            raise
    
    def _init_groq(self, config: Dict):
        """Initialize Groq provider"""
        try:
            api_key = config.get("api_key")
            if not api_key:
                raise ValueError("No API key configured for Groq")
            
            models = [m["name"] for m in config.get("models", [])]
            
            self.providers["groq"] = {
                "api_key": api_key,
                "base_url": config.get("base_url"),
                "config": config
            }
            
            self.provider_info["groq"] = ProviderInfo(
                name="groq",
                enabled=True,
                status=ProviderStatus.ACTIVE,
                models=models,
                api_key_configured=True,
                rate_limits=config.get("rate_limits", {})
            )
            
            logger.info(f"Groq initialized with {len(models)} models")
            
        except Exception as e:
            logger.error(f"Failed to initialize Groq: {str(e)}")
            raise
    
    def _init_openrouter(self, config: Dict):
        """Initialize OpenRouter provider"""
        try:
            api_key = config.get("api_key")
            if not api_key:
                raise ValueError("No API key configured for OpenRouter")
            
            models = [m["name"] for m in config.get("models", [])]
            
            self.providers["openrouter"] = {
                "api_key": api_key,
                "base_url": config.get("base_url"),
                "config": config
            }
            
            self.provider_info["openrouter"] = ProviderInfo(
                name="openrouter",
                enabled=True,
                status=ProviderStatus.ACTIVE,
                models=models,
                api_key_configured=True,
                rate_limits={}
            )
            
            logger.info(f"OpenRouter initialized with {len(models)} models")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter: {str(e)}")
            raise
    
    def _init_huggingface(self, config: Dict):
        """Initialize HuggingFace provider"""
        try:
            api_key = config.get("api_key")
            if not api_key:
                raise ValueError("No API key configured for HuggingFace")
            
            models = [m["name"] for m in config.get("models", [])]
            
            self.providers["huggingface"] = {
                "api_key": api_key,
                "base_url": config.get("base_url"),
                "config": config
            }
            
            self.provider_info["huggingface"] = ProviderInfo(
                name="huggingface",
                enabled=True,
                status=ProviderStatus.ACTIVE,
                models=models,
                api_key_configured=True,
                rate_limits=config.get("rate_limits", {})
            )
            
            logger.info(f"HuggingFace initialized with {len(models)} models")
            
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace: {str(e)}")
            raise
    
    def _init_cohere(self, config: Dict):
        """Initialize Cohere provider"""
        try:
            api_key = config.get("api_key")
            if not api_key:
                raise ValueError("No API key configured for Cohere")
            
            models = [m["name"] for m in config.get("models", [])]
            
            self.providers["cohere"] = {
                "api_key": api_key,
                "base_url": config.get("base_url"),
                "config": config
            }
            
            self.provider_info["cohere"] = ProviderInfo(
                name="cohere",
                enabled=True,
                status=ProviderStatus.ACTIVE,
                models=models,
                api_key_configured=True,
                rate_limits=config.get("rate_limits", {})
            )
            
            logger.info(f"Cohere initialized with {len(models)} models")
            
        except Exception as e:
            logger.error(f"Failed to initialize Cohere: {str(e)}")
            raise
    
    def _init_together(self, config: Dict):
        """Initialize Together AI provider"""
        try:
            api_key = config.get("api_key")
            if not api_key:
                raise ValueError("No API key configured for Together AI")
            
            models = [m["name"] for m in config.get("models", [])]
            
            self.providers["together"] = {
                "api_key": api_key,
                "base_url": config.get("base_url"),
                "config": config
            }
            
            self.provider_info["together"] = ProviderInfo(
                name="together",
                enabled=True,
                status=ProviderStatus.ACTIVE,
                models=models,
                api_key_configured=True,
                rate_limits={}
            )
            
            logger.info(f"Together AI initialized with {len(models)} models")
            
        except Exception as e:
            logger.error(f"Failed to initialize Together AI: {str(e)}")
            raise
    
    def _init_anthropic(self, config: Dict):
        """Initialize Anthropic Claude provider (optional)"""
        try:
            api_key = config.get("api_key")
            if not api_key:
                logger.info("Anthropic provider enabled but no API key configured (optional)")
                return
            
            models = [m["name"] for m in config.get("models", [])]
            
            self.providers["anthropic"] = {
                "api_key": api_key,
                "base_url": config.get("base_url"),
                "config": config
            }
            
            self.provider_info["anthropic"] = ProviderInfo(
                name="anthropic",
                enabled=True,
                status=ProviderStatus.ACTIVE,
                models=models,
                api_key_configured=True,
                rate_limits={}
            )
            
            logger.info(f"Anthropic initialized with {len(models)} models")
            
        except Exception as e:
            logger.warning(f"Failed to initialize Anthropic (optional): {str(e)}")
    
    def _init_replicate(self, config: Dict):
        """Initialize Replicate provider (optional)"""
        try:
            api_token = config.get("api_token")
            if not api_token:
                logger.info("Replicate provider enabled but no API token configured (optional)")
                return
            
            self.providers["replicate"] = {
                "api_token": api_token,
                "base_url": config.get("base_url"),
                "config": config
            }
            
            self.provider_info["replicate"] = ProviderInfo(
                name="replicate",
                enabled=True,
                status=ProviderStatus.ACTIVE,
                models=[],
                api_key_configured=True,
                rate_limits={}
            )
            
            logger.info("Replicate initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize Replicate (optional): {str(e)}")
    
    def get_provider(self, name: str) -> Optional[Dict]:
        """Get provider connection"""
        return self.providers.get(name)
    
    def get_provider_info(self, name: str) -> Optional[ProviderInfo]:
        """Get provider information"""
        return self.provider_info.get(name)
    
    def get_all_providers(self) -> Dict[str, Dict]:
        """Get all provider connections"""
        return self.providers
    
    def get_all_provider_info(self) -> Dict[str, ProviderInfo]:
        """Get all provider information"""
        return self.provider_info
    
    def get_active_providers(self) -> List[str]:
        """Get list of active providers"""
        return [
            name for name, info in self.provider_info.items()
            if info.status == ProviderStatus.ACTIVE
        ]
    
    def get_provider_models(self, provider: str) -> List[str]:
        """Get list of models for a provider"""
        info = self.provider_info.get(provider)
        return info.models if info else []
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all providers"""
        health = {}
        
        for name, info in self.provider_info.items():
            health[name] = {
                "status": info.status.value,
                "enabled": info.enabled,
                "models": len(info.models),
                "api_key_configured": info.api_key_configured,
                "error": info.error_message
            }
        
        return health


# Example usage
if __name__ == "__main__":
    import logging
    from config_loader import get_config
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load configuration
    config = get_config()
    
    # Initialize providers
    providers_manager = AIProvidersManager(config)
    
    # Print active providers
    print("\nActive AI Providers:")
    for provider in providers_manager.get_active_providers():
        info = providers_manager.get_provider_info(provider)
        print(f"  - {provider}: {len(info.models)} models")
    
    # Health check
    print("\nProvider Health Check:")
    health = providers_manager.health_check()
    for provider, status in health.items():
        print(f"  - {provider}: {status['status']} ({status['models']} models)")
