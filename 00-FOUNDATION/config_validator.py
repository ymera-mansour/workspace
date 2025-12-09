"""
Configuration Validator
=======================

Validates platform configuration for completeness and correctness.
"""

import logging
from typing import Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validates platform configuration"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize config validator
        
        Args:
            config: Configuration to validate
        """
        self.config = config
        self.errors = []
        self.warnings = []
        
    def validate(self) -> bool:
        """
        Validate configuration
        
        Returns:
            bool: True if valid, False otherwise
        """
        logger.info("Validating configuration...")
        
        # Validate AI providers
        self._validate_ai_providers()
        
        # Validate paths
        self._validate_paths()
        
        # Validate Phase X settings
        self._validate_phase_x()
        
        # Validate optimization settings
        self._validate_optimizations()
        
        # Log results
        if self.errors:
            logger.error(f"❌ Configuration validation failed with {len(self.errors)} errors:")
            for error in self.errors:
                logger.error(f"   - {error}")
            return False
        
        if self.warnings:
            logger.warning(f"⚠️  Configuration has {len(self.warnings)} warnings:")
            for warning in self.warnings:
                logger.warning(f"   - {warning}")
        
        logger.info("✅ Configuration validation passed")
        return True
    
    def _validate_ai_providers(self):
        """Validate AI provider configuration"""
        providers = self.config.get("ai_providers", {})
        
        if not providers:
            self.errors.append("No AI providers configured")
            return
        
        required_providers = ["gemini", "groq", "mistral"]
        for provider in required_providers:
            if provider not in providers:
                self.warnings.append(f"Recommended provider '{provider}' not configured")
        
        # Validate each provider
        for provider_name, provider_config in providers.items():
            if not isinstance(provider_config, dict):
                self.errors.append(f"Provider '{provider_name}' config must be a dict")
                continue
            
            # Check if enabled
            if not provider_config.get("enabled", False):
                continue
            
            # Check for models
            models = provider_config.get("models", [])
            if not models:
                self.warnings.append(f"Provider '{provider_name}' has no models configured")
    
    def _validate_paths(self):
        """Validate file paths in configuration"""
        # Check if state directory is writable
        state_dir = self.config.get("state_manager", {}).get("state_dir", ".ymera_state")
        try:
            Path(state_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.errors.append(f"Cannot create state directory '{state_dir}': {e}")
    
    def _validate_phase_x(self):
        """Validate Phase X configuration"""
        phase_x = self.config.get("phase_x", {})
        
        # Check quality threshold
        threshold = phase_x.get("quality_threshold", 7.0)
        if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 10:
            self.errors.append(f"Invalid phase_x.quality_threshold: {threshold} (must be 0-10)")
        
        # Check min scores
        for key in ["min_accuracy", "min_completeness"]:
            value = phase_x.get(key, 7.0)
            if not isinstance(value, (int, float)) or value < 0 or value > 10:
                self.errors.append(f"Invalid phase_x.{key}: {value} (must be 0-10)")
    
    def _validate_optimizations(self):
        """Validate optimization settings"""
        # Validate caching
        cache_config = self.config.get("semantic_cache", {})
        threshold = cache_config.get("similarity_threshold", 0.95)
        if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
            self.errors.append(f"Invalid semantic_cache.similarity_threshold: {threshold} (must be 0-1)")
        
        # Validate batch processing
        batch_config = self.config.get("batch_processing", {})
        max_concurrent = batch_config.get("max_concurrent", 10)
        if not isinstance(max_concurrent, int) or max_concurrent < 1:
            self.errors.append(f"Invalid batch_processing.max_concurrent: {max_concurrent} (must be >= 1)")
    
    def get_errors(self) -> List[str]:
        """Get validation errors"""
        return self.errors.copy()
    
    def get_warnings(self) -> List[str]:
        """Get validation warnings"""
        return self.warnings.copy()
