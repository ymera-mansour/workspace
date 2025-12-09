"""
API Key Manager for Phase 0
Manages API keys for all AI providers
"""

import os
import re
from typing import Dict, List, Optional
from pathlib import Path


class APIKeyManager:
    """Manages API keys for all providers"""
    
    def __init__(self, env_file: str = ".env"):
        self.env_file = env_file
        self.providers = {
            'gemini': ['GEMINI_API_KEY', 'GOOGLE_API_KEY'],
            'mistral': ['MISTRAL_API_KEY'],
            'groq': ['GROQ_API_KEY'],
            'openrouter': ['OPENROUTER_API_KEY'],
            'huggingface': ['HUGGINGFACE_API_KEY', 'HF_TOKEN'],
            'cohere': ['COHERE_API_KEY'],
            'together': ['TOGETHER_API_KEY'],
            'anthropic': ['ANTHROPIC_API_KEY', 'CLAUDE_API_KEY'],
            'replicate': ['REPLICATE_API_TOKEN']
        }
        self.keys = {}
        self.missing_keys = []
        self.invalid_keys = []
        
    def validate_all(self) -> bool:
        """Validate all API keys"""
        print("\nAPI Key Manager")
        print("-" * 40)
        
        # Load environment variables
        self._load_env()
        
        # Check each provider
        self._check_providers()
        
        # Generate report
        return self._generate_report()
    
    def _load_env(self):
        """Load environment variables from .env file"""
        if not Path(self.env_file).exists():
            print(f"⚠️  {self.env_file} not found")
            return
        
        try:
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if value:  # Only set if not empty
                            os.environ[key] = value
            
            print(f"✅ Loaded environment from {self.env_file}")
        except Exception as e:
            print(f"❌ Failed to load {self.env_file}: {e}")
    
    def _check_providers(self):
        """Check API keys for each provider"""
        for provider, key_names in self.providers.items():
            found = False
            for key_name in key_names:
                value = os.environ.get(key_name)
                if value and value != 'your_key_here':
                    # Basic format validation
                    if self._validate_key_format(provider, value):
                        self.keys[provider] = key_name
                        found = True
                        break
                    else:
                        self.invalid_keys.append((provider, key_name))
            
            if not found:
                self.missing_keys.append(provider)
    
    def _validate_key_format(self, provider: str, key: str) -> bool:
        """Validate key format (basic checks)"""
        if not key or len(key) < 10:
            return False
        
        # Provider-specific format validation
        patterns = {
            'gemini': r'^AIza[A-Za-z0-9_-]+$',
            'openrouter': r'^sk-or-',
            'anthropic': r'^sk-ant-',
            'huggingface': r'^hf_[A-Za-z0-9]+$',
        }
        
        pattern = patterns.get(provider)
        if pattern:
            return bool(re.match(pattern, key))
        
        # Generic validation for others
        return len(key) >= 20
    
    def _generate_report(self) -> bool:
        """Generate validation report"""
        print("\n" + "-" * 40)
        print("API Key Validation Results")
        print("-" * 40)
        
        print(f"\n✅ Valid keys: {len(self.keys)}/{len(self.providers)}")
        for provider, key_name in self.keys.items():
            print(f"  ✅ {provider}: {key_name}")
        
        if self.missing_keys:
            print(f"\n⚠️  Missing keys ({len(self.missing_keys)}):")
            for provider in self.missing_keys:
                print(f"  ⚠️  {provider}")
        
        if self.invalid_keys:
            print(f"\n❌ Invalid key formats ({len(self.invalid_keys)}):")
            for provider, key_name in self.invalid_keys:
                print(f"  ❌ {provider}: {key_name}")
        
        # Pass if at least one provider has a valid key
        success = len(self.keys) > 0
        
        if success:
            print("\n✅ At least one provider configured")
        else:
            print("\n❌ No valid API keys found")
            print("💡 Add API keys to .env file")
        
        return success


if __name__ == "__main__":
    import sys
    manager = APIKeyManager()
    success = manager.validate_all()
    sys.exit(0 if success else 1)
