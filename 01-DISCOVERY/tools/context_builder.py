"""Context accumulation tool"""
from typing import Dict, Any

class ContextBuilder:
    """Build and accumulate context across layers"""
    
    def __init__(self):
        self.context = {}
        
    def add(self, key: str, value: Any):
        """Add to context"""
        self.context[key] = value
        
    def get_context(self) -> Dict[str, Any]:
        """Get accumulated context"""
        return self.context.copy()
        
    def merge(self, new_context: Dict[str, Any]):
        """Merge new context"""
        self.context.update(new_context)
