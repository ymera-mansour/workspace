"""Layer 1: Basic File Scanning using fast models"""
import asyncio
from typing import Dict, Any

class Layer1BasicScan:
    """Fast file scanning with Ministral-3B, Gemini Flash-8B"""
    
    def __init__(self, config):
        self.config = config
        self.models = ["ministral-3b", "gemini-flash-8b"]
        
    async def execute(self, repo_path: str) -> Dict[str, Any]:
        """Execute basic file scanning"""
        # Simulate fast scanning
        await asyncio.sleep(0.1)
        return {
            "layer": "Layer 1: Basic Scan",
            "files_found": 150,
            "directories": 25,
            "models_used": self.models,
            "duration_ms": 100
        }
