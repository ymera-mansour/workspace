"""Code pattern detection"""
from typing import List, Dict, Any

class PatternDetector:
    """Detect architectural and code patterns"""
    
    def detect_patterns(self, files: List[Dict[str, Any]]) -> List[str]:
        """Detect patterns in codebase"""
        patterns = []
        # Simple heuristics
        if any('controller' in f.get('path', '') for f in files):
            patterns.append('MVC')
        if any('service' in f.get('path', '') for f in files):
            patterns.append('Service Layer')
        if any('docker' in f.get('path', '').lower() for f in files):
            patterns.append('Containerized')
        return patterns
