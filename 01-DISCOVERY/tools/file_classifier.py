"""File type classifier"""
from typing import Dict, Any

class FileClassifier:
    """Classify files by type and purpose"""
    
    def classify(self, filepath: str) -> Dict[str, Any]:
        """Classify a file"""
        ext = filepath.split('.')[-1].lower()
        type_map = {
            'py': 'python',
            'js': 'javascript',
            'ts': 'typescript',
            'java': 'java',
            'go': 'go',
            'rs': 'rust',
            'md': 'documentation',
            'txt': 'text',
            'json': 'config',
            'yaml': 'config',
            'yml': 'config',
        }
        return {
            "type": type_map.get(ext, 'unknown'),
            "extension": ext,
            "path": filepath
        }
