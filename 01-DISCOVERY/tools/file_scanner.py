"""File scanning tool"""
import os
from typing import List, Dict, Any

class FileScanner:
    """Recursively scan directories for files"""
    
    def __init__(self, ignore_patterns=None):
        self.ignore_patterns = ignore_patterns or ['.git', 'node_modules', '__pycache__']
        
    def scan_directory(self, path: str, recursive=True) -> List[Dict[str, Any]]:
        """Scan directory and return file list"""
        files = []
        for root, dirs, filenames in os.walk(path):
            # Filter ignored directories
            dirs[:] = [d for d in dirs if d not in self.ignore_patterns]
            for filename in filenames:
                filepath = os.path.join(root, filename)
                files.append({
                    "path": filepath,
                    "name": filename,
                    "size": os.path.getsize(filepath) if os.path.exists(filepath) else 0,
                    "extension": os.path.splitext(filename)[1]
                })
            if not recursive:
                break
        return files
