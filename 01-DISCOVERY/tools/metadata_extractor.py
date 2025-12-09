"""Metadata extraction"""
from typing import Dict, Any
import os

class MetadataExtractor:
    """Extract file metadata"""
    
    def extract(self, filepath: str) -> Dict[str, Any]:
        """Extract metadata from file"""
        stat = os.stat(filepath)
        return {
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "created": stat.st_ctime,
            "path": filepath,
            "name": os.path.basename(filepath)
        }
