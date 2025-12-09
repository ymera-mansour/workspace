"""Multi-format file reader"""
import os
from typing import Any

class FileReader:
    """Read files in various formats"""
    
    def read_file(self, filepath: str) -> str:
        """Read file content"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {e}"
