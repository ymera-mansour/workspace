"""Dependency analysis tool"""
import re
from typing import List, Dict, Any

class DependencyAnalyzer:
    """Analyze code dependencies"""
    
    def analyze_python(self, content: str) -> List[str]:
        """Extract Python imports"""
        imports = re.findall(r'^import (\w+)', content, re.MULTILINE)
        from_imports = re.findall(r'^from (\w+)', content, re.MULTILINE)
        return list(set(imports + from_imports))
        
    def analyze_javascript(self, content: str) -> List[str]:
        """Extract JavaScript imports"""
        imports = re.findall(r"require\(['"](.+?)['"]\)", content)
        es6_imports = re.findall(r"import .+ from ['"](.+?)['"]", content)
        return list(set(imports + es6_imports))
