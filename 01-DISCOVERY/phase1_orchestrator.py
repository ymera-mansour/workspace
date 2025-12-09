"""Phase 1 Discovery Orchestrator - Controls all 5 processing + 3 validation layers"""
import asyncio
from typing import Dict, List, Any
from .layers import layer1_basic_scan, layer2_classification, layer3_semantic_analysis
from .layers import layer4_pattern_recognition, layer5_expert_integration
from .validators import cross_validator, quality_checker, expert_reviewer
from .tools import file_scanner, context_builder

class Phase1Orchestrator:
    """Orchestrates Phase 1 Discovery with 5 processing + 3 validation layers"""
    
    def __init__(self, config: Dict[str, Any], monitor=None):
        self.config = config
        self.monitor = monitor
        self.context = {}
        
    async def execute(self, repo_path: str) -> Dict[str, Any]:
        """Execute complete Phase 1 Discovery"""
        results = {"phase": "Phase 1: Discovery", "layers": []}
        
        # Layer 1: Basic File Scanning
        print("🔍 Layer 1: Basic File Scanning...")
        layer1 = layer1_basic_scan.Layer1BasicScan(self.config)
        l1_result = await layer1.execute(repo_path)
        results["layers"].append(l1_result)
        self.context.update(l1_result)
        
        # Layer 2: Initial Classification
        print("📋 Layer 2: Initial Classification...")
        layer2 = layer2_classification.Layer2Classification(self.config)
        l2_result = await layer2.execute(self.context)
        results["layers"].append(l2_result)
        self.context.update(l2_result)
        
        # Layer 3: Semantic Analysis
        print("🧠 Layer 3: Semantic Analysis...")
        layer3 = layer3_semantic_analysis.Layer3SemanticAnalysis(self.config)
        l3_result = await layer3.execute(self.context)
        results["layers"].append(l3_result)
        self.context.update(l3_result)
        
        # Layer 4: Advanced Pattern Recognition
        print("🔬 Layer 4: Pattern Recognition...")
        layer4 = layer4_pattern_recognition.Layer4PatternRecognition(self.config)
        l4_result = await layer4.execute(self.context)
        results["layers"].append(l4_result)
        self.context.update(l4_result)
        
        # Layer 5: Expert Knowledge Integration
        print("🎓 Layer 5: Expert Integration...")
        layer5 = layer5_expert_integration.Layer5ExpertIntegration(self.config)
        l5_result = await layer5.execute(self.context)
        results["layers"].append(l5_result)
        self.context.update(l5_result)
        
        # Validation Layer 1: Cross-validation
        print("✓ Validation 1: Cross-validation...")
        val1 = cross_validator.CrossValidator(self.config)
        v1_result = await val1.validate(self.context)
        results["validations"] = [v1_result]
        
        # Validation Layer 2: Quality checks
        print("✓ Validation 2: Quality Checks...")
        val2 = quality_checker.QualityChecker(self.config)
        v2_result = await val2.validate(self.context)
        results["validations"].append(v2_result)
        
        # Validation Layer 3: Expert review + Human approval
        print("✓ Validation 3: Expert Review...")
        val3 = expert_reviewer.ExpertReviewer(self.config)
        v3_result = await val3.validate(self.context)
        results["validations"].append(v3_result)
        
        results["success"] = all(v.get("passed", False) for v in results["validations"])
        return results
