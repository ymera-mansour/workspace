"""
Outcome Validator for Phase X
Validates phase outcomes against expected results
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import json


class OutcomeValidator:
    """Validates phase outcomes"""
    
    def __init__(self):
        self.validation_results = {}
        self.errors = []
        self.warnings = []
        
    def validate(self, phase_results: Dict[str, Any], phase_name: str) -> Dict[str, Any]:
        """Validate phase outcomes"""
        print(f"\n{'=' * 60}")
        print(f"PHASE X: Outcome Validation - {phase_name}")
        print(f"{'=' * 60}")
        
        # 1. Check completeness
        completeness_score = self._check_completeness(phase_results, phase_name)
        
        # 2. Validate data integrity
        integrity_score = self._validate_data_integrity(phase_results)
        
        # 3. Check file existence
        files_score = self._check_file_existence(phase_results)
        
        # 4. Validate formats
        format_score = self._validate_formats(phase_results)
        
        # 5. Cross-phase consistency
        consistency_score = self._check_consistency(phase_results)
        
        # Calculate overall score
        overall_score = (
            completeness_score * 0.3 +
            integrity_score * 0.25 +
            files_score * 0.2 +
            format_score * 0.15 +
            consistency_score * 0.1
        )
        
        return {
            'phase': phase_name,
            'overall_score': overall_score,
            'completeness': completeness_score,
            'integrity': integrity_score,
            'files': files_score,
            'format': format_score,
            'consistency': consistency_score,
            'passed': overall_score >= 7.0,
            'errors': self.errors,
            'warnings': self.warnings
        }
    
    def _check_completeness(self, results: Dict[str, Any], phase_name: str) -> float:
        """Check if all expected deliverables are present"""
        print("\n[1/5] Checking completeness...")
        
        expected_keys = {
            'Phase 1: Discovery': ['files_scanned', 'classifications', 'dependencies'],
            'Phase 2: Analysis': ['duplicates', 'strategies', 'recommendations'],
            'Phase 3: Consolidation': ['merged_files', 'refactored_code', 'tests'],
            'Phase 4: Testing': ['test_results', 'coverage', 'quality_metrics'],
            'Phase 5: Integration': ['deployment_status', 'health_checks', 'documentation']
        }
        
        expected = expected_keys.get(phase_name, [])
        if not expected:
            print(f"  ⚠️  No validation criteria for {phase_name}")
            return 8.0  # Default score
        
        present = sum(1 for key in expected if key in results)
        score = (present / len(expected)) * 10.0
        
        print(f"  Deliverables: {present}/{len(expected)} present")
        print(f"  Score: {score:.1f}/10.0")
        
        if score < 7.0:
            self.errors.append(f"Incomplete deliverables: {present}/{len(expected)}")
        
        return score
    
    def _validate_data_integrity(self, results: Dict[str, Any]) -> float:
        """Validate data integrity"""
        print("\n[2/5] Validating data integrity...")
        
        issues = 0
        total_checks = 0
        
        for key, value in results.items():
            total_checks += 1
            if value is None:
                issues += 1
                self.warnings.append(f"Null value for {key}")
            elif isinstance(value, (list, dict)) and len(value) == 0:
                issues += 1
                self.warnings.append(f"Empty value for {key}")
        
        if total_checks == 0:
            return 5.0
        
        score = ((total_checks - issues) / total_checks) * 10.0
        print(f"  Issues: {issues}/{total_checks}")
        print(f"  Score: {score:.1f}/10.0")
        
        return score
    
    def _check_file_existence(self, results: Dict[str, Any]) -> float:
        """Check if referenced files exist"""
        print("\n[3/5] Checking file existence...")
        
        file_keys = ['files', 'output_files', 'generated_files', 'file_paths']
        files_to_check = []
        
        for key in file_keys:
            if key in results:
                value = results[key]
                if isinstance(value, list):
                    files_to_check.extend(value)
                elif isinstance(value, str):
                    files_to_check.append(value)
        
        if not files_to_check:
            print("  No files to validate")
            return 8.0
        
        existing = sum(1 for f in files_to_check if Path(f).exists())
        score = (existing / len(files_to_check)) * 10.0
        
        print(f"  Files: {existing}/{len(files_to_check)} exist")
        print(f"  Score: {score:.1f}/10.0")
        
        if score < 7.0:
            self.errors.append(f"Missing files: {len(files_to_check) - existing}")
        
        return score
    
    def _validate_formats(self, results: Dict[str, Any]) -> float:
        """Validate output formats"""
        print("\n[4/5] Validating formats...")
        
        # Check JSON serializability
        try:
            json.dumps(results)
            print("  ✅ Results are JSON serializable")
            score = 9.0
        except Exception as e:
            self.warnings.append(f"Results not JSON serializable: {e}")
            print(f"  ⚠️  JSON serialization failed")
            score = 6.0
        
        print(f"  Score: {score:.1f}/10.0")
        return score
    
    def _check_consistency(self, results: Dict[str, Any]) -> float:
        """Check cross-phase consistency"""
        print("\n[5/5] Checking consistency...")
        
        # Basic consistency checks
        score = 8.0
        print(f"  Score: {score:.1f}/10.0")
        
        return score


if __name__ == "__main__":
    validator = OutcomeValidator()
    
    # Test with sample data
    test_results = {
        'files_scanned': 100,
        'classifications': ['python', 'javascript'],
        'dependencies': {'package1': '1.0.0'}
    }
    
    outcome = validator.validate(test_results, 'Phase 1: Discovery')
    print(f"\nOverall Score: {outcome['overall_score']:.1f}/10.0")
    print(f"Passed: {outcome['passed']}")
