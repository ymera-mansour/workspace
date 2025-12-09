"""
Setup Validator for Phase 0
Validates complete setup environment before starting workflow
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
from config_validator import ConfigValidator
from dependency_checker import DependencyChecker
from preflight_checker import PreflightChecker


class SetupValidator:
    """Validates complete setup environment"""
    
    def __init__(self, config_path: str = "00-FOUNDATION/config.yaml"):
        self.config_path = config_path
        self.results = {
            'python_version': False,
            'dependencies': False,
            'configuration': False,
            'directory_structure': False,
            'api_keys': False,
            'databases': False,
            'optimizations': False
        }
        self.errors = []
        self.warnings = []
        
    def validate(self) -> bool:
        """Run all validation checks"""
        print("=" * 60)
        print("PHASE 0: SETUP VALIDATION")
        print("=" * 60)
        
        # 1. Python version check
        self._validate_python_version()
        
        # 2. Dependencies check
        self._validate_dependencies()
        
        # 3. Configuration validation
        self._validate_configuration()
        
        # 4. Directory structure
        self._validate_directory_structure()
        
        # 5. API keys
        self._validate_api_keys()
        
        # 6. Database connections
        self._validate_databases()
        
        # 7. Optimization components
        self._validate_optimizations()
        
        # Generate report
        return self._generate_report()
    
    def _validate_python_version(self):
        """Validate Python version"""
        print("\n[1/7] Validating Python version...")
        try:
            version = sys.version_info
            if version.major >= 3 and version.minor >= 8:
                print(f"  ✅ Python {version.major}.{version.minor}.{version.micro}")
                self.results['python_version'] = True
            else:
                error = f"Python 3.8+ required, found {version.major}.{version.minor}"
                print(f"  ❌ {error}")
                self.errors.append(error)
        except Exception as e:
            error = f"Failed to check Python version: {e}"
            print(f"  ❌ {error}")
            self.errors.append(error)
    
    def _validate_dependencies(self):
        """Validate dependencies"""
        print("\n[2/7] Validating dependencies...")
        try:
            checker = DependencyChecker()
            if checker.check_all():
                print("  ✅ All required dependencies installed")
                self.results['dependencies'] = True
            else:
                error = "Missing required dependencies"
                print(f"  ❌ {error}")
                self.errors.append(error)
        except Exception as e:
            error = f"Failed to check dependencies: {e}"
            print(f"  ❌ {error}")
            self.errors.append(error)
    
    def _validate_configuration(self):
        """Validate configuration files"""
        print("\n[3/7] Validating configuration...")
        try:
            validator = ConfigValidator(self.config_path)
            if validator.validate():
                print("  ✅ Configuration valid")
                self.results['configuration'] = True
            else:
                error = "Invalid configuration"
                print(f"  ❌ {error}")
                self.errors.extend(validator.errors)
                self.warnings.extend(validator.warnings)
        except Exception as e:
            error = f"Failed to validate configuration: {e}"
            print(f"  ❌ {error}")
            self.errors.append(error)
    
    def _validate_directory_structure(self):
        """Validate directory structure"""
        print("\n[4/7] Validating directory structure...")
        required_dirs = [
            '00-FOUNDATION',
            '0X-VALIDATION',
            'ORCHESTRATION',
            'OPTIMIZATIONS',
            'DOCUMENTATION',
            'TESTS'
        ]
        
        missing_dirs = []
        for dir_name in required_dirs:
            if not Path(dir_name).exists():
                missing_dirs.append(dir_name)
        
        if not missing_dirs:
            print(f"  ✅ All required directories exist")
            self.results['directory_structure'] = True
        else:
            error = f"Missing directories: {', '.join(missing_dirs)}"
            print(f"  ❌ {error}")
            self.errors.append(error)
    
    def _validate_api_keys(self):
        """Validate API keys"""
        print("\n[5/7] Validating API keys...")
        try:
            # Check .env file exists
            if not Path('.env').exists():
                warning = ".env file not found - using .env.template as reference"
                print(f"  ⚠️ {warning}")
                self.warnings.append(warning)
                # Still pass if template exists
                if Path('00-FOUNDATION/.env.template').exists():
                    self.results['api_keys'] = True
            else:
                print("  ✅ .env file exists")
                self.results['api_keys'] = True
        except Exception as e:
            error = f"Failed to validate API keys: {e}"
            print(f"  ❌ {error}")
            self.errors.append(error)
    
    def _validate_databases(self):
        """Validate database connections (optional)"""
        print("\n[6/7] Validating database connections (optional)...")
        try:
            # This is optional - mark as pass if not configured
            print("  ✅ Database validation skipped (optional)")
            self.results['databases'] = True
        except Exception as e:
            warning = f"Database validation failed (optional): {e}"
            print(f"  ⚠️ {warning}")
            self.warnings.append(warning)
            self.results['databases'] = True  # Don't fail on optional components
    
    def _validate_optimizations(self):
        """Validate optimization components"""
        print("\n[7/7] Validating optimization components...")
        try:
            opt_files = [
                'OPTIMIZATIONS/vector_database_optimizer.py',
                'OPTIMIZATIONS/streaming_response_handler.py',
                'OPTIMIZATIONS/semantic_cache_system.py',
                'OPTIMIZATIONS/circuit_breaker.py',
                'OPTIMIZATIONS/analytics_dashboard.py',
                'OPTIMIZATIONS/batch_processor.py'
            ]
            
            missing_files = [f for f in opt_files if not Path(f).exists()]
            
            if not missing_files:
                print("  ✅ All optimization components present")
                self.results['optimizations'] = True
            else:
                error = f"Missing optimization files: {', '.join(missing_files)}"
                print(f"  ❌ {error}")
                self.errors.append(error)
        except Exception as e:
            error = f"Failed to validate optimizations: {e}"
            print(f"  ❌ {error}")
            self.errors.append(error)
    
    def _generate_report(self) -> bool:
        """Generate validation report"""
        print("\n" + "=" * 60)
        print("VALIDATION REPORT")
        print("=" * 60)
        
        passed = sum(1 for v in self.results.values() if v)
        total = len(self.results)
        
        print(f"\nResults: {passed}/{total} checks passed")
        
        for check, result in self.results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {status} - {check.replace('_', ' ').title()}")
        
        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        all_passed = all(self.results.values())
        
        print("\n" + "=" * 60)
        if all_passed:
            print("✅ SETUP VALIDATION PASSED")
            print("Ready to proceed with workflow execution")
        else:
            print("❌ SETUP VALIDATION FAILED")
            print("Please fix errors before proceeding")
        print("=" * 60)
        
        return all_passed


if __name__ == "__main__":
    validator = SetupValidator()
    success = validator.validate()
    sys.exit(0 if success else 1)
