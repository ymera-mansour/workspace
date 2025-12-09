"""
Dependency Checker for Phase 0
Checks all Python package dependencies
"""

import sys
import subprocess
from typing import Dict, List, Tuple
from pathlib import Path


class DependencyChecker:
    """Checks Python package dependencies"""
    
    def __init__(self, requirements_file: str = "00-FOUNDATION/requirements.txt"):
        self.requirements_file = requirements_file
        self.required_packages = []
        self.optional_packages = []
        self.installed_packages = {}
        self.missing_packages = []
        self.version_conflicts = []
        
    def check_all(self) -> bool:
        """Check all dependencies"""
        print("\nDependency Checker")
        print("-" * 40)
        
        # 1. Load requirements
        self._load_requirements()
        
        # 2. Get installed packages
        self._get_installed_packages()
        
        # 3. Check required packages
        self._check_required_packages()
        
        # 4. Check optional packages
        self._check_optional_packages()
        
        # 5. Check version conflicts
        self._check_version_conflicts()
        
        # 6. Generate report
        return self._generate_report()
    
    def _load_requirements(self):
        """Load requirements from file"""
        try:
            if not Path(self.requirements_file).exists():
                print(f"⚠️  Requirements file not found: {self.requirements_file}")
                return
            
            with open(self.requirements_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Determine if optional (has comment with 'optional')
                        if '# optional' in line.lower():
                            self.optional_packages.append(line.split('#')[0].strip())
                        else:
                            self.required_packages.append(line)
            
            print(f"✅ Loaded {len(self.required_packages)} required packages")
            print(f"✅ Loaded {len(self.optional_packages)} optional packages")
        except Exception as e:
            print(f"❌ Failed to load requirements: {e}")
    
    def _get_installed_packages(self):
        """Get list of installed packages"""
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'list', '--format=json'],
                capture_output=True,
                text=True,
                check=True
            )
            
            import json
            packages = json.loads(result.stdout)
            self.installed_packages = {pkg['name'].lower(): pkg['version'] for pkg in packages}
            
            print(f"✅ Found {len(self.installed_packages)} installed packages")
        except Exception as e:
            print(f"❌ Failed to get installed packages: {e}")
    
    def _check_required_packages(self):
        """Check required packages"""
        for pkg_spec in self.required_packages:
            pkg_name = self._extract_package_name(pkg_spec)
            if pkg_name.lower() not in self.installed_packages:
                self.missing_packages.append(pkg_spec)
    
    def _check_optional_packages(self):
        """Check optional packages (warnings only)"""
        missing_optional = []
        for pkg_spec in self.optional_packages:
            pkg_name = self._extract_package_name(pkg_spec)
            if pkg_name.lower() not in self.installed_packages:
                missing_optional.append(pkg_spec)
        
        if missing_optional:
            print(f"\n⚠️  Missing optional packages ({len(missing_optional)}):")
            for pkg in missing_optional[:5]:  # Show first 5
                print(f"  - {pkg}")
            if len(missing_optional) > 5:
                print(f"  ... and {len(missing_optional) - 5} more")
    
    def _check_version_conflicts(self):
        """Check for version conflicts"""
        # Simplified version checking
        # In production, would use pip-check or similar
        pass
    
    def _extract_package_name(self, pkg_spec: str) -> str:
        """Extract package name from spec"""
        # Handle formats: package, package>=version, package==version, etc.
        for sep in ['>=', '==', '<=', '>', '<', '~=']:
            if sep in pkg_spec:
                return pkg_spec.split(sep)[0].strip()
        return pkg_spec.strip()
    
    def _generate_report(self) -> bool:
        """Generate dependency report"""
        print("\n" + "-" * 40)
        print("Dependency Check Results")
        print("-" * 40)
        
        if not self.missing_packages:
            print("✅ All required dependencies installed")
            return True
        else:
            print(f"❌ Missing {len(self.missing_packages)} required packages:")
            for pkg in self.missing_packages:
                print(f"  - {pkg}")
            
            print("\n💡 To install missing packages:")
            print(f"  pip install -r {self.requirements_file}")
            
            return False


if __name__ == "__main__":
    checker = DependencyChecker()
    success = checker.check_all()
    sys.exit(0 if success else 1)
