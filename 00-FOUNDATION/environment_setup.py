"""
Environment Setup for Phase 0
Automated environment setup
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict


class EnvironmentSetup:
    """Automated environment setup"""
    
    def __init__(self):
        self.base_path = Path.cwd()
        self.directories = [
            '00-FOUNDATION',
            '0X-VALIDATION',
            '01-DISCOVERY',
            '02-ANALYSIS',
            '03-CONSOLIDATION',
            '04-TESTING',
            '05-INTEGRATION',
            'ORCHESTRATION',
            'OPTIMIZATIONS',
            'DOCUMENTATION',
            'TESTS',
            'UTILITIES',
            'logs',
            'checkpoints',
            'results',
            'cache'
        ]
        self.created_dirs = []
        self.errors = []
        
    def setup(self) -> bool:
        """Run complete setup"""
        print("=" * 60)
        print("PHASE 0: ENVIRONMENT SETUP")
        print("=" * 60)
        
        # 1. Create directory structure
        self._create_directories()
        
        # 2. Initialize configuration files
        self._initialize_configs()
        
        # 3. Setup logging directories
        self._setup_logging()
        
        # 4. Create checkpoint directories
        self._setup_checkpoints()
        
        # 5. Validate permissions
        self._validate_permissions()
        
        # Generate report
        return self._generate_report()
    
    def _create_directories(self):
        """Create directory structure"""
        print("\n[1/5] Creating directory structure...")
        
        for dir_name in self.directories:
            dir_path = self.base_path / dir_name
            try:
                if not dir_path.exists():
                    dir_path.mkdir(parents=True, exist_ok=True)
                    self.created_dirs.append(dir_name)
                    print(f"  ✅ Created {dir_name}/")
                else:
                    print(f"  ⏭️  {dir_name}/ already exists")
            except Exception as e:
                error = f"Failed to create {dir_name}: {e}"
                print(f"  ❌ {error}")
                self.errors.append(error)
    
    def _initialize_configs(self):
        """Initialize configuration files"""
        print("\n[2/5] Initializing configuration files...")
        
        # Create .env if doesn't exist
        env_path = self.base_path / '.env'
        template_path = self.base_path / '00-FOUNDATION' / '.env.template'
        
        if not env_path.exists() and template_path.exists():
            try:
                shutil.copy(template_path, env_path)
                print("  ✅ Created .env from template")
                print("  ⚠️  Remember to add your API keys to .env")
            except Exception as e:
                error = f"Failed to create .env: {e}"
                print(f"  ❌ {error}")
                self.errors.append(error)
        else:
            print("  ⏭️  .env already exists or template not found")
    
    def _setup_logging(self):
        """Setup logging directories"""
        print("\n[3/5] Setting up logging directories...")
        
        log_dirs = ['logs/workflow', 'logs/phases', 'logs/errors', 'logs/performance']
        
        for log_dir in log_dirs:
            dir_path = self.base_path / log_dir
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"  ✅ Created {log_dir}/")
            except Exception as e:
                error = f"Failed to create log directory {log_dir}: {e}"
                print(f"  ❌ {error}")
                self.errors.append(error)
    
    def _setup_checkpoints(self):
        """Create checkpoint directories"""
        print("\n[4/5] Creating checkpoint directories...")
        
        checkpoint_dirs = ['checkpoints/states', 'checkpoints/snapshots', 'checkpoints/backups']
        
        for checkpoint_dir in checkpoint_dirs:
            dir_path = self.base_path / checkpoint_dir
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"  ✅ Created {checkpoint_dir}/")
            except Exception as e:
                error = f"Failed to create checkpoint directory {checkpoint_dir}: {e}"
                print(f"  ❌ {error}")
                self.errors.append(error)
    
    def _validate_permissions(self):
        """Validate directory permissions"""
        print("\n[5/5] Validating permissions...")
        
        # Check write permissions on key directories
        test_dirs = ['logs', 'checkpoints', 'results', 'cache']
        
        for dir_name in test_dirs:
            dir_path = self.base_path / dir_name
            try:
                # Try to create a test file
                test_file = dir_path / '.permission_test'
                test_file.write_text('test')
                test_file.unlink()
                print(f"  ✅ {dir_name}/ is writable")
            except Exception as e:
                error = f"Permission denied for {dir_name}: {e}"
                print(f"  ❌ {error}")
                self.errors.append(error)
    
    def _generate_report(self) -> bool:
        """Generate setup report"""
        print("\n" + "=" * 60)
        print("SETUP REPORT")
        print("=" * 60)
        
        print(f"\nCreated directories: {len(self.created_dirs)}")
        
        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")
            success = False
        else:
            print("\n✅ Environment setup completed successfully")
            success = True
        
        print("=" * 60)
        
        return success


if __name__ == "__main__":
    import sys
    setup = EnvironmentSetup()
    success = setup.setup()
    sys.exit(0 if success else 1)
