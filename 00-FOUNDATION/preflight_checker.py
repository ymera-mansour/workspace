"""
Preflight Checker
================

Runs pre-flight checks before starting the workflow to ensure
all dependencies and requirements are met.
"""

import asyncio
import logging
import sys
from typing import Dict, Any, List
from pathlib import Path
import importlib.util

logger = logging.getLogger(__name__)


class PreflightChecker:
    """Runs pre-flight checks"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize preflight checker
        
        Args:
            config: Platform configuration
        """
        self.config = config
        self.checks_passed = []
        self.checks_failed = []
        
    async def check(self) -> bool:
        """
        Run all pre-flight checks
        
        Returns:
            bool: True if all checks passed
        """
        logger.info("Running pre-flight checks...")
        logger.info(f"{'='*60}")
        
        # Run checks
        await self._check_python_version()
        await self._check_dependencies()
        await self._check_directories()
        await self._check_ai_providers()
        await self._check_optimizations()
        await self._check_disk_space()
        await self._check_memory()
        
        # Log results
        logger.info(f"{'='*60}")
        logger.info(f"✅ Checks passed: {len(self.checks_passed)}")
        logger.info(f"❌ Checks failed: {len(self.checks_failed)}")
        
        if self.checks_failed:
            logger.error("Failed checks:")
            for check in self.checks_failed:
                logger.error(f"   - {check}")
            return False
        
        logger.info("✅ All pre-flight checks passed")
        return True
    
    async def _check_python_version(self):
        """Check Python version"""
        check_name = "Python version"
        
        min_version = (3, 8)
        current_version = sys.version_info[:2]
        
        if current_version >= min_version:
            logger.info(f"✅ {check_name}: {sys.version.split()[0]}")
            self.checks_passed.append(check_name)
        else:
            logger.error(f"❌ {check_name}: {current_version} < {min_version}")
            self.checks_failed.append(check_name)
    
    async def _check_dependencies(self):
        """Check required Python packages"""
        check_name = "Dependencies"
        
        required_packages = [
            "yaml",
            "asyncio",
            "pathlib",
            "json"
        ]
        
        optional_packages = [
            "numpy",
            "langchain",
            "faiss",
            "chromadb",
            "prometheus_client"
        ]
        
        missing = []
        missing_optional = []
        
        for package in required_packages:
            if not self._check_package(package):
                missing.append(package)
        
        for package in optional_packages:
            if not self._check_package(package):
                missing_optional.append(package)
        
        if missing:
            logger.error(f"❌ {check_name}: Missing required packages: {', '.join(missing)}")
            self.checks_failed.append(check_name)
        else:
            logger.info(f"✅ {check_name}: All required packages available")
            self.checks_passed.append(check_name)
            
            if missing_optional:
                logger.warning(f"⚠️  Missing optional packages: {', '.join(missing_optional)}")
    
    def _check_package(self, package_name: str) -> bool:
        """Check if a package is available"""
        try:
            # Handle package name variations
            import_name = package_name
            if package_name == "yaml":
                import_name = "yaml"
            
            spec = importlib.util.find_spec(import_name)
            return spec is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            return False
    
    async def _check_directories(self):
        """Check required directories"""
        check_name = "Directories"
        
        required_dirs = [
            "00-FOUNDATION",
            "0X-VALIDATION",
            "ORCHESTRATION",
            "OPTIMIZATIONS"
        ]
        
        missing = []
        
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                missing.append(dir_name)
        
        if missing:
            logger.error(f"❌ {check_name}: Missing directories: {', '.join(missing)}")
            self.checks_failed.append(check_name)
        else:
            logger.info(f"✅ {check_name}: All required directories exist")
            self.checks_passed.append(check_name)
    
    async def _check_ai_providers(self):
        """Check AI provider availability"""
        check_name = "AI Providers"
        
        providers = self.config.get("ai_providers", {})
        enabled_providers = [
            name for name, config in providers.items()
            if config.get("enabled", False)
        ]
        
        if not enabled_providers:
            logger.warning(f"⚠️  {check_name}: No AI providers enabled")
        else:
            logger.info(f"✅ {check_name}: {len(enabled_providers)} providers enabled")
            self.checks_passed.append(check_name)
    
    async def _check_optimizations(self):
        """Check optimization components"""
        check_name = "Optimizations"
        
        optimization_files = [
            "OPTIMIZATIONS/vector_database_optimizer.py",
            "OPTIMIZATIONS/streaming_response_handler.py",
            "OPTIMIZATIONS/semantic_cache_system.py",
            "OPTIMIZATIONS/circuit_breaker.py",
            "OPTIMIZATIONS/analytics_dashboard.py",
            "OPTIMIZATIONS/batch_processor.py"
        ]
        
        missing = []
        
        for file_path in optimization_files:
            if not Path(file_path).exists():
                missing.append(Path(file_path).name)
        
        if missing:
            logger.warning(f"⚠️  {check_name}: Missing optimization files: {', '.join(missing)}")
        else:
            logger.info(f"✅ {check_name}: All optimization components available")
            self.checks_passed.append(check_name)
    
    async def _check_disk_space(self):
        """Check available disk space"""
        check_name = "Disk space"
        
        try:
            import shutil
            
            stat = shutil.disk_usage(".")
            free_gb = stat.free / (1024 ** 3)
            
            min_free_gb = 1.0  # Minimum 1GB free
            
            if free_gb >= min_free_gb:
                logger.info(f"✅ {check_name}: {free_gb:.2f} GB available")
                self.checks_passed.append(check_name)
            else:
                logger.error(f"❌ {check_name}: Only {free_gb:.2f} GB available (need {min_free_gb} GB)")
                self.checks_failed.append(check_name)
                
        except Exception as e:
            logger.warning(f"⚠️  {check_name}: Could not check disk space: {e}")
    
    async def _check_memory(self):
        """Check available memory"""
        check_name = "Memory"
        
        try:
            import psutil
            
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024 ** 3)
            
            min_available_gb = 0.5  # Minimum 500MB available
            
            if available_gb >= min_available_gb:
                logger.info(f"✅ {check_name}: {available_gb:.2f} GB available")
                self.checks_passed.append(check_name)
            else:
                logger.warning(f"⚠️  {check_name}: Only {available_gb:.2f} GB available")
                
        except ImportError:
            logger.info(f"ℹ️  {check_name}: psutil not available, skipping memory check")
        except Exception as e:
            logger.warning(f"⚠️  {check_name}: Could not check memory: {e}")
    
    def get_results(self) -> Dict[str, List[str]]:
        """Get check results"""
        return {
            "passed": self.checks_passed.copy(),
            "failed": self.checks_failed.copy()
        }
