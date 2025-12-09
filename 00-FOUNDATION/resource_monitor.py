"""
Resource Monitor for Phase 0
System resource monitoring
"""

import os
import psutil
import time
from typing import Dict, List, Tuple
from pathlib import Path


class ResourceMonitor:
    """System resource monitoring"""
    
    def __init__(self):
        self.thresholds = {
            'cpu_percent': 90.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0,
            'min_disk_gb': 1.0,
            'min_memory_mb': 500.0
        }
        self.warnings = []
        self.errors = []
        
    def check_resources(self) -> bool:
        """Check system resources"""
        print("\nResource Monitor")
        print("-" * 40)
        
        # 1. CPU usage
        cpu_ok = self._check_cpu()
        
        # 2. Memory usage
        memory_ok = self._check_memory()
        
        # 3. Disk space
        disk_ok = self._check_disk()
        
        # Generate report
        return self._generate_report(cpu_ok, memory_ok, disk_ok)
    
    def _check_cpu(self) -> bool:
        """Check CPU usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            print(f"\nCPU:")
            print(f"  Usage: {cpu_percent}%")
            print(f"  Cores: {cpu_count}")
            
            if cpu_percent > self.thresholds['cpu_percent']:
                warning = f"High CPU usage: {cpu_percent}%"
                print(f"  ⚠️  {warning}")
                self.warnings.append(warning)
                return False
            else:
                print(f"  ✅ CPU usage normal")
                return True
        except Exception as e:
            error = f"Failed to check CPU: {e}"
            print(f"  ❌ {error}")
            self.errors.append(error)
            return False
    
    def _check_memory(self) -> bool:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            memory_available_gb = memory.available / (1024**3)
            
            print(f"\nMemory:")
            print(f"  Total: {memory_gb:.2f} GB")
            print(f"  Available: {memory_available_gb:.2f} GB")
            print(f"  Usage: {memory.percent}%")
            
            if memory.percent > self.thresholds['memory_percent']:
                warning = f"High memory usage: {memory.percent}%"
                print(f"  ⚠️  {warning}")
                self.warnings.append(warning)
                return False
            
            min_memory_gb = self.thresholds['min_memory_mb'] / 1024
            if memory_available_gb < min_memory_gb:
                error = f"Insufficient available memory: {memory_available_gb:.2f} GB"
                print(f"  ❌ {error}")
                self.errors.append(error)
                return False
            
            print(f"  ✅ Memory adequate")
            return True
        except Exception as e:
            error = f"Failed to check memory: {e}"
            print(f"  ❌ {error}")
            self.errors.append(error)
            return False
    
    def _check_disk(self) -> bool:
        """Check disk space"""
        try:
            disk = psutil.disk_usage('.')
            disk_gb = disk.total / (1024**3)
            disk_free_gb = disk.free / (1024**3)
            
            print(f"\nDisk:")
            print(f"  Total: {disk_gb:.2f} GB")
            print(f"  Free: {disk_free_gb:.2f} GB")
            print(f"  Usage: {disk.percent}%")
            
            if disk.percent > self.thresholds['disk_percent']:
                warning = f"High disk usage: {disk.percent}%"
                print(f"  ⚠️  {warning}")
                self.warnings.append(warning)
                return False
            
            if disk_free_gb < self.thresholds['min_disk_gb']:
                error = f"Insufficient disk space: {disk_free_gb:.2f} GB"
                print(f"  ❌ {error}")
                self.errors.append(error)
                return False
            
            print(f"  ✅ Disk space adequate")
            return True
        except Exception as e:
            error = f"Failed to check disk: {e}"
            print(f"  ❌ {error}")
            self.errors.append(error)
            return False
    
    def _generate_report(self, cpu_ok: bool, memory_ok: bool, disk_ok: bool) -> bool:
        """Generate resource report"""
        print("\n" + "-" * 40)
        print("Resource Check Results")
        print("-" * 40)
        
        all_ok = cpu_ok and memory_ok and disk_ok
        
        if all_ok:
            print("\n✅ All resources adequate")
        else:
            print("\n⚠️  Resource warnings detected")
        
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")
        
        return all_ok or len(self.errors) == 0  # Pass if no errors


if __name__ == "__main__":
    import sys
    monitor = ResourceMonitor()
    success = monitor.check_resources()
    sys.exit(0 if success else 1)
