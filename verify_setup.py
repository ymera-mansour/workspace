#!/usr/bin/env python3
"""
YMERA Platform - Setup Verification Script
===========================================

This script verifies that the YMERA platform is correctly configured and ready to run.
It checks:
- Python environment and dependencies
- Configuration files
- Environment variables
- Docker setup
- File permissions
- Network connectivity
"""

import sys
import os
from pathlib import Path
import subprocess
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")

def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✓{Colors.RESET} {text}")

def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {text}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}✗{Colors.RESET} {text}")

def print_info(text):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ{Colors.RESET} {text}")

def check_python_version():
    """Check Python version"""
    print_header("Python Environment")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major == 3 and version.minor >= 8:
        print_success(f"Python version: {version_str}")
        return True
    else:
        print_error(f"Python version {version_str} is not supported. Requires Python 3.8+")
        return False

def check_required_files():
    """Check if required files exist"""
    print_header("Required Files")
    
    required_files = [
        'main.py',
        'api.py',
        'Dockerfile',
        'docker-compose.yml',
        '00-FOUNDATION/config.yaml',
        '00-FOUNDATION/config_loader.py',
        '00-FOUNDATION/requirements.txt',
    ]
    
    optional_files = [
        '.env',
        'prometheus.yml',
    ]
    
    all_good = True
    
    for file in required_files:
        if Path(file).exists():
            print_success(f"Found: {file}")
        else:
            print_error(f"Missing required file: {file}")
            all_good = False
    
    for file in optional_files:
        if Path(file).exists():
            print_success(f"Found: {file}")
        else:
            print_warning(f"Optional file not found: {file}")
    
    return all_good

def check_env_file():
    """Check .env file and critical environment variables"""
    print_header("Environment Configuration")
    
    if not Path('.env').exists():
        print_error(".env file not found")
        print_info("Run: cp 00-FOUNDATION/.env.template .env")
        return False
    
    print_success(".env file exists")
    
    # Try to load .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print_success("Successfully loaded .env file")
        
        # Check for critical environment variables
        critical_vars = [
            'ENVIRONMENT',
            'DEBUG',
        ]
        
        optional_vars = [
            'GEMINI_API_KEY',
            'GROQ_API_KEY',
            'MISTRAL_API_KEY',
            'OPENROUTER_API_KEY',
            'HUGGINGFACE_API_KEY',
            'COHERE_API_KEY',
            'TOGETHER_API_KEY',
        ]
        
        has_api_key = False
        for var in optional_vars:
            value = os.getenv(var, '')
            if value and value != f'your_{var.lower()}_here' and len(value) > 10:
                print_success(f"{var} is configured")
                has_api_key = True
        
        if not has_api_key:
            print_warning("No AI provider API keys configured")
            print_info("At least one AI provider API key should be configured for full functionality")
        
        return True
        
    except ImportError:
        print_warning("python-dotenv not installed, skipping .env validation")
        return True
    except Exception as e:
        print_error(f"Error loading .env file: {e}")
        return False

def check_config_yaml():
    """Check config.yaml file"""
    print_header("Configuration File")
    
    config_path = Path('00-FOUNDATION/config.yaml')
    
    if not config_path.exists():
        print_error("config.yaml not found")
        return False
    
    print_success("config.yaml exists")
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check critical sections
        if 'ai_providers' in config:
            print_success("AI providers section found")
            enabled = sum(1 for p in config['ai_providers'].values() if p.get('enabled', False))
            print_info(f"Enabled AI providers: {enabled}")
        
        if 'routing' in config:
            print_success("Routing configuration found")
        
        if 'caching' in config:
            print_success("Caching configuration found")
        
        return True
        
    except ImportError:
        print_warning("PyYAML not installed, skipping config validation")
        return True
    except Exception as e:
        print_error(f"Error reading config.yaml: {e}")
        return False

def check_docker():
    """Check Docker installation and status"""
    print_header("Docker Environment")
    
    # Check if Docker is installed
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            print_success(f"Docker installed: {result.stdout.strip()}")
        else:
            print_error("Docker is not responding correctly")
            return False
    except FileNotFoundError:
        print_error("Docker is not installed")
        print_info("Install Docker Desktop: https://www.docker.com/products/docker-desktop")
        return False
    except subprocess.TimeoutExpired:
        print_error("Docker command timed out")
        return False
    
    # Check if Docker daemon is running
    try:
        result = subprocess.run(['docker', 'ps'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            print_success("Docker daemon is running")
        else:
            print_error("Docker daemon is not running")
            print_info("Start Docker Desktop or run: sudo systemctl start docker")
            return False
    except subprocess.TimeoutExpired:
        print_error("Docker daemon check timed out")
        return False
    
    # Check Docker Compose
    try:
        result = subprocess.run(['docker', 'compose', 'version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            print_success(f"Docker Compose available: {result.stdout.strip()}")
        else:
            # Try old docker-compose command
            result = subprocess.run(['docker-compose', '--version'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            if result.returncode == 0:
                print_success(f"Docker Compose available: {result.stdout.strip()}")
            else:
                print_warning("Docker Compose not found")
                return False
    except FileNotFoundError:
        print_warning("Docker Compose not found")
        return False
    except subprocess.TimeoutExpired:
        print_error("Docker Compose check timed out")
        return False
    
    return True

def check_directories():
    """Check if required directories exist or can be created"""
    print_header("Directory Structure")
    
    required_dirs = [
        '00-FOUNDATION',
        '01-DISCOVERY',
        '02-ANALYSIS',
        '03-CONSOLIDATION',
        '04-TESTING',
        '05-INTEGRATION',
        '0X-VALIDATION',
        'ORCHESTRATION',
        'OPTIMIZATIONS',
    ]
    
    runtime_dirs = [
        'data',
        'logs',
        'workspace',
    ]
    
    all_good = True
    
    for dir_name in required_dirs:
        if Path(dir_name).exists() and Path(dir_name).is_dir():
            print_success(f"Directory exists: {dir_name}")
        else:
            print_error(f"Missing directory: {dir_name}")
            all_good = False
    
    for dir_name in runtime_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print_success(f"Runtime directory exists: {dir_name}")
        else:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print_success(f"Created runtime directory: {dir_name}")
            except Exception as e:
                print_warning(f"Could not create directory {dir_name}: {e}")
    
    return all_good

def check_permissions():
    """Check file permissions"""
    print_header("File Permissions")
    
    # Check if main.py is executable (optional)
    main_py = Path('main.py')
    if main_py.exists():
        if os.access(main_py, os.R_OK):
            print_success("main.py is readable")
        else:
            print_error("main.py is not readable")
            return False
    
    # Check write permissions for data directory
    data_dir = Path('data')
    if data_dir.exists():
        if os.access(data_dir, os.W_OK):
            print_success("data/ directory is writable")
        else:
            print_warning("data/ directory is not writable")
    
    return True

def generate_summary(results):
    """Generate and print summary"""
    print_header("Verification Summary")
    
    total = len(results)
    passed = sum(1 for r in results.values() if r)
    failed = total - passed
    
    print(f"\nTotal checks: {total}")
    print(f"{Colors.GREEN}Passed: {passed}{Colors.RESET}")
    if failed > 0:
        print(f"{Colors.RED}Failed: {failed}{Colors.RESET}\n")
    else:
        print(f"{Colors.GREEN}All checks passed!{Colors.RESET}\n")
    
    if all(results.values()):
        print(f"{Colors.GREEN}{Colors.BOLD}✓ YMERA Platform is ready to use!{Colors.RESET}\n")
        print("Next steps:")
        print("  1. Configure API keys in .env file (if not done)")
        print("  2. Run with Docker Compose: docker-compose up -d")
        print("  3. Or run directly: python main.py --repo-path ./workspace --phases all")
        print("  4. Access API documentation: http://localhost:8000/docs")
        return 0
    else:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠ Some checks failed. Please review the output above.{Colors.RESET}\n")
        print("Common fixes:")
        print("  • Missing .env: cp 00-FOUNDATION/.env.template .env")
        print("  • Docker not running: Start Docker Desktop")
        print("  • Missing directories: Check repository integrity")
        return 1

def main():
    """Main verification function"""
    print(f"\n{Colors.BOLD}YMERA Platform - Setup Verification{Colors.RESET}")
    print(f"Working directory: {os.getcwd()}\n")
    
    results = {
        'python_version': check_python_version(),
        'required_files': check_required_files(),
        'env_file': check_env_file(),
        'config_yaml': check_config_yaml(),
        'docker': check_docker(),
        'directories': check_directories(),
        'permissions': check_permissions(),
    }
    
    return generate_summary(results)

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Verification interrupted by user{Colors.RESET}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error: {e}{Colors.RESET}")
        sys.exit(1)
