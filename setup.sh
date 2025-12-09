#!/bin/bash

# ============================================
# YMERA AI Platform - Setup Script
# ============================================
# This script sets up the complete YMERA platform
# including all dependencies and configurations

set -e  # Exit on error

echo "================================================"
echo "YMERA AI Platform - Initial Setup"
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo "ℹ $1"
}

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then 
    print_success "Python $PYTHON_VERSION detected (>= $REQUIRED_VERSION required)"
else
    print_error "Python $REQUIRED_VERSION or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists, skipping creation"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
print_success "pip upgraded"

# Install Python dependencies
echo ""
echo "Installing Python dependencies (this may take 10-15 minutes)..."
pip install -r requirements.txt
print_success "Python dependencies installed"

# Create .env file from template
echo ""
echo "Setting up environment configuration..."
if [ ! -f ".env" ]; then
    cp .env.template .env
    print_success ".env file created from template"
    print_warning "Please edit .env file and add your API keys"
else
    print_warning ".env file already exists, skipping"
fi

# Create necessary directories
echo ""
echo "Creating required directories..."
mkdir -p data/faiss
mkdir -p data/chroma
mkdir -p logs
mkdir -p mlruns
mkdir -p workspace
print_success "Directories created"

# Download spaCy models
echo ""
echo "Downloading spaCy language models..."
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
print_success "spaCy models downloaded"

# Download NLTK data
echo ""
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
print_success "NLTK data downloaded"

# Check Node.js installation
echo ""
echo "Checking Node.js installation..."
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    print_success "Node.js $NODE_VERSION detected"
    
    # Install MCP tools
    echo ""
    echo "Installing MCP tools..."
    print_info "This will install 18 MCP servers globally"
    read -p "Do you want to install MCP tools now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ./install_mcp_tools.sh
    else
        print_warning "Skipping MCP tools installation. Run './install_mcp_tools.sh' later."
    fi
else
    print_warning "Node.js not found. MCP tools require Node.js v18+"
    print_info "Install Node.js from: https://nodejs.org/"
fi

# Initialize configuration
echo ""
echo "Validating configuration..."
python config_loader.py
print_success "Configuration validated"

# Run provider initialization test
echo ""
echo "Testing AI provider connections..."
python providers_init.py
print_success "Provider connections tested"

# Setup complete
echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
print_success "YMERA AI Platform is ready to use"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your API keys"
echo "2. Review config.yaml for system configuration"
echo "3. Activate virtual environment: source venv/bin/activate"
echo "4. Run the platform: python main.py"
echo ""
print_info "For MCP tools installation, run: ./install_mcp_tools.sh"
print_info "For documentation, see the .md files in the repository"
echo ""
