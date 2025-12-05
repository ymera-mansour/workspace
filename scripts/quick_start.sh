#!/bin/bash

# YMERA Multi-Agent Workspace Platform
# Quick Start Setup Script
# For Linux/macOS/WSL

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Banner
echo -e "${GREEN}"
cat << "EOF"
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ██╗   ██╗███╗   ███╗███████╗██████╗  █████╗              ║
║   ╚██╗ ██╔╝████╗ ████║██╔════╝██╔══██╗██╔══██╗             ║
║    ╚████╔╝ ██╔████╔██║█████╗  ██████╔╝███████║             ║
║     ╚██╔╝  ██║╚██╔╝██║██╔══╝  ██╔══██╗██╔══██║             ║
║      ██║   ██║ ╚═╝ ██║███████╗██║  ██║██║  ██║             ║
║      ╚═╝   ╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝             ║
║                                                              ║
║        Multi-Agent Workspace Platform - Quick Start         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

log_info "Starting YMERA setup..."
echo ""

# Step 1: Check prerequisites
log_info "Step 1/8: Checking prerequisites..."

if ! command_exists python3; then
    log_error "Python 3.11+ is required but not installed."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
log_success "Python $PYTHON_VERSION found"

if ! command_exists git; then
    log_error "Git is required but not installed."
    exit 1
fi
log_success "Git found"

if ! command_exists node; then
    log_warning "Node.js not found. MCP servers will not be available."
else
    log_success "Node.js found"
fi

if ! command_exists docker; then
    log_warning "Docker not found. Using manual installation mode."
    DOCKER_MODE=false
else
    log_success "Docker found"
    DOCKER_MODE=true
fi

echo ""

# Step 2: Check if .env exists
log_info "Step 2/8: Checking configuration..."

if [ ! -f .env ]; then
    log_info "Creating .env file from template..."
    cp .env.example .env
    log_warning "Please edit .env file with your API keys before continuing."
    log_info "Press Enter when ready..."
    read -r
fi

# Load environment
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    log_success "Environment loaded"
fi

echo ""

# Step 3: Choose installation mode
log_info "Step 3/8: Selecting installation mode..."

if [ "$DOCKER_MODE" = true ]; then
    echo "Choose installation mode:"
    echo "  1) Docker (recommended - easiest)"
    echo "  2) Manual (more control)"
    read -p "Enter choice [1-2]: " INSTALL_MODE
else
    INSTALL_MODE=2
fi

echo ""

# Step 4: Docker installation
if [ "$INSTALL_MODE" = "1" ]; then
    log_info "Step 4/8: Docker installation selected..."
    
    # Check if docker-compose exists
    if ! command_exists docker-compose; then
        log_error "docker-compose is required but not installed."
        exit 1
    fi
    
    log_info "Building Docker images..."
    docker-compose build
    log_success "Docker images built"
    
    log_info "Starting services..."
    docker-compose up -d
    log_success "Services started"
    
    log_info "Waiting for services to be healthy..."
    sleep 10
    
    # Check health
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        log_success "Application is running!"
    else
        log_error "Application failed to start. Check logs: docker-compose logs"
        exit 1
    fi
    
    echo ""
    log_info "Docker installation complete!"
    log_info "Access the platform at: http://localhost:8000"
    log_info "View logs: docker-compose logs -f"
    log_info "Stop services: docker-compose down"
    
else
    # Manual installation
    log_info "Step 4/8: Manual installation selected..."
    
    # Create virtual environment
    log_info "Creating Python virtual environment..."
    python3 -m venv venv
    log_success "Virtual environment created"
    
    # Activate virtual environment
    log_info "Activating virtual environment..."
    source venv/bin/activate
    log_success "Virtual environment activated"
    
    # Install Python dependencies
    log_info "Installing Python dependencies (this may take a few minutes)..."
    pip install --upgrade pip >/dev/null 2>&1
    pip install -r requirements.txt
    log_success "Python dependencies installed"
    
    # Install Node.js dependencies
    if command_exists npm; then
        log_info "Installing Node.js dependencies..."
        npm install
        log_success "Node.js dependencies installed"
        
        # Install MCP servers
        log_info "Installing MCP servers..."
        npm install @modelcontextprotocol/server-brave-search
        npm install @modelcontextprotocol/server-filesystem
        npm install @modelcontextprotocol/server-github
        log_success "MCP servers installed"
    fi
    
    echo ""
    
    # Step 5: Check Redis
    log_info "Step 5/8: Checking Redis..."
    
    if command_exists redis-cli; then
        if redis-cli ping >/dev/null 2>&1; then
            log_success "Redis is running"
        else
            log_warning "Redis is not running. Starting Redis..."
            if command_exists redis-server; then
                redis-server --daemonize yes
                sleep 2
                if redis-cli ping >/dev/null 2>&1; then
                    log_success "Redis started"
                else
                    log_error "Failed to start Redis"
                fi
            else
                log_error "Redis is not installed. Please install Redis."
                exit 1
            fi
        fi
    else
        log_warning "Redis not found. Install: sudo apt install redis-server (Ubuntu) or brew install redis (macOS)"
    fi
    
    echo ""
    
    # Step 6: Initialize database
    log_info "Step 6/8: Initializing database..."
    
    if [ -f scripts/init_db.py ]; then
        python scripts/init_db.py
        log_success "Database initialized"
    else
        log_warning "Database initialization script not found. Skipping..."
    fi
    
    echo ""
    
    # Step 7: Verify API keys
    log_info "Step 7/8: Verifying API keys..."
    
    KEYS_SET=0
    KEYS_TOTAL=0
    
    check_key() {
        KEYS_TOTAL=$((KEYS_TOTAL + 1))
        if [ ! -z "${!1}" ] && [ "${!1}" != "your_${1,,}_here" ]; then
            log_success "$1 is set"
            KEYS_SET=$((KEYS_SET + 1))
            return 0
        else
            log_warning "$1 is not set"
            return 1
        fi
    }
    
    check_key "GROQ_API_KEY"
    check_key "GEMINI_API_KEY"
    check_key "GITHUB_TOKEN"
    
    echo ""
    log_info "API keys configured: $KEYS_SET/$KEYS_TOTAL"
    
    if [ $KEYS_SET -eq 0 ]; then
        log_error "No API keys configured! Please edit .env file."
        log_info "Get free API keys from:"
        echo "  - Groq: https://console.groq.com/ (FREE, unlimited)"
        echo "  - Gemini: https://ai.google.dev/ (FREE tier)"
        echo "  - GitHub: https://github.com/settings/tokens"
        exit 1
    fi
    
    echo ""
    
    # Step 8: Start application
    log_info "Step 8/8: Starting application..."
    
    log_info "The application will start in background..."
    log_info "Starting MCP server..."
    
    if [ -d mcp-server ]; then
        cd mcp-server
        node server.js > ../logs/mcp-server.log 2>&1 &
        MCP_PID=$!
        echo $MCP_PID > ../tmp/mcp-server.pid
        cd ..
        log_success "MCP server started (PID: $MCP_PID)"
    fi
    
    log_info "Starting YMERA platform..."
    
    # Create logs directory
    mkdir -p logs
    
    # Start main application
    uvicorn src.core.api:app --host 0.0.0.0 --port 8000 > logs/app.log 2>&1 &
    APP_PID=$!
    echo $APP_PID > tmp/app.pid
    
    log_success "YMERA platform started (PID: $APP_PID)"
    
    # Wait for application to start
    log_info "Waiting for application to start..."
    sleep 5
    
    # Test health
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        log_success "Application is running!"
    else
        log_error "Application failed to start. Check logs: tail -f logs/app.log"
        exit 1
    fi
fi

echo ""
echo -e "${GREEN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║                 🎉 SETUP COMPLETE! 🎉                       ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo ""
log_success "YMERA Multi-Agent Platform is now running!"
echo ""
log_info "Access the platform:"
echo "  - Web UI: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo "  - Health Check: http://localhost:8000/health"
echo ""

# Get local IP
if command_exists ip; then
    LOCAL_IP=$(ip route get 1 | awk '{print $7;exit}')
    log_info "Access from iPhone: http://$LOCAL_IP:8000"
elif command_exists ifconfig; then
    LOCAL_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -1)
    log_info "Access from iPhone: http://$LOCAL_IP:8000"
fi

echo ""
log_info "Useful commands:"
if [ "$INSTALL_MODE" = "1" ]; then
    echo "  - View logs: docker-compose logs -f"
    echo "  - Stop services: docker-compose down"
    echo "  - Restart: docker-compose restart"
else
    echo "  - View logs: tail -f logs/app.log"
    echo "  - Stop services: ./scripts/stop.sh"
    echo "  - Restart: ./scripts/restart.sh"
fi

echo ""
log_info "Next steps:"
echo "  1. Test a simple request"
echo "  2. Explore the API documentation"
echo "  3. Read the guides in docs/"
echo "  4. Join our community for support"

echo ""
log_success "Happy building with YMERA! 🚀"
