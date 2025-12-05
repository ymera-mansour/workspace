# YMERA Multi-Agent Workspace Platform
# Quick Start Setup Script for Windows PowerShell
# Run as: .\scripts\quick_start.ps1

param(
    [switch]$Docker,
    [switch]$Manual
)

# Requires -Version 5.1

# Colors for output
function Write-Info {
    Write-Host "ℹ $args" -ForegroundColor Blue
}

function Write-Success {
    Write-Host "✓ $args" -ForegroundColor Green
}

function Write-Warning {
    Write-Host "⚠ $args" -ForegroundColor Yellow
}

function Write-Error-Message {
    Write-Host "✗ $args" -ForegroundColor Red
}

# Check if command exists
function Test-CommandExists {
    param($Command)
    $null -ne (Get-Command $Command -ErrorAction SilentlyContinue)
}

# Banner
Write-Host @"

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

"@ -ForegroundColor Green

Write-Info "Starting YMERA setup for Windows..."
Write-Host ""

# Step 1: Check prerequisites
Write-Info "Step 1/8: Checking prerequisites..."

# Check Python
if (-not (Test-CommandExists python)) {
    Write-Error-Message "Python 3.11+ is required but not installed."
    Write-Info "Download from: https://www.python.org/downloads/"
    exit 1
}

$pythonVersion = python --version
Write-Success "Python found: $pythonVersion"

# Check Git
if (-not (Test-CommandExists git)) {
    Write-Error-Message "Git is required but not installed."
    Write-Info "Download from: https://git-scm.com/download/win"
    exit 1
}
Write-Success "Git found"

# Check Node.js
if (-not (Test-CommandExists node)) {
    Write-Warning "Node.js not found. MCP servers will not be available."
    Write-Info "Download from: https://nodejs.org/"
} else {
    $nodeVersion = node --version
    Write-Success "Node.js found: $nodeVersion"
}

# Check Docker
$dockerAvailable = Test-CommandExists docker
if (-not $dockerAvailable) {
    Write-Warning "Docker not found. Using manual installation mode."
} else {
    Write-Success "Docker found"
}

Write-Host ""

# Step 2: Check configuration
Write-Info "Step 2/8: Checking configuration..."

if (-not (Test-Path .env)) {
    Write-Info "Creating .env file from template..."
    Copy-Item .env.example .env
    Write-Warning "Please edit .env file with your API keys."
    Write-Info "Opening .env file in notepad..."
    notepad .env
    Write-Info "Press Enter when ready to continue..."
    Read-Host
}

Write-Success "Configuration file found"
Write-Host ""

# Step 3: Choose installation mode
Write-Info "Step 3/8: Selecting installation mode..."

if (-not $Docker -and -not $Manual) {
    if ($dockerAvailable) {
        Write-Host "Choose installation mode:"
        Write-Host "  1) Docker (recommended - easiest)"
        Write-Host "  2) Manual (more control)"
        $choice = Read-Host "Enter choice [1-2]"
        
        if ($choice -eq "1") {
            $Docker = $true
        } else {
            $Manual = $true
        }
    } else {
        $Manual = $true
    }
}

Write-Host ""

# Step 4: Installation
if ($Docker) {
    Write-Info "Step 4/8: Docker installation selected..."
    
    # Check docker-compose
    if (-not (Test-CommandExists docker-compose)) {
        Write-Error-Message "docker-compose is required but not installed."
        exit 1
    }
    
    Write-Info "Building Docker images..."
    docker-compose build
    Write-Success "Docker images built"
    
    Write-Info "Starting services..."
    docker-compose up -d
    Write-Success "Services started"
    
    Write-Info "Waiting for services to be healthy..."
    Start-Sleep -Seconds 10
    
    # Check health
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            Write-Success "Application is running!"
        }
    } catch {
        Write-Error-Message "Application failed to start. Check logs: docker-compose logs"
        exit 1
    }
    
    Write-Host ""
    Write-Success "Docker installation complete!"
    Write-Info "Access the platform at: http://localhost:8000"
    Write-Info "View logs: docker-compose logs -f"
    Write-Info "Stop services: docker-compose down"
    
} else {
    Write-Info "Step 4/8: Manual installation selected..."
    
    # Create virtual environment
    Write-Info "Creating Python virtual environment..."
    python -m venv venv
    Write-Success "Virtual environment created"
    
    # Activate virtual environment
    Write-Info "Activating virtual environment..."
    .\venv\Scripts\Activate.ps1
    Write-Success "Virtual environment activated"
    
    # Upgrade pip
    Write-Info "Upgrading pip..."
    python -m pip install --upgrade pip | Out-Null
    
    # Install Python dependencies
    Write-Info "Installing Python dependencies (this may take a few minutes)..."
    pip install -r requirements.txt
    Write-Success "Python dependencies installed"
    
    # Install Node.js dependencies
    if (Test-CommandExists npm) {
        Write-Info "Installing Node.js dependencies..."
        npm install
        Write-Success "Node.js dependencies installed"
        
        # Install MCP servers
        Write-Info "Installing MCP servers..."
        npm install @modelcontextprotocol/server-brave-search
        npm install @modelcontextprotocol/server-filesystem
        npm install @modelcontextprotocol/server-github
        Write-Success "MCP servers installed"
    }
    
    Write-Host ""
    
    # Step 5: Check Redis
    Write-Info "Step 5/8: Checking Redis..."
    
    # Try to connect to Redis
    try {
        if (Test-CommandExists redis-cli) {
            $redisTest = redis-cli ping
            if ($redisTest -eq "PONG") {
                Write-Success "Redis is running"
            } else {
                Write-Warning "Redis is not responding"
            }
        } else {
            Write-Warning "Redis not found."
            Write-Info "Install Memurai (Redis for Windows): https://www.memurai.com/get-memurai"
            Write-Info "Or install Redis in WSL2: wsl sudo service redis-server start"
        }
    } catch {
        Write-Warning "Could not connect to Redis"
    }
    
    Write-Host ""
    
    # Step 6: Initialize database
    Write-Info "Step 6/8: Initializing database..."
    
    if (Test-Path scripts\init_db.py) {
        python scripts\init_db.py
        Write-Success "Database initialized"
    } else {
        Write-Warning "Database initialization script not found. Skipping..."
    }
    
    Write-Host ""
    
    # Step 7: Verify API keys
    Write-Info "Step 7/8: Verifying API keys..."
    
    $envContent = Get-Content .env
    $keysSet = 0
    $keysTotal = 3
    
    function Test-ApiKey {
        param($KeyName)
        $line = $envContent | Where-Object { $_ -match "^$KeyName=" }
        if ($line -and $line -notmatch "your_.*_here") {
            Write-Success "$KeyName is set"
            return $true
        } else {
            Write-Warning "$KeyName is not set"
            return $false
        }
    }
    
    if (Test-ApiKey "GROQ_API_KEY") { $keysSet++ }
    if (Test-ApiKey "GEMINI_API_KEY") { $keysSet++ }
    if (Test-ApiKey "GITHUB_TOKEN") { $keysSet++ }
    
    Write-Host ""
    Write-Info "API keys configured: $keysSet/$keysTotal"
    
    if ($keysSet -eq 0) {
        Write-Error-Message "No API keys configured! Please edit .env file."
        Write-Info "Get free API keys from:"
        Write-Host "  - Groq: https://console.groq.com/ (FREE, unlimited)"
        Write-Host "  - Gemini: https://ai.google.dev/ (FREE tier)"
        Write-Host "  - GitHub: https://github.com/settings/tokens"
        exit 1
    }
    
    Write-Host ""
    
    # Step 8: Configure Windows Firewall
    Write-Info "Step 8/8: Configuring Windows Firewall..."
    
    try {
        $firewallRule = Get-NetFirewallRule -DisplayName "YMERA Platform" -ErrorAction SilentlyContinue
        if (-not $firewallRule) {
            Write-Info "Adding firewall rule..."
            New-NetFirewallRule -DisplayName "YMERA Platform" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow | Out-Null
            Write-Success "Firewall rule added"
        } else {
            Write-Success "Firewall rule already exists"
        }
    } catch {
        Write-Warning "Could not add firewall rule. You may need to run as Administrator."
        Write-Info "Or add manually: Control Panel → Windows Defender Firewall → Advanced Settings → Inbound Rules"
    }
    
    Write-Host ""
    
    # Start application
    Write-Info "Starting application..."
    
    # Create directories
    New-Item -ItemType Directory -Force -Path logs | Out-Null
    New-Item -ItemType Directory -Force -Path tmp | Out-Null
    
    # Start MCP server
    if (Test-Path mcp-server\server.js) {
        Write-Info "Starting MCP server..."
        Start-Process -FilePath "node" -ArgumentList "mcp-server\server.js" -RedirectStandardOutput "logs\mcp-server.log" -RedirectStandardError "logs\mcp-server-error.log"
        Write-Success "MCP server started"
    }
    
    # Start main application
    Write-Info "Starting YMERA platform..."
    Start-Process -FilePath "uvicorn" -ArgumentList "src.core.api:app --host 0.0.0.0 --port 8000" -RedirectStandardOutput "logs\app.log" -RedirectStandardError "logs\app-error.log"
    Write-Success "YMERA platform started"
    
    # Wait for application to start
    Write-Info "Waiting for application to start..."
    Start-Sleep -Seconds 5
    
    # Test health
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            Write-Success "Application is running!"
        }
    } catch {
        Write-Error-Message "Application failed to start. Check logs: Get-Content logs\app.log -Wait"
        exit 1
    }
}

# Success message
Write-Host ""
Write-Host @"

╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║                 🎉 SETUP COMPLETE! 🎉                       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

"@ -ForegroundColor Green

Write-Success "YMERA Multi-Agent Platform is now running!"
Write-Host ""
Write-Info "Access the platform:"
Write-Host "  - Web UI: http://localhost:8000"
Write-Host "  - API Docs: http://localhost:8000/docs"
Write-Host "  - Health Check: http://localhost:8000/health"
Write-Host ""

# Get local IP
try {
    $localIP = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -notmatch "Loopback" } | Select-Object -First 1).IPAddress
    Write-Info "Access from iPhone: http://${localIP}:8000"
    Write-Host ""
} catch {
    Write-Info "Run 'ipconfig' to find your local IP address"
    Write-Host ""
}

Write-Info "Useful commands:"
if ($Docker) {
    Write-Host "  - View logs: docker-compose logs -f"
    Write-Host "  - Stop services: docker-compose down"
    Write-Host "  - Restart: docker-compose restart"
} else {
    Write-Host "  - View logs: Get-Content logs\app.log -Wait"
    Write-Host "  - Stop services: .\scripts\stop.ps1"
    Write-Host "  - Restart: .\scripts\restart.ps1"
}

Write-Host ""
Write-Info "Next steps:"
Write-Host "  1. Test a simple request"
Write-Host "  2. Explore the API documentation"
Write-Host "  3. Read the guides in docs\"
Write-Host "  4. Join our community for support"

Write-Host ""
Write-Success "Happy building with YMERA! 🚀"
