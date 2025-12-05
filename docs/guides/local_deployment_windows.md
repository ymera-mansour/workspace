# Complete Local Deployment Guide - Windows + iPhone

## 🎯 Overview

This guide will help you deploy the YMERA Multi-Agent Workspace Platform on your Windows desktop and access it from your iPhone on the same network.

## 📋 Prerequisites

### Required Software
- **Windows 10/11** (64-bit)
- **Python 3.11+**
- **Git for Windows**
- **Redis** (we'll install this)
- **Node.js 18+** (for MCP servers)

### Optional but Recommended
- **Docker Desktop for Windows** (easiest setup)
- **WSL2** (Windows Subsystem for Linux)
- **VS Code** (for editing configuration)

### Network Requirements
- Windows PC and iPhone on same WiFi network
- Windows Firewall configured to allow inbound connections

## 🚀 Quick Start (15 Minutes)

### Option 1: Docker Installation (Recommended)

#### Step 1: Install Docker Desktop

1. Download Docker Desktop from: https://www.docker.com/products/docker-desktop/
2. Run the installer
3. Enable WSL2 backend when prompted
4. Restart Windows
5. Start Docker Desktop

#### Step 2: Clone Repository

```powershell
# Open PowerShell as Administrator
cd C:\
mkdir Projects
cd Projects
git clone https://github.com/ymera-mansour/workspace.git
cd workspace
```

#### Step 3: Configure Environment

```powershell
# Copy environment template
copy .env.example .env

# Edit .env with your API keys
notepad .env
```

**Required API Keys** (Get from these free services):
```
GROQ_API_KEY=your_groq_key  # https://console.groq.com/ - FREE!
GEMINI_API_KEY=your_gemini_key  # https://ai.google.dev/ - FREE!
GITHUB_TOKEN=your_github_token  # https://github.com/settings/tokens
```

**Important**: Set these values in .env:
```
HOST=0.0.0.0  # Allow external connections
PORT=8000
```

#### Step 4: Start Services

```powershell
# Build and start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

#### Step 5: Verify Installation

```powershell
# Test from Windows
curl http://localhost:8000/health

# Should return: {"status": "healthy", "version": "1.0.0"}
```

#### Step 6: Get Your PC's IP Address

```powershell
# Find your local IP
ipconfig

# Look for "IPv4 Address" under your WiFi adapter
# Example: 192.168.1.100
```

#### Step 7: Configure Windows Firewall

```powershell
# Run as Administrator
# Allow inbound connections on port 8000
New-NetFirewallRule -DisplayName "YMERA Platform" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
```

#### Step 8: Access from iPhone

1. Open Safari on your iPhone
2. Go to: `http://YOUR_PC_IP:8000` (e.g., http://192.168.1.100:8000)
3. You should see the YMERA dashboard
4. Tap Share → Add to Home Screen for app-like experience

**Success!** 🎉 You're now running the platform!

---

## Option 2: Manual Installation (More Control)

### Step 1: Install Python

1. Download Python 3.11+ from: https://www.python.org/downloads/
2. Run installer
3. **Important**: Check "Add Python to PATH"
4. Verify installation:

```powershell
python --version
# Should show: Python 3.11.x or higher

pip --version
# Should show pip version
```

### Step 2: Install Redis for Windows

#### Option A: Using Memurai (Redis for Windows)

1. Download Memurai: https://www.memurai.com/get-memurai
2. Install with default settings
3. Start Memurai service:

```powershell
# Start service
net start Memurai

# Verify it's running
redis-cli ping
# Should return: PONG
```

#### Option B: Using WSL2 + Ubuntu

```powershell
# Install WSL2
wsl --install

# Restart Windows

# Install Redis in Ubuntu
wsl
sudo apt update
sudo apt install redis-server
sudo service redis-server start

# Test
redis-cli ping
```

### Step 3: Install Node.js

1. Download Node.js 18+ LTS from: https://nodejs.org/
2. Run installer with default settings
3. Verify installation:

```powershell
node --version
# Should show: v18.x.x or higher

npm --version
# Should show npm version
```

### Step 4: Clone and Setup Project

```powershell
# Create project directory
cd C:\
mkdir Projects
cd Projects

# Clone repository
git clone https://github.com/ymera-mansour/workspace.git
cd workspace

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies (for MCP servers)
npm install
```

### Step 5: Install MCP Servers

```powershell
# Install all MCP servers
npm install @modelcontextprotocol/server-brave-search
npm install @modelcontextprotocol/server-filesystem
npm install @modelcontextprotocol/server-github
npm install @modelcontextprotocol/server-postgres
npm install @modelcontextprotocol/server-puppeteer
```

### Step 6: Configure Environment

```powershell
# Copy environment template
copy .env.example .env

# Edit with VS Code or Notepad
code .env
# or
notepad .env
```

**Essential Configuration**:
```bash
# Server (MUST set to 0.0.0.0 for iPhone access)
HOST=0.0.0.0
PORT=8000

# Free AI Providers
GROQ_API_KEY=your_groq_key_here
GEMINI_API_KEY=your_gemini_key_here
OPENROUTER_API_KEY=your_openrouter_key_here

# GitHub Pro
GITHUB_TOKEN=your_github_token_here

# Redis
REDIS_URL=redis://localhost:6379
```

### Step 7: Initialize Database

```powershell
# Create database tables
python scripts/init_db.py

# Run migrations
python scripts/migrate.py
```

### Step 8: Start Services

#### Terminal 1: Start Redis (if not running as service)
```powershell
redis-server
```

#### Terminal 2: Start MCP Server
```powershell
cd mcp-server
node server.js
```

#### Terminal 3: Start YMERA Platform
```powershell
# Activate virtual environment
.\venv\Scripts\activate

# Start server
python src/core/agent_platform.py

# Or use uvicorn
uvicorn src.core.api:app --host 0.0.0.0 --port 8000 --reload
```

### Step 9: Verify Installation

```powershell
# Test health endpoint
curl http://localhost:8000/health

# Test AI completion
curl -X POST http://localhost:8000/v1/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"Hello, how are you?\"}"
```

### Step 10: Configure Windows Firewall

**Using GUI**:
1. Open Windows Defender Firewall
2. Click "Advanced settings"
3. Click "Inbound Rules" → "New Rule"
4. Select "Port" → Next
5. Select "TCP" → Enter port 8000 → Next
6. Allow the connection → Next
7. Apply to all profiles → Next
8. Name it "YMERA Platform" → Finish

**Using PowerShell** (Run as Administrator):
```powershell
New-NetFirewallRule -DisplayName "YMERA Platform" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
```

### Step 11: Get Your IP and Access from iPhone

```powershell
# Get your local IP address
ipconfig

# Look for "Wireless LAN adapter Wi-Fi"
# Find "IPv4 Address" (e.g., 192.168.1.100)
```

**On your iPhone**:
1. Connect to same WiFi as your Windows PC
2. Open Safari
3. Navigate to: `http://192.168.1.100:8000` (use YOUR IP)
4. Test a request

**Add to Home Screen**:
1. Tap the Share button in Safari
2. Scroll and tap "Add to Home Screen"
3. Name it "YMERA"
4. Tap "Add"
5. Now you have an app icon!

---

## 🔧 Configuration Tips

### Performance Optimization

**For Windows**:
```powershell
# Increase file descriptor limit
# Add to System Environment Variables
ULIMIT_N=4096

# Optimize Python for performance
set PYTHONOPTIMIZE=1
```

**For Redis**:
```bash
# Edit redis.conf (C:\Program Files\Memurai\memurai.conf)
maxmemory 256mb
maxmemory-policy allkeys-lru
```

### Security Best Practices

1. **Use Strong API Keys**: Generate secure keys for JWT
   ```powershell
   # Generate secure key
   python -c "import secrets; print(secrets.token_hex(32))"
   ```

2. **Enable HTTPS** (for production):
   ```powershell
   # Generate self-signed certificate
   openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365
   ```

3. **Restrict CORS**:
   ```bash
   CORS_ORIGINS=http://localhost:3000,http://192.168.1.100:8000
   ```

### Network Configuration

**For Better Performance**:
```powershell
# Optimize TCP settings
netsh int tcp set global autotuninglevel=normal
netsh int tcp set global chimney=enabled
netsh int tcp set global dca=enabled
netsh int tcp set global netdma=enabled
```

**Check Port Availability**:
```powershell
# Check if port 8000 is in use
netstat -ano | findstr :8000

# Kill process if needed (replace PID with actual process ID)
taskkill /PID 1234 /F
```

---

## 📱 iPhone Setup Details

### Safari Configuration

1. **Enable JavaScript**: Settings → Safari → Advanced → JavaScript (ON)
2. **Allow Popups**: Settings → Safari → Block Pop-ups (OFF for this site)
3. **Clear Cache**: Settings → Safari → Clear History and Website Data

### Network Troubleshooting

#### Can't Connect from iPhone?

1. **Check WiFi**: Ensure both devices on same network
   - Windows: `ipconfig` → Look for "Default Gateway"
   - iPhone: Settings → WiFi → Tap (i) → Check "Router" matches

2. **Verify Firewall**:
   ```powershell
   # Test if port is open
   Test-NetConnection -ComputerName localhost -Port 8000
   ```

3. **Check Server is Running**:
   ```powershell
   # From Windows
   curl http://localhost:8000/health
   
   # From Windows using local IP
   curl http://192.168.1.100:8000/health
   ```

4. **Router Configuration**:
   - Some routers have "AP Isolation" enabled
   - Log into router settings and disable "AP Isolation"
   - Or enable "Allow clients to communicate"

### Mobile UI Optimization

The platform automatically detects mobile browsers and adjusts the UI. You can force mobile mode:

```bash
# In .env
ENABLE_MOBILE_UI=true
MOBILE_VIEWPORT_WIDTH=390  # iPhone default
```

---

## 🧪 Testing the Installation

### Basic Health Check

```powershell
# Test health endpoint
curl http://localhost:8000/health

# Expected response:
# {
#   "status": "healthy",
#   "version": "1.0.0",
#   "services": {
#     "redis": "connected",
#     "mcp": "active",
#     "ai_providers": ["groq", "gemini"]
#   }
# }
```

### Test AI Completion

```powershell
# Create test request
$body = @{
    prompt = "Write a Python function to calculate fibonacci"
    user_id = "test_user"
} | ConvertTo-Json

# Send request
Invoke-RestMethod -Uri "http://localhost:8000/v1/completions" -Method Post -Body $body -ContentType "application/json"
```

### Test from iPhone

1. Open Safari on iPhone
2. Navigate to `http://YOUR_PC_IP:8000`
3. You should see the dashboard
4. Try a test request from the web interface

---

## 🔄 Daily Operations

### Starting the Platform

**Docker Method**:
```powershell
cd C:\Projects\workspace
docker-compose up -d
```

**Manual Method**:
```powershell
# Terminal 1: Redis (if not service)
redis-server

# Terminal 2: MCP Server
cd C:\Projects\workspace\mcp-server
node server.js

# Terminal 3: YMERA
cd C:\Projects\workspace
.\venv\Scripts\activate
python src/core/agent_platform.py
```

### Stopping the Platform

**Docker Method**:
```powershell
docker-compose down
```

**Manual Method**:
```powershell
# Press Ctrl+C in each terminal
```

### Viewing Logs

**Docker Method**:
```powershell
docker-compose logs -f
docker-compose logs -f app  # Specific service
```

**Manual Method**:
```powershell
# Logs are in /var/log/ymera/app.log
Get-Content C:\Projects\workspace\logs\app.log -Wait
```

### Updating the Platform

```powershell
cd C:\Projects\workspace

# Pull latest changes
git pull

# Update dependencies
pip install -r requirements.txt --upgrade
npm install

# Restart services
docker-compose restart  # or restart manually
```

---

## 🐛 Troubleshooting

### Issue: "Python not found"

**Solution**:
```powershell
# Add Python to PATH
$env:Path += ";C:\Users\YourUsername\AppData\Local\Programs\Python\Python311"

# Or reinstall Python with "Add to PATH" checked
```

### Issue: "Redis connection failed"

**Solution**:
```powershell
# Check if Redis is running
redis-cli ping

# Start Redis service
net start Memurai

# Or start Redis manually
redis-server
```

### Issue: "Port 8000 already in use"

**Solution**:
```powershell
# Find what's using port 8000
netstat -ano | findstr :8000

# Kill the process (replace 1234 with actual PID)
taskkill /PID 1234 /F

# Or use a different port
# In .env: PORT=8001
```

### Issue: "Can't access from iPhone"

**Solutions**:

1. **Verify server is accessible from Windows**:
   ```powershell
   curl http://YOUR_PC_IP:8000/health
   ```

2. **Check Firewall**:
   ```powershell
   # Disable temporarily to test
   netsh advfirewall set allprofiles state off
   
   # Re-enable after testing
   netsh advfirewall set allprofiles state on
   ```

3. **Check Router Settings**:
   - Log into router (usually http://192.168.1.1)
   - Disable "AP Isolation" or "Client Isolation"
   - Enable "Allow clients to communicate"

4. **Try Different Browser**:
   - Use Chrome on iPhone instead of Safari
   - Try Firefox Focus

### Issue: "API Key Invalid"

**Solution**:
```powershell
# Verify API keys are set
Get-Content .env | Select-String "API_KEY"

# Test individual providers
python scripts/test_providers.py
```

---

## 📊 Monitoring & Maintenance

### Resource Usage

```powershell
# Check CPU and Memory
Get-Process | Where-Object {$_.ProcessName -like "*python*"} | Select-Object ProcessName, CPU, WS

# Docker stats
docker stats
```

### Database Maintenance

```powershell
# Backup database
python scripts/backup_db.py

# Clean old logs
python scripts/cleanup_logs.py --days 7
```

### Cost Tracking

```powershell
# View API usage
python scripts/cost_report.py

# Expected output:
# Provider    | Requests | Tokens | Cost
# ------------|----------|--------|------
# Groq        | 1000     | 50000  | $0.00
# Gemini      | 500      | 25000  | $0.00
# Total       | 1500     | 75000  | $0.00
```

---

## 🎓 Next Steps

1. **Configure AI Providers**: [AI Providers Guide](./ai_providers_complete_guide.md)
2. **Set up MCP Tools**: [MCP Guide](./mcp_complete_guide.md)
3. **Create Workflows**: [Workflow Guide](./workflow_guide.md)
4. **Explore Examples**: Check `examples/` directory

---

## 📞 Getting Help

- **Documentation**: [docs/](../../)
- **GitHub Issues**: [Report a bug](https://github.com/ymera-mansour/workspace/issues)
- **Community**: [Discussions](https://github.com/ymera-mansour/workspace/discussions)

---

**Success Tips**:
- Start with free providers (Groq, Gemini)
- Use Docker for easier setup
- Keep your API keys secure
- Monitor costs regularly
- Join our community for support

**Enjoy building with YMERA!** 🚀
