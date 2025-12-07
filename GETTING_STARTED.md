# Getting Started with YMERA

Welcome to the YMERA Multi-Agent Workspace Platform! This guide will help you get up and running in minutes.

## 🎯 What is YMERA?

YMERA is an intelligent multi-agent platform that orchestrates AI models to execute complex tasks. It automatically:

- ✅ **Selects the best AI model** for each task
- ✅ **Breaks complex tasks** into manageable phases
- ✅ **Uses multiple models** for different workflow stages
- ✅ **Optimizes for cost** by using free models when possible
- ✅ **Self-heals** with automatic fallback chains
- ✅ **Validates quality** automatically

## 🚀 Quick Start (3 Easy Steps)

### Step 1: Get Free API Keys (5 minutes)

You need at least ONE of these free API keys:

1. **Groq** (Recommended - FREE & Fast!)
   - Visit: https://console.groq.com/
   - Sign up with email
   - Click "API Keys" → "Create API Key"
   - Copy the key

2. **Google Gemini** (Recommended - FREE with generous limits)
   - Visit: https://ai.google.dev/
   - Click "Get API Key"
   - Sign in with Google account
   - Create API key
   - Copy the key

3. **GitHub Token** (Optional but recommended)
   - Visit: https://github.com/settings/tokens
   - Click "Generate new token (classic)"
   - Select scopes: `repo`, `workflow`
   - Generate and copy token

### Step 2: Install YMERA

**Option A: Quick Install (Docker - Recommended)**

```bash
# Clone repository
git clone https://github.com/ymera-mansour/workspace.git
cd workspace

# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
# Windows: notepad .env
# Mac/Linux: nano .env

# Start everything with Docker
docker-compose up -d

# Done! Access at http://localhost:8000
```

**Option B: Windows PowerShell**

```powershell
# Clone repository
git clone https://github.com/ymera-mansour/workspace.git
cd workspace

# Run automated setup
.\scripts\quick_start.ps1

# Follow the prompts
```

**Option C: Mac/Linux/WSL**

```bash
# Clone repository
git clone https://github.com/ymera-mansour/workspace.git
cd workspace

# Run automated setup
chmod +x scripts/quick_start.sh
./scripts/quick_start.sh

# Follow the prompts
```

### Step 3: Test It!

```bash
# Test health endpoint
curl http://localhost:8000/health

# Try a simple request
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum computing in simple terms"}'
```

## 📱 Access from iPhone

1. Find your computer's IP address:
   - **Windows**: Open Command Prompt → `ipconfig`
   - **Mac**: System Preferences → Network
   - **Linux**: `ip addr` or `ifconfig`

2. Look for "IPv4 Address" (example: 192.168.1.100)

3. On your iPhone, open Safari and go to:
   ```
   http://192.168.1.100:8000
   ```

4. Add to Home Screen:
   - Tap Share button (square with arrow)
   - Scroll down and tap "Add to Home Screen"
   - Name it "YMERA"
   - Now you have an app icon!

## 🎓 Your First Task

Let's create a simple Python function:

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a Python function that calculates fibonacci numbers with memoization",
    "user_id": "me"
  }'
```

The platform will:
1. Analyze the request
2. Select the best model (probably Codestral for code)
3. Generate high-quality code
4. Validate the output
5. Return the result

## 📚 Next Steps

### Learn the Basics
1. [AI Providers Guide](docs/guides/ai_providers_complete_guide.md) - Configure all providers
2. [MCP Tools](docs/guides/mcp_complete_guide.md) - Add powerful tools
3. [Workflow Guide](docs/guides/workflow_orchestration.md) - Create complex workflows

### Configuration
4. [Model Selection](docs/guides/model_selection.md) - Choose the right models
5. [Cost Optimization](docs/guides/cost_optimization.md) - Minimize expenses
6. [Security Setup](docs/guides/security_best_practices.md) - Secure your deployment

### Advanced Features
7. [Multi-Model Execution](docs/architecture/multi_model_documentation.md) - Advanced workflows
8. [Custom Agents](docs/guides/custom_agents.md) - Build your own agents
9. [Performance Tuning](docs/architecture/performance_benchmarking.md) - Optimize speed

## 💡 Example Use Cases

### Code Generation
```json
{
  "prompt": "Create a REST API with authentication using FastAPI",
  "agent": "coding_agent"
}
```

### Data Analysis
```json
{
  "prompt": "Analyze this sales data and provide insights",
  "agent": "analysis_agent",
  "data": "..."
}
```

### Research
```json
{
  "prompt": "Research the latest trends in AI and summarize",
  "agent": "research_agent"
}
```

### Documentation
```json
{
  "prompt": "Generate API documentation from this code",
  "agent": "documentation_agent",
  "code": "..."
}
```

## 🆓 Free Resources Used

YMERA is optimized to use free resources:

### Completely Free Models
- **Groq**: Llama 3.1 70B (unlimited!)
- **OpenRouter**: 40+ free models
- **HuggingFace**: Various open-source models

### Free Tier Models
- **Gemini 1.5 Flash**: 15 requests/minute
- **Gemini 1.5 Pro**: 2 requests/minute
- **Mistral Small**: Limited free credits

### Free Tools
- **Brave Search**: 2,000 queries/month
- **GitHub API**: Generous limits with GitHub Pro
- **Local Tools**: Filesystem, PostgreSQL, etc.

## ⚙️ Configuration Tips

### Minimum Setup (.env)
```bash
# Just add these for basic functionality
GROQ_API_KEY=your_groq_key_here
GEMINI_API_KEY=your_gemini_key_here

HOST=0.0.0.0  # For iPhone access
PORT=8000
```

### Recommended Setup (.env)
```bash
# AI Providers
GROQ_API_KEY=your_groq_key_here
GEMINI_API_KEY=your_gemini_key_here
OPENROUTER_API_KEY=your_openrouter_key_here

# GitHub Pro (optional but useful)
GITHUB_TOKEN=your_github_token_here

# MCP Tools (optional)
BRAVE_API_KEY=your_brave_key_here

# Server
HOST=0.0.0.0
PORT=8000

# Strategy (recommended)
DEFAULT_STRATEGY=multi_model
ENABLE_MULTI_MODEL=true
```

### Full Setup
See [.env.example](.env.example) for all configuration options.

## 🐛 Troubleshooting

### Can't start the server?

**Python not found:**
```bash
# Install Python 3.11+
# Windows: https://www.python.org/downloads/
# Mac: brew install python@3.11
# Ubuntu: sudo apt install python3.11
```

**Redis connection failed:**
```bash
# Install Redis
# Windows: Install Memurai from https://www.memurai.com/
# Mac: brew install redis && brew services start redis
# Ubuntu: sudo apt install redis-server && sudo service redis-server start
```

**Port 8000 in use:**
```bash
# Change port in .env
PORT=8001

# Or kill existing process
# Windows: netstat -ano | findstr :8000
# Mac/Linux: lsof -ti:8000 | xargs kill
```

### Can't access from iPhone?

1. **Check both devices are on same WiFi**
2. **Verify firewall allows port 8000**
   ```powershell
   # Windows PowerShell (as Administrator)
   New-NetFirewallRule -DisplayName "YMERA" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
   ```
3. **Check HOST is set to 0.0.0.0 in .env**
4. **Disable AP Isolation on router** (if enabled)

### API key errors?

```bash
# Verify keys are set correctly
# Windows:
type .env | findstr API_KEY

# Mac/Linux:
grep API_KEY .env

# Test individual providers
python scripts/test_providers.py
```

## 📞 Getting Help

- **Documentation**: Browse [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/ymera-mansour/workspace/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ymera-mansour/workspace/discussions)
- **Examples**: Check [examples/](examples/) directory

## 🎉 Success!

If you see this, you're ready to build!

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "services": {
    "redis": "connected",
    "ai_providers": ["groq", "gemini"],
    "mcp": "active"
  }
}
```

## 💪 What Can You Build?

- **Code Generators**: Automate repetitive coding tasks
- **Research Assistants**: Gather and synthesize information
- **Data Analysts**: Process and visualize data
- **Content Creators**: Generate articles, documentation
- **Task Automators**: Chain multiple AI operations
- **Custom Workflows**: Design your own agent pipelines

## 🚀 Advanced Usage

Once you're comfortable, explore:

1. **Custom Agents**: Create specialized agents for your needs
2. **Multi-Model Workflows**: Combine multiple models strategically
3. **Tool Integration**: Add custom MCP tools
4. **Production Deployment**: Scale to production workloads
5. **Cost Optimization**: Fine-tune model selection
6. **Quality Benchmarking**: Measure and improve output quality

## 📖 Learn More

- [Complete README](README.md) - Full documentation
- [API Documentation](http://localhost:8000/docs) - Interactive API docs
- [Architecture Guide](docs/architecture/) - System design
- [Deployment Guide](docs/guides/local_deployment_windows.md) - Production setup

---

**Welcome to the YMERA community! Let's build something amazing together.** 🚀

*Questions? Open an issue or start a discussion on GitHub!*
