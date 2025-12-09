# YMERA AI Platform - Setup Guide

Complete setup guide for the YMERA AI Platform with 53 AI models, 120+ tools, and LangChain framework.

## Quick Start (15 minutes)

```bash
# 1. Clone repository
git clone <repository-url>
cd workspace

# 2. Run automated setup
chmod +x setup.sh
./setup.sh

# 3. Configure API keys
nano .env  # Add your API keys

# 4. Activate environment
source venv/bin/activate

# 5. Run the platform
python main.py
```

## Prerequisites

- **Python**: 3.8 or higher
- **Node.js**: v18 or higher (for MCP tools)
- **Git**: Latest version
- **Operating System**: Linux, macOS, or Windows with WSL

## Detailed Setup Instructions

### Step 1: Environment Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# OR
.\venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Step 2: Install Python Dependencies

```bash
# Install all dependencies (10-15 minutes)
pip install -r requirements.txt

# Download NLP models
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
python -c "import nltk; nltk.download('all')"
```

### Step 3: Configure API Keys

```bash
# Copy template
cp .env.template .env

# Edit and add your API keys
nano .env
```

#### Required API Keys (FREE)

1. **Gemini** (Google): https://makersuite.google.com/app/apikey
   - Free tier: 60 requests/minute
   
2. **Mistral AI**: https://console.mistral.ai/
   - Free tier: 1B tokens/month
   
3. **Groq**: https://console.groq.com/
   - Free tier: 14,400 requests/day
   
4. **OpenRouter**: https://openrouter.ai/
   - Free tier: Multiple free models
   
5. **HuggingFace**: https://huggingface.co/settings/tokens
   - Free tier: 20 requests/minute
   
6. **Cohere**: https://dashboard.cohere.com/
   - Free tier: 100 calls/minute
   
7. **Together AI**: https://api.together.xyz/
   - Free tier: $25 monthly credits

#### Optional API Keys (PAID)

8. **Anthropic Claude**: https://console.anthropic.com/
   - Paid service (optional)
   
9. **Replicate**: https://replicate.com/
   - Paid service (optional)

### Step 4: Install MCP Tools (Optional, 35 minutes)

```bash
# Make script executable
chmod +x install_mcp_tools.sh

# Run installation
./install_mcp_tools.sh
```

MCP tools provide additional capabilities:
- Code execution (Python, Node.js)
- File system operations
- Git/GitHub integration
- Database access (PostgreSQL, SQLite, Redis)
- Container management (Docker, Kubernetes)
- Testing frameworks (Jest, Pytest)
- And more...

### Step 5: Validate Configuration

```bash
# Test configuration loading
python config_loader.py

# Test provider connections
python providers_init.py
```

## Configuration Files

### `.env` - Environment Variables

Contains all API keys and sensitive configuration. **Never commit this file!**

```bash
# AI Provider Keys
GEMINI_API_KEY_1=your_key_here
MISTRAL_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
# ... more keys
```

### `config.yaml` - System Configuration

Main configuration file for all system components:

- AI providers and models
- Routing strategies
- Caching configuration
- LangChain settings
- MCP tools
- ML/Learning settings
- Infrastructure components

Edit this file to customize system behavior.

## System Components

### AI Providers (9 total)

| Provider | Models | Free Tier | Use Cases |
|----------|--------|-----------|-----------|
| Gemini | 4 | 60 RPM | General, Vision, Reasoning |
| Mistral | 6 | 1B tokens/mo | Code, Multimodal |
| Groq | 6 | 14.4K RPD | Ultra-fast inference |
| OpenRouter | 14 | Various | Unified API, variety |
| HuggingFace | 12 | 20 RPM | Code, Open models |
| Cohere | 3 | 100 RPM | Embeddings, RAG |
| Together AI | 6 | $25 credits | Fast inference |
| Claude | 2 | Paid | Advanced reasoning |
| Replicate | Many | Paid | Specialized models |

### MCP Tools (18 total)

- **Critical Infrastructure** (7): Python, Node.js, Filesystem, Git, PostgreSQL, SQLite, Redis
- **Development Tools** (6): Docker, Kubernetes, Jest, Pytest, HTTP, Search
- **Specialized Tools** (5): Prometheus, Elasticsearch, Email, Slack, S3

### ML/Learning Tools (15 total)

- **Frameworks** (5): TensorFlow, PyTorch, Scikit-learn, XGBoost, Keras
- **Training** (3): Optuna, Ray Tune, Weights & Biases
- **Continuous Learning** (3): MLflow, DVC, Auto-sklearn
- **Monitoring** (4): TensorBoard, Evidently AI, WhyLogs, Neptune.ai

### Infrastructure Tools (25 total)

- **Security** (5): Bandit, Safety, OWASP ZAP, Trivy, Semgrep
- **NLP** (5): spaCy, NLTK, Transformers, Gensim, TextBlob
- **File Management** (5): PyPDF2, python-docx, openpyxl, Pillow, python-magic
- **Communication** (5): Celery, RabbitMQ, ZeroMQ, gRPC, Socket.IO
- **Quality** (5): pytest, coverage.py, pylint, SonarQube, Locust

## Usage Examples

### Basic AI Inference

```python
from config_loader import get_config
from providers_init import AIProvidersManager

# Load configuration
config = get_config()

# Initialize providers
providers = AIProvidersManager(config)

# Use a provider
gemini = providers.get_provider("gemini")
# ... make API calls
```

### LangChain RAG Pipeline

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings

# Initialize components
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
embeddings = CohereEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# Create RAG chain
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

# Query
result = qa_chain.run("Your question here")
```

## Monitoring & Maintenance

### Health Check

```bash
# Check system health
python providers_init.py

# Check MCP tools
npm list -g | grep mcp
```

### View Logs

```bash
# Application logs
tail -f logs/ymera.log

# MLflow UI
mlflow ui --port 5000

# TensorBoard
tensorboard --logdir=./logs
```

### Update Dependencies

```bash
# Update Python packages
pip install --upgrade -r requirements.txt

# Update MCP tools
npm update -g
```

## Troubleshooting

### API Key Issues

**Problem**: "API key not configured" errors

**Solution**:
1. Check `.env` file has all required keys
2. Verify keys are valid (no extra spaces)
3. Restart application after changing `.env`

### Import Errors

**Problem**: `ModuleNotFoundError` for packages

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### MCP Tools Not Working

**Problem**: MCP tools commands fail

**Solution**:
1. Verify Node.js v18+ is installed: `node --version`
2. Reinstall MCP tools: `./install_mcp_tools.sh`
3. Check global npm packages: `npm list -g`

### Performance Issues

**Problem**: Slow responses

**Solution**:
1. Enable caching in `config.yaml`
2. Use faster models (Groq, Together AI)
3. Check network connection
4. Review rate limits

## Cost Management

### Base System (FREE - $0/month)

All required components are FREE:
- 9 AI providers (7 FREE, 2 optional paid)
- 51 FREE AI models
- 18 FREE MCP tools
- 15 FREE ML/Learning tools
- 25 FREE Infrastructure tools

### Optional Paid Add-ons

- **Claude**: $10-50/month (advanced reasoning)
- **Replicate**: $1-20/month (specialized models)

### Tips to Stay Free

1. Use FREE models only (see `config.yaml`)
2. Monitor usage with provider dashboards
3. Enable caching to reduce API calls
4. Use local tools when possible
5. Implement rate limiting

## Next Steps

1. **Explore Documentation**: Read the comprehensive .md files
2. **Test Providers**: Run sample queries to each provider
3. **Configure Routing**: Customize intelligent routing in `config.yaml`
4. **Enable LangChain**: Set up RAG pipelines for your use case
5. **Deploy**: Use Docker or cloud platforms for production

## Support

- **Documentation**: See all .md files in repository
- **Issues**: Create GitHub issue for bugs
- **Configuration**: Review `config.yaml` comments
- **Examples**: Check provider documentation

## Security Best Practices

1. **Never commit `.env` file** - Already in `.gitignore`
2. **Rotate API keys regularly** - Update in `.env`
3. **Use environment-specific configs** - Separate dev/prod
4. **Enable security scanning** - Bandit, Safety, Trivy
5. **Monitor usage** - Track API calls and costs

## Performance Optimization

1. **Enable 3-tier caching** - Memory → Redis → Cloud
2. **Use fastest models** - Groq for speed, Gemini for quality
3. **Implement routing** - Complexity-based model selection
4. **Batch requests** - Group similar queries
5. **Monitor metrics** - Use Prometheus + Grafana

---

**Total Setup Time**: ~60 minutes (including MCP tools)

**Base Cost**: $0/month ✅

**System Status**: Production-ready with 120+ tools, 53 AI models, and LangChain framework
