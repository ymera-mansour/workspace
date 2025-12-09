# Complete Setup & Deployment Guide
## Multi-Agent Platform with Mistral, Gemini, Groq, HuggingFace

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start (5 Minutes)](#quick-start)
3. [Manual Setup](#manual-setup)
4. [Configuration](#configuration)
5. [Testing](#testing)
6. [Production Deployment](#production-deployment)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required

- **Python 3.11+** 
- **Redis 7+** (for caching)
- **Node.js 18+** (for MCP server)
- **Docker & Docker Compose** (recommended)

### API Keys Needed

Get your free API keys from:

1. **Mistral AI**: https://console.mistral.ai/
2. **Google AI (Gemini)**: https://ai.google.dev/
3. **Groq**: https://console.groq.com/ (FREE - very fast!)
4. **HuggingFace**: https://huggingface.co/settings/tokens
5. **GitHub**: https://github.com/settings/tokens (use your Pro account!)

---

## Quick Start (5 Minutes)

### Option 1: Docker (Recommended)

```bash
# 1. Clone/create project directory
mkdir multi-agent-platform && cd multi-agent-platform

# 2. Create .env file with your API keys
cat > .env << 'EOF'
MISTRAL_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
HF_API_KEY=your_key_here
GITHUB_TOKEN=your_token_here
REDIS_URL=redis://redis:6379
MCP_SERVER_URL=http://mcp-server:3000/mcp
EOF

# 3. Start everything
docker-compose up -d

# 4. Check status
curl http://localhost:8000/health

# 5. Test a request
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "prompt": "Write a Python function to calculate fibonacci numbers"
  }'
```

### Option 2: Local Development

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Start Redis (in separate terminal)
redis-server

# 3. Start MCP Server (in separate terminal)
cd mcp-server
npm install
node server.js

# 4. Set environment variables
export MISTRAL_API_KEY="your_key"
export GEMINI_API_KEY="your_key"
export GROQ_API_KEY="your_key"
export HF_API_KEY="your_key"
export GITHUB_TOKEN="your_token"

# 5. Run the platform
python agent_platform.py
```

---

## Manual Setup

### Step 1: Project Structure

Create this directory structure:

```
multi-agent-platform/
├── agent_platform.py          # Main orchestrator (from artifact)
├── api_server.py              # FastAPI server
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables
├── docker-compose.yml         # Docker setup
├── Dockerfile                 # App container
├── mcp-server/
│   ├── server.js             # MCP server
│   ├── package.json          # Node dependencies
│   └── Dockerfile            # MCP container
├── tests/
│   ├── test_basic.py
│   ├── test_agents.py
│   └── test_integration.py
└── logs/                      # Application logs
```

### Step 2: Install Dependencies

```bash
# Python packages
pip install -r requirements.txt

# Node.js packages (for MCP server)
cd mcp-server
npm install
cd ..
```

### Step 3: Configure Environment

Edit `.env` file:

```bash
# Essential Keys
MISTRAL_API_KEY=sk-...          # Get from https://console.mistral.ai/
GEMINI_API_KEY=AI...            # Get from https://ai.google.dev/
GROQ_API_KEY=gsk_...            # Get from https://console.groq.com/
HF_API_KEY=hf_...               # Get from https://huggingface.co/

# GitHub Pro (Your existing token)
GITHUB_TOKEN=ghp_...            # Use your GitHub Pro token

# Redis
REDIS_URL=redis://localhost:6379
REDIS_PASSWORD=                 # Optional
REDIS_DB=0

# MCP Server
MCP_SERVER_URL=http://localhost:3000/mcp

# Optional: Brave Search (for better web search)
BRAVE_API_KEY=                  # Get from https://brave.com/search/api/

# System Limits
MAX_REQUESTS_PER_MINUTE=100
CACHE_TTL_SECONDS=3600
DAILY_COST_LIMIT=10.00
```

### Step 4: Verify Setup

```bash
# Test Redis connection
redis-cli ping
# Should return: PONG

# Test MCP server
curl http://localhost:3000/health
# Should return: {"status":"healthy","timestamp":"..."}

# Test Python imports
python -c "from agent_platform import ProductionOrchestrator; print('✓ Imports OK')"
```

---

## Configuration

### Provider Priority

The system automatically selects providers in this order:

1. **Groq** (FREE, fastest) - for simple tasks
2. **Gemini Flash** (very cheap) - for balanced tasks
3. **Mistral Small** (cheap) - for quality tasks
4. **Gemini Pro / Mistral Large** - for complex tasks

You can override per-agent in `AgentRegistry`:

```python
AgentConfig(
    name="my_agent",
    preferred_provider="best",  # Options: "cheap", "balanced", "best"
    ...
)
```

### Cost Management

Set budget limits in `.env`:

```bash
# Alert when daily cost exceeds $5
ALERT_COST_THRESHOLD=5.00

# Hard stop at $10/day
DAILY_COST_LIMIT=10.00
```

Monitor costs:

```bash
curl http://localhost:8000/v1/stats
```

### Cache Configuration

Adjust cache behavior:

```python
# In agent_platform.py
CACHE_TTL_SECONDS = 3600  # 1 hour (default)

# For frequently changing data
CACHE_TTL_SECONDS = 300   # 5 minutes

# For stable content
CACHE_TTL_SECONDS = 86400 # 24 hours
```

---

## Testing

### Unit Tests

```bash
# Test individual components
python -m pytest tests/test_basic.py -v

# Test all agents
python -m pytest tests/test_agents.py -v

# Test full integration
python -m pytest tests/test_integration.py -v
```

### Test Scripts

#### Test 1: Basic Request

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "prompt": "Explain async/await in Python in 3 sentences"
  }'
```

#### Test 2: Streaming Response

```bash
curl -X POST http://localhost:8000/v1/completions/stream \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "prompt": "Write a story about a robot"
  }'
```

#### Test 3: Specific Agent

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "prompt": "Review this code: def add(a,b): return a+b",
    "agent_name": "code_reviewer"
  }'
```

#### Test 4: Cache Performance

```bash
# First request (cache miss)
time curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test", "prompt": "Hello world"}'

# Second request (cache hit - should be instant)
time curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test", "prompt": "Hello world"}'
```

#### Test 5: GitHub Integration

```python
# test_github.py
import asyncio
from agent_platform import GitHubCopilotAgent
import os

async def test_github():
    github = GitHubCopilotAgent(os.getenv("GITHUB_TOKEN"))
    await github.initialize()
    
    # Search for code
    results = await github.search_code("async def", "python")
    print(f"Found {len(results)} examples")
    
    for result in results[:3]:
        print(f"- {result['repository']['full_name']}")
    
    await github.close()

asyncio.run(test_github())
```

### Load Testing

```bash
# Install Apache Bench
sudo apt-get install apache2-utils

# Test 100 requests with 10 concurrent
ab -n 100 -c 10 -p request.json -T application/json \
   http://localhost:8000/v1/completions

# request.json:
# {"user_id": "load_test", "prompt": "Hello"}
```

---

## Production Deployment

### Option 1: Docker Swarm (Small Scale)

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml agents

# Scale orchestrator
docker service scale agents_orchestrator=5

# Monitor
docker service logs -f agents_orchestrator
```

### Option 2: Kubernetes (Large Scale)

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-orchestrator
spec:
  replicas: 10
  selector:
    matchLabels:
      app: orchestrator
  template:
    metadata:
      labels:
        app: orchestrator
    spec:
      containers:
      - name: orchestrator
        image: your-registry/agent-orchestrator:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          value: redis://redis-service:6379
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: orchestrator-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: orchestrator
```

Deploy:

```bash
kubectl apply -f k8s-deployment.yaml
kubectl get pods
kubectl logs -f deployment/agent-orchestrator
```

### Option 3: AWS Lambda (Serverless)

```python
# lambda_handler.py
from agent_platform import ProductionOrchestrator
import json

orchestrator = None

def lambda_handler(event, context):
    global orchestrator
    
    if orchestrator is None:
        orchestrator = ProductionOrchestrator()
        # Initialize synchronously (Lambda supports)
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            orchestrator.initialize()
        )
    
    body = json.loads(event['body'])
    
    result = asyncio.get_event_loop().run_until_complete(
        orchestrator.process_request(
            user_id=body['user_id'],
            prompt=body['prompt']
        )
    )
    
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
```

### Monitoring Setup

#### Prometheus + Grafana

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'agent_platform'
    static_configs:
      - targets: ['orchestrator:8000']
```

```python
# Add to api_server.py
from prometheus_client import Counter, Histogram, generate_latest

REQUEST_COUNT = Counter('requests_total', 'Total requests', ['agent', 'status'])
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency', ['agent'])

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

---

## Troubleshooting

### Common Issues

#### 1. Redis Connection Failed

```bash
# Check Redis is running
redis-cli ping

# Check connection string
echo $REDIS_URL

# Test connection
python -c "import redis; r=redis.from_url('redis://localhost:6379'); print(r.ping())"
```

**Fix**: Ensure Redis is running and URL is correct in `.env`

#### 2. MCP Server Not Responding

```bash
# Check MCP server logs
docker logs agent_mcp

# Test directly
curl http://localhost:3000/health

# Check port is not in use
lsof -i :3000
```

**Fix**: Restart MCP server, check for port conflicts

#### 3. API Key Invalid

```bash
# Verify keys are set
env | grep API_KEY

# Test Groq key directly
curl https://api.groq.com/openai/v1/models \
  -H "Authorization: Bearer $GROQ_API_KEY"
```

**Fix**: Regenerate API keys from provider dashboards

#### 4. Out of Memory

```bash
# Check memory usage
docker stats

# Check Redis memory
redis-cli INFO memory
```

**Fix**: 
- Reduce `CACHE_TTL_SECONDS`
- Limit `MAX_CONVERSATION_HISTORY`
- Increase container memory limits

#### 5. Rate Limit Exceeded

```bash
# Check current usage
curl http://localhost:8000/v1/stats
```

**Fix**:
- Wait for rate limit reset
- Use cheaper providers (Groq is unlimited)
- Add more provider keys for rotation

### Debug Mode

Enable detailed logging:

```python
# In agent_platform.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check logs:

```bash
# Docker
docker logs -f agent_orchestrator

# Local
tail -f logs/agent_platform.log
```

### Performance Issues

If experiencing slow responses:

1. **Check cache hit rate**:
   ```bash
   curl http://localhost:8000/health | jq '.cache.hit_rate'
   ```
   - Should be >70% for good performance

2. **Monitor provider latency**:
   ```bash
   curl http://localhost:8000/v1/stats | jq '.calls_by_provider'
   ```

3. **Use faster providers**:
   - Groq: ~500ms
   - Gemini Flash: ~1000ms
   - Mistral: ~1500ms

---

## Advanced Features

### Adding Custom Agents

```python
# In agent_platform.py, add to register_default_agents()
self.agents[agent.name] = AgentConfig(
    name="my_custom_agent",
    description="Does something specific",
    system_prompt="You are a specialized assistant that...",
    keywords=["keyword1", "keyword2"],
    preferred_provider="cheap",
    requires_tools=False,
    cost_tier=1
)
```

### Adding MCP Tools

```javascript
// In mcp-server/server.js
case 'my_custom_tool':
    result = await handleMyTool(params.arguments);
    break;

async function handleMyTool(args) {
    // Your implementation
    return { success: true, data: "..." };
}
```

### Custom Provider Integration

```python
# Add to LLMProviderManager
async def _openai_complete(self, model, messages, max_tokens, temperature):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=self.config.providers["openai"].api_key)
    
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    return {
        "content": response.choices[0].message.content,
        "usage": {
            "total_tokens": response.usage.total_tokens
        }
    }
```

---

## Next Steps

1. **Scale horizontally**: Add more orchestrator instances behind load balancer
2. **Add observability**: Integrate DataDog, New Relic, or Elastic APM
3. **Implement auth**: Add JWT authentication to API endpoints
4. **Add web UI**: Build React dashboard for monitoring
5. **Fine-tune routing**: Use real embeddings instead of keyword matching
6. **Add GraphRAG**: Implement Neo4j for complex knowledge graphs

---

## Support

- **Documentation**: Check the inline comments in `agent_platform.py`
- **Issues**: Test each component individually with the provided tests
- **Performance**: Monitor `/v1/stats` and `/health` endpoints
- **Costs**: Track with `/v1/stats` - aim for <$0.01 per request

---

## Success Checklist

- [ ] All 4 LLM providers initialized
- [ ] Redis cache working (hit rate >70%)
- [ ] MCP server responding
- [ ] GitHub integration active
- [ ] API server healthy
- [ ] Test requests completing <2s
- [ ] Cost tracking showing <$0.01/request
- [ ] Cache saving >80% of costs

**You're ready for production! 🚀**