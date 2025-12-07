# 🔧 Complete Integration Guide for Existing Platform
## Step-by-Step Instructions for Qoder Agent

This guide provides detailed instructions for integrating the multi-agent system into your existing platform and expanding it with new models, MCP servers, tools, and AI/ML capabilities.

---

## 📋 Table of Contents

1. [Integration into Existing Platform](#integration-into-existing-platform)
2. [Adding New LLM Models](#adding-new-llm-models)
3. [Adding New MCP Servers](#adding-new-mcp-servers)
4. [Adding Custom Tools](#adding-custom-tools)
5. [AI/ML Integration](#aiml-integration)
6. [Testing & Validation](#testing--validation)

---

## 1. Integration into Existing Platform

### Step 1.1: Analyze Your Existing Architecture

**Action Items:**
```bash
# Document your current platform structure
mkdir docs/existing_platform
touch docs/existing_platform/architecture.md

# Answer these questions:
# - What framework? (Django, FastAPI, Express, etc.)
# - Database? (PostgreSQL, MongoDB, etc.)
# - Authentication? (JWT, OAuth, etc.)
# - Current API structure?
# - Existing microservices?
```

### Step 1.2: Choose Integration Strategy

**Option A: Microservice Integration (Recommended)**

```yaml
# Your existing platform continues unchanged
# Add agent platform as a separate service

existing-platform/
├── api-gateway/          # Your existing API
├── user-service/         # Your existing service
├── auth-service/         # Your existing service
└── agent-platform/       # NEW: Add this
    ├── core/
    ├── mcp-server/
    └── docker-compose.yml
```

**Integration Steps:**

```python
# 1. In your existing API gateway, add a proxy endpoint
# Example with FastAPI:

from fastapi import FastAPI
import httpx

app = FastAPI()

AGENT_PLATFORM_URL = "http://agent-platform:8000"

@app.post("/api/v1/ai/completions")
async def ai_completion(request: dict, user_id: str = Depends(get_current_user)):
    """Proxy to agent platform with authentication"""
    
    # Add your authentication/authorization logic
    if not await check_user_permissions(user_id, "ai_access"):
        raise HTTPException(403, "AI access denied")
    
    # Forward to agent platform
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{AGENT_PLATFORM_URL}/v1/completions",
            json={
                **request,
                "user_id": user_id  # Pass your user ID
            },
            timeout=60.0
        )
        return response.json()
```

**Option B: Library Integration**

```python
# Install as a Python package in your existing codebase

# 1. Add to your requirements.txt
echo "agent-platform @ git+https://github.com/yourorg/agent-platform.git" >> requirements.txt

# 2. Use in your existing code
from agent_platform import ProductionOrchestrator

class YourExistingService:
    def __init__(self):
        self.ai_orchestrator = ProductionOrchestrator()
        
    async def process_with_ai(self, user_request):
        result = await self.ai_orchestrator.process_request(
            user_id=self.current_user_id,
            prompt=user_request
        )
        return result
```

### Step 1.3: Database Integration

**Connect to Your Existing Database:**

```python
# In core/agent_platform.py, add database connector

import asyncpg  # or your DB library

class DatabaseIntegration:
    """Connect agent platform to existing database"""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.pool = None
        
    async def initialize(self):
        """Create database connection pool"""
        self.pool = await asyncpg.create_pool(self.db_url)
        
    async def store_agent_result(self, user_id: str, result: dict):
        """Store agent results in your existing database"""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO agent_interactions 
                (user_id, prompt, response, agent_name, cost, created_at)
                VALUES ($1, $2, $3, $4, $5, NOW())
            ''', user_id, result['prompt'], result['response'], 
                result['agent'], result['metadata']['cost'])
    
    async def get_user_history(self, user_id: str) -> list:
        """Get user's agent interaction history"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT * FROM agent_interactions
                WHERE user_id = $1
                ORDER BY created_at DESC
                LIMIT 10
            ''', user_id)
            return [dict(row) for row in rows]

# Add to ProductionOrchestrator
class ProductionOrchestrator:
    def __init__(self):
        # ... existing code ...
        self.db = DatabaseIntegration(os.getenv("DATABASE_URL"))
        
    async def initialize(self):
        # ... existing code ...
        await self.db.initialize()
        
    async def process_request(self, user_id: str, prompt: str):
        result = await super().process_request(user_id, prompt)
        
        # Store in your database
        await self.db.store_agent_result(user_id, {
            'prompt': prompt,
            **result
        })
        
        return result
```

### Step 1.4: Authentication Integration

**Connect to Your Auth System:**

```python
# In api_server.py, integrate with your auth

from fastapi import Depends, HTTPException, Header
import jwt

async def verify_token(authorization: str = Header(...)) -> str:
    """Verify JWT token from your existing auth system"""
    try:
        token = authorization.replace("Bearer ", "")
        
        # Decode using your secret key
        payload = jwt.decode(
            token, 
            os.getenv("JWT_SECRET"), 
            algorithms=["HS256"]
        )
        
        return payload["user_id"]
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token")

@app.post("/v1/completions")
async def create_completion(
    request: CompletionRequest,
    user_id: str = Depends(verify_token)
):
    """Protected endpoint with your authentication"""
    
    # Check user permissions
    user = await get_user_from_db(user_id)
    if not user.has_permission("ai_access"):
        raise HTTPException(403, "No AI access")
    
    # Check user's quota
    usage = await get_user_ai_usage(user_id)
    if usage.daily_requests >= user.daily_limit:
        raise HTTPException(429, "Daily limit exceeded")
    
    # Process request
    result = await orchestrator.process_request(
        user_id=user_id,
        prompt=request.prompt
    )
    
    # Update usage
    await increment_user_usage(user_id)
    
    return result
```

### Step 1.5: Update Docker Compose for Existing Platform

```yaml
# docker-compose.yml - Add to your existing compose file

version: '3.8'

services:
  # Your existing services
  your-api:
    build: ./your-api
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - AGENT_PLATFORM_URL=http://agent-platform:8000
    depends_on:
      - agent-platform
  
  your-database:
    image: postgres:15
    # ... your config ...
  
  # NEW: Agent platform services
  agent-platform:
    build: ./agent-platform
    environment:
      - MISTRAL_API_KEY=${MISTRAL_API_KEY}
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - GROQ_API_KEY=${GROQ_API_KEY}
      - HF_API_KEY=${HF_API_KEY}
      - DATABASE_URL=${DATABASE_URL}  # Share your database
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
      - mcp-server
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
  
  mcp-server:
    build: ./agent-platform/mcp-server
    environment:
      - GITHUB_TOKEN=${GITHUB_TOKEN}

networks:
  default:
    name: your-existing-network
```

---

## 2. Adding New LLM Models

### Step 2.1: Add OpenAI GPT-4

```python
# In core/agent_platform.py - ConfigManager

def _load_providers(self) -> Dict[str, ProviderConfig]:
    return {
        # ... existing providers ...
        
        "openai": ProviderConfig(
            name="openai",
            api_key=os.getenv("OPENAI_API_KEY", ""),
            models=[
                "gpt-4-turbo-preview",
                "gpt-4",
                "gpt-3.5-turbo"
            ],
            cost_per_1k_tokens=0.01,  # GPT-4 turbo
            max_tokens=128000,
            rate_limit_rpm=500
        )
    }
```

```python
# In LLMProviderManager class, add OpenAI implementation

async def initialize(self):
    # ... existing code ...
    
    # OpenAI
    if cfg["openai"].api_key:
        from openai import AsyncOpenAI
        self.clients["openai"] = AsyncOpenAI(api_key=cfg["openai"].api_key)
        print("✓ OpenAI initialized")

async def _openai_complete(self, model, messages, max_tokens, temperature, tools):
    """OpenAI implementation"""
    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    
    response = await self.clients["openai"].chat.completions.create(**kwargs)
    
    return {
        "content": response.choices[0].message.content,
        "tool_calls": response.choices[0].message.tool_calls or [],
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
    }

async def complete(self, provider: str, model: str, messages: list, **kwargs):
    # ... existing code ...
    
    if provider == "openai":
        result = await self._openai_complete(model, messages, max_tokens, temperature, tools)
    # ... rest of code ...
```

### Step 2.2: Add Anthropic Claude

```python
# Add to ConfigManager
"anthropic": ProviderConfig(
    name="anthropic",
    api_key=os.getenv("ANTHROPIC_API_KEY", ""),
    models=[
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "claude-haiku-4-20250514"
    ],
    cost_per_1k_tokens=0.003,
    max_tokens=200000,
    rate_limit_rpm=50
)

# Initialize client
if cfg["anthropic"].api_key:
    from anthropic import AsyncAnthropic
    self.clients["anthropic"] = AsyncAnthropic(api_key=cfg["anthropic"].api_key)
    print("✓ Anthropic initialized")

# Implementation
async def _anthropic_complete(self, model, messages, max_tokens, temperature, tools):
    """Anthropic Claude implementation"""
    # Separate system message
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_messages = [m for m in messages if m["role"] != "system"]
    
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": user_messages,
        "temperature": temperature
    }
    
    if system:
        kwargs["system"] = system
    
    if tools:
        kwargs["tools"] = tools
    
    response = await self.clients["anthropic"].messages.create(**kwargs)
    
    return {
        "content": response.content[0].text if response.content else "",
        "tool_calls": [c for c in response.content if c.type == "tool_use"],
        "usage": {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens
        }
    }
```

### Step 2.3: Add Local Models (Ollama)

```python
# Add to ConfigManager
"ollama": ProviderConfig(
    name="ollama",
    api_key="",  # No API key needed
    models=[
        "llama3.1:70b",
        "codellama:34b",
        "mistral:7b"
    ],
    cost_per_1k_tokens=0.0,  # FREE (local)
    max_tokens=8192,
    rate_limit_rpm=1000  # No real limit
)

# Initialize
if True:  # Always available if Ollama running
    import aiohttp
    self.clients["ollama"] = aiohttp.ClientSession()
    print("✓ Ollama initialized (localhost:11434)")

# Implementation
async def _ollama_complete(self, model, messages, max_tokens, temperature, tools):
    """Ollama local models implementation"""
    
    # Ollama uses simple format
    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    
    async with self.clients["ollama"].post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
    ) as response:
        data = await response.json()
        
        return {
            "content": data["response"],
            "tool_calls": [],
            "usage": {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
            }
        }
```

### Step 2.4: Generic Provider Template

```python
# Template for adding any new provider

class GenericProviderAdapter:
    """Template for adding new LLM providers"""
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        
    async def initialize(self):
        """Setup client/session"""
        self.session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
    
    async def complete(self, model: str, messages: list, **kwargs) -> dict:
        """Implement completion endpoint"""
        
        # Convert messages to provider format
        formatted_messages = self._format_messages(messages)
        
        # Call provider API
        async with self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": formatted_messages,
                "max_tokens": kwargs.get("max_tokens", 4096),
                "temperature": kwargs.get("temperature", 0.7)
            }
        ) as response:
            data = await response.json()
            
            # Convert to standard format
            return {
                "content": data["choices"][0]["message"]["content"],
                "tool_calls": [],
                "usage": data.get("usage", {})
            }
    
    def _format_messages(self, messages: list) -> list:
        """Convert standard format to provider format"""
        return messages  # Implement conversion if needed
```

---

## 3. Adding New MCP Servers

### Step 3.1: Add Database MCP Server

```javascript
// mcp-server/database_tools.js

async function handleDatabaseQuery(args) {
    const { Pool } = require('pg');
    
    const pool = new Pool({
        connectionString: process.env.DATABASE_URL
    });
    
    try {
        // Validate query (prevent SQL injection)
        if (!isValidQuery(args.query)) {
            throw new Error('Invalid SQL query');
        }
        
        const result = await pool.query(args.query, args.params || []);
        
        return {
            rows: result.rows,
            rowCount: result.rowCount,
            fields: result.fields.map(f => f.name)
        };
    } finally {
        await pool.end();
    }
}

function isValidQuery(query) {
    // Only allow SELECT queries
    const normalized = query.trim().toUpperCase();
    return normalized.startsWith('SELECT') && 
           !normalized.includes('DROP') &&
           !normalized.includes('DELETE') &&
           !normalized.includes('UPDATE') &&
           !normalized.includes('INSERT');
}

// Add to server.js
case 'database_query':
    result = await handleDatabaseQuery(params.arguments);
    break;
```

```python
# Register in agent_platform.py

self.mcp.register_tool(MCPToolDefinition(
    name="database_query",
    description="Query the database (read-only SELECT queries)",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "SQL SELECT query"
            },
            "params": {
                "type": "array",
                "description": "Query parameters"
            }
        },
        "required": ["query"]
    },
    server_url="http://mcp-server:3000/mcp"
))
```

### Step 3.2: Add Slack MCP Server

```javascript
// mcp-server/slack_tools.js

const { WebClient } = require('@slack/web-api');

const slack = new WebClient(process.env.SLACK_BOT_TOKEN);

async function handleSlackSendMessage(args) {
    const result = await slack.chat.postMessage({
        channel: args.channel,
        text: args.message,
        blocks: args.blocks  // Optional rich formatting
    });
    
    return {
        ok: result.ok,
        channel: result.channel,
        ts: result.ts,
        message_url: `https://slack.com/archives/${result.channel}/p${result.ts.replace('.', '')}`
    };
}

async function handleSlackGetMessages(args) {
    const result = await slack.conversations.history({
        channel: args.channel,
        limit: args.limit || 10
    });
    
    return {
        messages: result.messages.map(m => ({
            user: m.user,
            text: m.text,
            timestamp: m.ts
        }))
    };
}

// Add to server.js
case 'slack_send_message':
    result = await handleSlackSendMessage(params.arguments);
    break;

case 'slack_get_messages':
    result = await handleSlackGetMessages(params.arguments);
    break;
```

### Step 3.3: Add Email MCP Server

```javascript
// mcp-server/email_tools.js

const nodemailer = require('nodemailer');

const transporter = nodemailer.createTransporter({
    host: process.env.SMTP_HOST,
    port: process.env.SMTP_PORT,
    secure: true,
    auth: {
        user: process.env.SMTP_USER,
        pass: process.env.SMTP_PASS
    }
});

async function handleSendEmail(args) {
    const info = await transporter.sendMail({
        from: args.from || process.env.DEFAULT_FROM_EMAIL,
        to: args.to,
        subject: args.subject,
        text: args.text,
        html: args.html
    });
    
    return {
        messageId: info.messageId,
        accepted: info.accepted,
        rejected: info.rejected
    };
}

// Add to server.js
case 'send_email':
    result = await handleSendEmail(params.arguments);
    break;
```

### Step 3.4: MCP Server Template

```javascript
// Template for creating new MCP tools

async function handleYourCustomTool(args) {
    try {
        // 1. Validate inputs
        if (!args.required_param) {
            throw new Error('Missing required parameter');
        }
        
        // 2. Initialize client/connection
        const client = new YourAPIClient(process.env.YOUR_API_KEY);
        
        // 3. Execute operation
        const result = await client.yourMethod(args);
        
        // 4. Format response
        return {
            success: true,
            data: result,
            metadata: {
                timestamp: new Date().toISOString()
            }
        };
        
    } catch (error) {
        // 5. Handle errors gracefully
        return {
            success: false,
            error: error.message
        };
    }
}

// Add to server.js MCP endpoint
case 'your_custom_tool':
    result = await handleYourCustomTool(params.arguments);
    break;
```

---

## 4. Adding Custom Tools

### Step 4.1: Python Tool Integration (Without MCP)

```python
# core/custom_tools.py

class CustomToolRegistry:
    """Direct Python tool integration"""
    
    def __init__(self):
        self.tools = {}
        
    def register(self, name: str, func: Callable):
        """Register a Python function as a tool"""
        self.tools[name] = func
        
    async def execute(self, name: str, **kwargs) -> Any:
        """Execute tool"""
        if name not in self.tools:
            raise ValueError(f"Tool {name} not found")
        
        tool = self.tools[name]
        
        # Execute async or sync
        if asyncio.iscoroutinefunction(tool):
            return await tool(**kwargs)
        else:
            return tool(**kwargs)

# Example tools
async def scrape_website(url: str) -> dict:
    """Scrape website content"""
    import aiohttp
    from bs4 import BeautifulSoup
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')
            
            return {
                "title": soup.title.string if soup.title else "",
                "text": soup.get_text()[:1000],
                "links": [a['href'] for a in soup.find_all('a', href=True)][:10]
            }

def calculate_sentiment(text: str) -> dict:
    """Analyze text sentiment"""
    from textblob import TextBlob
    
    blob = TextBlob(text)
    
    return {
        "polarity": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity,
        "sentiment": "positive" if blob.sentiment.polarity > 0 else "negative"
    }

# Register tools
tools = CustomToolRegistry()
tools.register("scrape_website", scrape_website)
tools.register("calculate_sentiment", calculate_sentiment)

# Use in orchestrator
class ProductionOrchestrator:
    def __init__(self):
        # ... existing code ...
        self.custom_tools = CustomToolRegistry()
        self._register_custom_tools()
    
    def _register_custom_tools(self):
        from custom_tools import scrape_website, calculate_sentiment
        self.custom_tools.register("scrape_website", scrape_website)
        self.custom_tools.register("calculate_sentiment", calculate_sentiment)
```

### Step 4.2: Image Processing Tools

```python
# core/image_tools.py

from PIL import Image
import io
import base64

async def analyze_image(image_data: bytes) -> dict:
    """Analyze image properties"""
    img = Image.open(io.BytesIO(image_data))
    
    return {
        "width": img.width,
        "height": img.height,
        "format": img.format,
        "mode": img.mode,
        "size_bytes": len(image_data)
    }

async def resize_image(image_data: bytes, width: int, height: int) -> str:
    """Resize image and return base64"""
    img = Image.open(io.BytesIO(image_data))
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    
    return base64.b64encode(buffer.getvalue()).decode()

async def extract_text_from_image(image_data: bytes) -> str:
    """OCR - extract text from image"""
    import pytesseract
    
    img = Image.open(io.BytesIO(image_data))
    text = pytesseract.image_to_string(img)
    
    return text

# Register
tools.register("analyze_image", analyze_image)
tools.register("resize_image", resize_image)
tools.register("extract_text_from_image", extract_text_from_image)
```

### Step 4.3: API Integration Tools

```python
# core/api_tools.py

async def call_external_api(url: str, method: str = "GET", **kwargs) -> dict:
    """Generic API caller"""
    import aiohttp
    
    async with aiohttp.ClientSession() as session:
        async with session.request(method, url, **kwargs) as response:
            return {
                "status": response.status,
                "data": await response.json() if response.content_type == 'application/json' else await response.text(),
                "headers": dict(response.headers)
            }

async def stripe_create_payment(amount: int, currency: str = "usd") -> dict:
    """Create Stripe payment intent"""
    import stripe
    stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
    
    intent = stripe.PaymentIntent.create(
        amount=amount,
        currency=currency
    )
    
    return {
        "id": intent.id,
        "client_secret": intent.client_secret,
        "amount": intent.amount
    }

async def twilio_send_sms(to: str, message: str) -> dict:
    """Send SMS via Twilio"""
    from twilio.rest import Client
    
    client = Client(
        os.getenv("TWILIO_ACCOUNT_SID"),
        os.getenv("TWILIO_AUTH_TOKEN")
    )
    
    message = client.messages.create(
        to=to,
        from_=os.getenv("TWILIO_PHONE_NUMBER"),
        body=message
    )
    
    return {
        "sid": message.sid,
        "status": message.status
    }
```

---

## 5. AI/ML Integration

### Step 5.1: Embeddings for Semantic Search

```python
# core/ml_integration.py

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class EmbeddingsEngine:
    """Generate and compare text embeddings"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings"""
        return self.model.encode(texts)
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity"""
        emb1 = self.model.encode([text1])
        emb2 = self.model.encode([text2])
        
        return np.dot(emb1[0], emb2[0]) / (
            np.linalg.norm(emb1[0]) * np.linalg.norm(emb2[0])
        )
    
    def find_similar(self, query: str, documents: List[str], top_k: int = 5) -> List[tuple]:
        """Find most similar documents"""
        query_emb = self.model.encode([query])
        doc_embs = self.model.encode(documents)
        
        similarities = np.dot(doc_embs, query_emb.T).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [(documents[i], similarities[i]) for i in top_indices]

# Integrate into orchestrator
class ProductionOrchestrator:
    def __init__(self):
        # ... existing code ...
        self.embeddings = EmbeddingsEngine()
    
    async def semantic_search(self, query: str, limit: int = 10):
        """Search past responses semantically"""
        
        # Get all cached responses
        all_responses = await self.cache.get_all_keys()
        
        # Find similar
        similar = self.embeddings.find_similar(query, all_responses, limit)
        
        return similar
```

### Step 5.2: Fine-Tuning Integration

```python
# core/finetuning.py

class FineTuningManager:
    """Manage model fine-tuning"""
    
    def __init__(self, provider: str = "openai"):
        self.provider = provider
        
    async def create_training_dataset(self, 
                                     interactions: List[dict]) -> str:
        """Convert interactions to training format"""
        import json
        
        dataset = []
        
        for interaction in interactions:
            dataset.append({
                "messages": [
                    {"role": "system", "content": interaction["system"]},
                    {"role": "user", "content": interaction["prompt"]},
                    {"role": "assistant", "content": interaction["response"]}
                ]
            })
        
        # Save as JSONL
        filename = f"training_data_{int(time.time())}.jsonl"
        
        with open(filename, 'w') as f:
            for item in dataset:
                f.write(json.dumps(item) + '\n')
        
        return filename
    
    async def start_finetuning(self, 
                              training_file: str,
                              model: str = "gpt-3.5-turbo") -> str:
        """Start fine-tuning job"""
        
        if self.provider == "openai":
            from openai import AsyncOpenAI
            client = AsyncOpenAI()
            
            # Upload file
            with open(training_file, 'rb') as f:
                file_obj = await client.files.create(
                    file=f,
                    purpose='fine-tune'
                )
            
            # Create fine-tune job
            job = await client.fine_tuning.jobs.create(
                training_file=file_obj.id,
                model=model
            )
            
            return job.id
    
    async def get_finetuning_status(self, job_id: str) -> dict:
        """Check fine-tuning status"""
        