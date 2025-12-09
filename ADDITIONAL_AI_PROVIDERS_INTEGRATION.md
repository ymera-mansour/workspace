# Additional AI Providers Integration Guide

**Complete integration for 4 additional AI providers - Cohere, Together AI, Anthropic Claude, and Replicate**

**Total: 11 new models added (4 FREE, 2 optional paid)**

**Cost: $0 base + optional paid features**

---

## Table of Contents

1. [Provider Overview](#provider-overview)
2. [Cohere Integration (FREE)](#cohere-integration-free)
3. [Together AI Integration (FREE)](#together-ai-integration-free)
4. [Anthropic Claude Integration (OPTIONAL)](#anthropic-claude-integration-optional)
5. [Replicate Integration (OPTIONAL)](#replicate-integration-optional)
6. [Complete System Summary](#complete-system-summary)
7. [Installation Guide](#installation-guide)
8. [Cost Analysis](#cost-analysis)

---

## Provider Overview

### Summary of New Providers

| Provider | Models | Free Tier | Primary Use Case | Priority |
|----------|--------|-----------|------------------|----------|
| **Cohere** | 3 | 100 calls/min | Embeddings, RAG, Classification | **CRITICAL** |
| **Together AI** | 6 | $25/month credits | Fast inference, Open-source models | **CRITICAL** |
| **Anthropic Claude** | 2 | Limited trial | Advanced reasoning, Safety | **OPTIONAL** |
| **Replicate** | Varies | $0.50/month | Specialized ML models | **OPTIONAL** |

### Integration with Existing System

**Current System**: 42 models across 5 providers (Gemini, Mistral, Groq, OpenRouter, HuggingFace)

**After Integration**: 53 models across 9 providers

**Cost Impact**:
- Base cost: $0 (Cohere + Together AI free tiers)
- Optional: Claude + Replicate (paid features)

---

## Cohere Integration (FREE)

### Overview

**Provider**: Cohere
**Website**: https://cohere.com/
**Free Tier**: 100 API calls/minute
**Sign Up**: Required (free account)

### Use Cases

1. **Document Embeddings**: Convert text to vector representations
2. **RAG (Retrieval Augmented Generation)**: Context-aware generation
3. **Text Classification**: Categorize text into predefined classes
4. **Semantic Search**: Find similar documents

### Models Available

#### 1. embed-english-v3.0 (Embeddings)

**Specifications**:
- **Dimension**: 1024
- **Context Length**: 512 tokens
- **Use Case**: Text embeddings for search and classification
- **Performance**: State-of-the-art on MTEB benchmark
- **Cost**: FREE (100 calls/min)

**Configuration**:
```python
{
    "model_id": "embed-english-v3.0",
    "endpoint": "https://api.cohere.ai/v1/embed",
    "capabilities": ["embeddings", "semantic_search"],
    "input_type": "search_document",  # or "search_query"
    "embedding_types": ["float"],
    "max_tokens": 512,
    "rate_limit": {
        "requests_per_minute": 100,
        "requests_per_month": 1000000  # 100 req/min * 60 min * 24 hr * 30 days
    }
}
```

#### 2. command-r (RAG)

**Specifications**:
- **Parameters**: 35B
- **Context Length**: 128K tokens
- **Use Case**: Retrieval-augmented generation
- **Strengths**: Long context, grounded responses
- **Cost**: FREE (100 calls/min)

**Configuration**:
```python
{
    "model_id": "command-r",
    "endpoint": "https://api.cohere.ai/v1/chat",
    "capabilities": ["rag", "long_context", "chat"],
    "max_tokens": 128000,
    "rate_limit": {
        "requests_per_minute": 100,
        "requests_per_month": 1000000
    }
}
```

#### 3. command-r+ (Complex Reasoning)

**Specifications**:
- **Parameters**: 104B
- **Context Length**: 128K tokens
- **Use Case**: Complex reasoning, advanced RAG
- **Strengths**: Superior reasoning, coding, math
- **Cost**: FREE (100 calls/min)

**Configuration**:
```python
{
    "model_id": "command-r-plus",
    "endpoint": "https://api.cohere.ai/v1/chat",
    "capabilities": ["rag", "reasoning", "coding", "math"],
    "max_tokens": 128000,
    "rate_limit": {
        "requests_per_minute": 100,
        "requests_per_month": 1000000
    }
}
```

### Complete Cohere Configuration

**cohere_advanced_config.py**:
```python
"""
Cohere AI Provider Configuration
Supports embeddings, RAG, and text classification
FREE tier: 100 calls/minute
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import cohere
import time
from collections import deque

@dataclass
class CohereModelConfig:
    """Configuration for a Cohere model"""
    model_id: str
    endpoint: str
    capabilities: List[str]
    max_tokens: int
    rate_limit_rpm: int
    rate_limit_rpm_total: int
    description: str
    embedding_dimension: Optional[int] = None

class CohereAPIKeyManager:
    """Manages multiple Cohere API keys with rotation"""
    
    def __init__(self):
        self.keys = self._load_keys()
        self.current_key_index = 0
        self.key_usage = {key: 0 for key in self.keys}
        self.key_errors = {key: 0 for key in self.keys}
        
    def _load_keys(self) -> List[str]:
        """Load Cohere API keys from environment"""
        keys = []
        i = 1
        while True:
            key = os.getenv(f'COHERE_API_KEY_{i}')
            if not key:
                if i == 1:
                    key = os.getenv('COHERE_API_KEY')
                    if key:
                        keys.append(key)
                break
            keys.append(key)
            i += 1
        return keys if keys else []
    
    def get_key(self, strategy='round_robin') -> str:
        """Get API key based on rotation strategy"""
        if not self.keys:
            raise ValueError("No Cohere API keys configured")
        
        if strategy == 'round_robin':
            key = self.keys[self.current_key_index]
            self.current_key_index = (self.current_key_index + 1) % len(self.keys)
        elif strategy == 'least_used':
            key = min(self.keys, key=lambda k: self.key_usage[k])
        else:
            key = self.keys[0]
        
        self.key_usage[key] += 1
        return key

class CohereRateLimiter:
    """Rate limiter for Cohere API calls"""
    
    def __init__(self, rpm_limit: int = 100):
        self.rpm_limit = rpm_limit
        self.request_times = deque()
        
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        now = time.time()
        
        # Remove requests older than 1 minute
        while self.request_times and now - self.request_times[0] > 60:
            self.request_times.popleft()
        
        # If at limit, wait
        if len(self.request_times) >= self.rpm_limit:
            sleep_time = 60 - (now - self.request_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
                self.wait_if_needed()  # Retry
        
        self.request_times.append(now)

class CohereProvider:
    """Complete Cohere AI provider implementation"""
    
    # Model configurations
    MODELS = {
        'embed-english-v3.0': CohereModelConfig(
            model_id='embed-english-v3.0',
            endpoint='https://api.cohere.ai/v1/embed',
            capabilities=['embeddings', 'semantic_search'],
            max_tokens=512,
            rate_limit_rpm=100,
            rate_limit_rpm_total=1000000,
            description='Text embeddings for search and classification',
            embedding_dimension=1024
        ),
        'command-r': CohereModelConfig(
            model_id='command-r',
            endpoint='https://api.cohere.ai/v1/chat',
            capabilities=['rag', 'long_context', 'chat'],
            max_tokens=128000,
            rate_limit_rpm=100,
            rate_limit_rpm_total=1000000,
            description='RAG and long-context generation'
        ),
        'command-r-plus': CohereModelConfig(
            model_id='command-r-plus',
            endpoint='https://api.cohere.ai/v1/chat',
            capabilities=['rag', 'reasoning', 'coding', 'math'],
            max_tokens=128000,
            rate_limit_rpm=100,
            rate_limit_rpm_total=1000000,
            description='Advanced reasoning and complex tasks'
        )
    }
    
    def __init__(self):
        self.key_manager = CohereAPIKeyManager()
        self.rate_limiter = CohereRateLimiter(rpm_limit=100)
        self.clients = {}
        
    def get_client(self) -> cohere.Client:
        """Get Cohere client with current API key"""
        api_key = self.key_manager.get_key()
        if api_key not in self.clients:
            self.clients[api_key] = cohere.Client(api_key)
        return self.clients[api_key]
    
    def generate_embeddings(self, texts: List[str], 
                          input_type: str = 'search_document') -> List[List[float]]:
        """Generate embeddings for texts"""
        self.rate_limiter.wait_if_needed()
        
        client = self.get_client()
        response = client.embed(
            texts=texts,
            model='embed-english-v3.0',
            input_type=input_type
        )
        
        return response.embeddings
    
    def chat(self, message: str, model: str = 'command-r',
             documents: Optional[List[Dict]] = None,
             **kwargs) -> str:
        """Chat with RAG support"""
        self.rate_limiter.wait_if_needed()
        
        client = self.get_client()
        response = client.chat(
            message=message,
            model=model,
            documents=documents,
            **kwargs
        )
        
        return response.text
    
    def classify(self, texts: List[str], examples: List[Dict]) -> List[Dict]:
        """Classify texts using examples"""
        self.rate_limiter.wait_if_needed()
        
        client = self.get_client()
        response = client.classify(
            inputs=texts,
            examples=examples
        )
        
        return [
            {
                'text': classification.input,
                'prediction': classification.prediction,
                'confidence': classification.confidence
            }
            for classification in response.classifications
        ]

# Example usage
if __name__ == '__main__':
    provider = CohereProvider()
    
    # Generate embeddings
    embeddings = provider.generate_embeddings(
        texts=['Hello world', 'Machine learning is great'],
        input_type='search_document'
    )
    print(f"Generated {len(embeddings)} embeddings")
    
    # Chat with RAG
    response = provider.chat(
        message="What is machine learning?",
        model='command-r',
        documents=[
            {'text': 'Machine learning is a subset of AI...'},
            {'text': 'Deep learning uses neural networks...'}
        ]
    )
    print(f"Response: {response}")
```

### Installation

```bash
# Install Cohere SDK
pip install cohere

# Set up API key
export COHERE_API_KEY="your_api_key_here"

# Multiple keys for rotation
export COHERE_API_KEY_1="key_1"
export COHERE_API_KEY_2="key_2"
```

---

## Together AI Integration (FREE)

### Overview

**Provider**: Together AI
**Website**: https://together.ai/
**Free Tier**: $25/month in credits
**Sign Up**: Required (free account)

### Use Cases

1. **Fast Inference**: Ultra-fast open-source model inference
2. **Alternative to Groq**: Similar speed with more model options
3. **Open-Source Models**: Llama, Mixtral, Qwen, and more

### Models Available

#### 1. meta-llama/Llama-3-70b-chat-hf

**Specifications**:
- **Parameters**: 70B
- **Context Length**: 8K tokens
- **Use Case**: General-purpose chat
- **Performance**: High quality responses
- **Cost**: $0.88 per 1M tokens (included in free $25 credits)

#### 2. mistralai/Mixtral-8x22B-Instruct-v0.1

**Specifications**:
- **Parameters**: 176B (MoE)
- **Context Length**: 64K tokens
- **Use Case**: Complex reasoning, long context
- **Performance**: Top-tier quality
- **Cost**: $1.20 per 1M tokens

#### 3. Qwen/Qwen2-72B-Instruct

**Specifications**:
- **Parameters**: 72B
- **Context Length**: 32K tokens
- **Use Case**: Reasoning, multilingual
- **Performance**: Excellent on coding and math
- **Cost**: $0.90 per 1M tokens

#### 4. codellama/CodeLlama-70b-Instruct-hf

**Specifications**:
- **Parameters**: 70B
- **Context Length**: 16K tokens
- **Use Case**: Code generation
- **Performance**: Specialized for programming
- **Cost**: $0.90 per 1M tokens

#### 5. deepseek-ai/deepseek-coder-33b-instruct

**Specifications**:
- **Parameters**: 33B
- **Context Length**: 16K tokens
- **Use Case**: Code generation
- **Performance**: Strong coding capabilities
- **Cost**: $0.80 per 1M tokens

#### 6. WizardLM/WizardCoder-Python-34B-V1.0

**Specifications**:
- **Parameters**: 34B
- **Context Length**: 16K tokens
- **Use Case**: Python coding
- **Performance**: Python specialist
- **Cost**: $0.80 per 1M tokens

### Complete Together AI Configuration

**together_ai_config.py**:
```python
"""
Together AI Provider Configuration
Fast inference for open-source models
FREE tier: $25/month in credits
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass
import together
import time

@dataclass
class TogetherModelConfig:
    model_id: str
    context_length: int
    cost_per_1m_tokens: float
    capabilities: List[str]
    description: str

class TogetherAIProvider:
    """Complete Together AI provider implementation"""
    
    MODELS = {
        'llama-3-70b': TogetherModelConfig(
            model_id='meta-llama/Llama-3-70b-chat-hf',
            context_length=8192,
            cost_per_1m_tokens=0.88,
            capabilities=['chat', 'general'],
            description='General-purpose chat model'
        ),
        'mixtral-8x22b': TogetherModelConfig(
            model_id='mistralai/Mixtral-8x22B-Instruct-v0.1',
            context_length=65536,
            capabilities=['reasoning', 'long_context'],
            cost_per_1m_tokens=1.20,
            description='Complex reasoning with long context'
        ),
        'qwen2-72b': TogetherModelConfig(
            model_id='Qwen/Qwen2-72B-Instruct',
            context_length=32768,
            cost_per_1m_tokens=0.90,
            capabilities=['reasoning', 'coding', 'multilingual'],
            description='Strong reasoning and coding'
        ),
        'codellama-70b': TogetherModelConfig(
            model_id='codellama/CodeLlama-70b-Instruct-hf',
            context_length=16384,
            cost_per_1m_tokens=0.90,
            capabilities=['coding'],
            description='Code generation specialist'
        ),
        'deepseek-coder-33b': TogetherModelConfig(
            model_id='deepseek-ai/deepseek-coder-33b-instruct',
            context_length=16384,
            cost_per_1m_tokens=0.80,
            capabilities=['coding'],
            description='Strong coding capabilities'
        ),
        'wizardcoder-python-34b': TogetherModelConfig(
            model_id='WizardLM/WizardCoder-Python-34B-V1.0',
            context_length=16384,
            cost_per_1m_tokens=0.80,
            capabilities=['coding', 'python'],
            description='Python coding specialist'
        )
    }
    
    def __init__(self):
        self.api_key = os.getenv('TOGETHER_API_KEY')
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY not found in environment")
        together.api_key = self.api_key
        
    def chat(self, prompt: str, model: str = 'llama-3-70b',
             max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate response from model"""
        model_config = self.MODELS.get(model)
        if not model_config:
            raise ValueError(f"Unknown model: {model}")
        
        response = together.Complete.create(
            model=model_config.model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response['output']['choices'][0]['text']
    
    def get_usage_cost(self, tokens_used: int, model: str) -> float:
        """Calculate cost for tokens used"""
        model_config = self.MODELS.get(model)
        if not model_config:
            return 0.0
        
        return (tokens_used / 1_000_000) * model_config.cost_per_1m_tokens
```

### Installation

```bash
# Install Together AI SDK
pip install together

# Set up API key
export TOGETHER_API_KEY="your_api_key_here"
```

---

## Anthropic Claude Integration (OPTIONAL)

### Overview

**Provider**: Anthropic
**Website**: https://console.anthropic.com/
**Free Tier**: Limited trial credits
**Sign Up**: Required
**Priority**: OPTIONAL (Gemini can substitute)

### Models Available

#### 1. claude-3-haiku (Fast, Cheap)

**Specifications**:
- **Context Length**: 200K tokens
- **Use Case**: Fast responses, high throughput
- **Performance**: Good quality at low cost
- **Cost**: $0.25 per 1M input tokens, $1.25 per 1M output tokens

#### 2. claude-3-sonnet (Balanced)

**Specifications**:
- **Context Length**: 200K tokens
- **Use Case**: Balanced performance
- **Performance**: High quality reasoning
- **Cost**: $3 per 1M input tokens, $15 per 1M output tokens

### Configuration

**claude_config.py**:
```python
"""
Anthropic Claude Provider Configuration
Advanced reasoning and safety
OPTIONAL - Paid service
"""

import os
from anthropic import Anthropic

class ClaudeProvider:
    """Anthropic Claude provider"""
    
    MODELS = {
        'haiku': {
            'model_id': 'claude-3-haiku-20240307',
            'context_length': 200000,
            'cost_input': 0.25,  # per 1M tokens
            'cost_output': 1.25
        },
        'sonnet': {
            'model_id': 'claude-3-sonnet-20240229',
            'context_length': 200000,
            'cost_input': 3.0,
            'cost_output': 15.0
        }
    }
    
    def __init__(self):
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")
        self.client = Anthropic(api_key=self.api_key)
    
    def chat(self, message: str, model: str = 'haiku',
             max_tokens: int = 1024) -> str:
        """Generate response"""
        model_config = self.MODELS[model]
        
        response = self.client.messages.create(
            model=model_config['model_id'],
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": message}]
        )
        
        return response.content[0].text
```

### Installation

```bash
# Install Anthropic SDK
pip install anthropic

# Set up API key (optional)
export ANTHROPIC_API_KEY="your_api_key_here"
```

---

## Replicate Integration (OPTIONAL)

### Overview

**Provider**: Replicate
**Website**: https://replicate.com/
**Free Tier**: $0.50/month in credits
**Sign Up**: Required
**Priority**: OPTIONAL (for specialized models)

### Use Cases

1. **Image Generation**: SDXL, Stable Diffusion
2. **Audio Processing**: Whisper, music generation
3. **Video Processing**: Video models
4. **Specialized ML Models**: Various community models

### Configuration

**replicate_config.py**:
```python
"""
Replicate Provider Configuration
Specialized ML models
OPTIONAL - Minimal free tier
"""

import os
import replicate

class ReplicateProvider:
    """Replicate provider for specialized models"""
    
    def __init__(self):
        self.api_key = os.getenv('REPLICATE_API_TOKEN')
        if self.api_key:
            os.environ['REPLICATE_API_TOKEN'] = self.api_key
    
    def run_model(self, model: str, input_data: dict) -> any:
        """Run a Replicate model"""
        output = replicate.run(model, input=input_data)
        return output
    
    # Example: Image generation
    def generate_image(self, prompt: str) -> str:
        """Generate image using SDXL"""
        output = replicate.run(
            "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            input={"prompt": prompt}
        )
        return output[0]
```

### Installation

```bash
# Install Replicate SDK
pip install replicate

# Set up API key (optional)
export REPLICATE_API_TOKEN="your_api_token_here"
```

---

## Complete System Summary

### Updated Tool Inventory

**Total: 115 Tools (53 AI models + 18 MCPs + 15 ML + 25 Infrastructure + 4 Providers)**

**AI Models by Provider**:
1. Gemini: 4 models
2. Mistral: 6 models
3. Groq: 6 models
4. OpenRouter: 14 models
5. HuggingFace: 12 models
6. **Cohere: 3 models** ✅ NEW
7. **Together AI: 6 models** ✅ NEW
8. **Anthropic: 2 models** (optional)
9. **Replicate: Variable** (optional)

**Total AI Models**: 53 (42 existing + 11 new)

### Use Case Mapping

**For Embeddings & Semantic Search**:
- Primary: Cohere embed-english-v3.0
- Alternative: HuggingFace sentence-transformers

**For RAG (Retrieval Augmented Generation)**:
- Primary: Cohere command-r
- Advanced: Cohere command-r+
- Alternative: Gemini 1.5 Pro

**For Fast Inference**:
- Primary: Groq (llama-3.1-8b-instant)
- Alternative: Together AI (various models)

**For Complex Reasoning**:
- Primary: Cohere command-r+
- Alternative: Claude sonnet (paid), Gemini 1.5 Pro

**For Code Generation**:
- Primary: HuggingFace Qwen2.5-Coder-32B
- Alternative: Together AI CodeLlama-70b
- Specialist: Together AI WizardCoder-Python-34B

**For Long Context**:
- Primary: Cohere command-r (128K)
- Alternative: Together AI Mixtral-8x22B (64K)

---

## Installation Guide

### Phase 1: Cohere Setup (5 minutes)

```bash
# Install SDK
pip install cohere

# Set up API key
export COHERE_API_KEY="your_cohere_api_key"

# Test installation
python -c "import cohere; print('Cohere installed successfully')"
```

### Phase 2: Together AI Setup (5 minutes)

```bash
# Install SDK
pip install together

# Set up API key
export TOGETHER_API_KEY="your_together_api_key"

# Test installation
python -c "import together; print('Together AI installed successfully')"
```

### Phase 3: Anthropic Claude Setup (5 minutes) - OPTIONAL

```bash
# Install SDK
pip install anthropic

# Set up API key (optional)
export ANTHROPIC_API_KEY="your_anthropic_api_key"

# Test installation
python -c "import anthropic; print('Anthropic installed successfully')"
```

### Phase 4: Replicate Setup (5 minutes) - OPTIONAL

```bash
# Install SDK
pip install replicate

# Set up API key (optional)
export REPLICATE_API_TOKEN="your_replicate_token"

# Test installation
python -c "import replicate; print('Replicate installed successfully')"
```

**Total Installation Time**: 20 minutes (10 minutes for required, 10 minutes for optional)

---

## Cost Analysis

### Required Providers (FREE)

**Cohere**:
- Free Tier: 100 API calls/minute
- Monthly Limit: ~4.3M calls/month (100/min * 60 * 24 * 30)
- Cost: **$0**

**Together AI**:
- Free Credits: $25/month
- Estimated Usage:
  - Llama-3-70b: ~28M tokens ($0.88/1M)
  - Mixtral-8x22B: ~20M tokens ($1.20/1M)
  - Code models: ~30M tokens ($0.80/1M)
- Cost: **$0** (within free credits)

**Total Required Cost**: **$0/month**

### Optional Providers (PAID)

**Anthropic Claude**:
- Haiku: $0.25-$1.25 per 1M tokens
- Sonnet: $3-$15 per 1M tokens
- Estimated Cost: $10-50/month (depends on usage)

**Replicate**:
- Free: $0.50/month
- Paid: Variable (depends on models used)
- Estimated Cost: $1-20/month

**Total Optional Cost**: $11-70/month (if used)

### Complete System Cost

**Base System**: **$0/month** ✅
- 42 existing AI models (Gemini, Mistral, Groq, OpenRouter, HuggingFace)
- 18 MCP tools
- 15 ML/Learning tools
- 25 Infrastructure tools
- 3 Cohere models (NEW)
- 6 Together AI models (NEW)

**With Optional Features**: $11-70/month
- All base system tools
- 2 Anthropic Claude models
- Replicate specialized models

---

## Integration Examples

### Example 1: RAG Pipeline with Cohere

```python
from cohere_advanced_config import CohereProvider

# Initialize provider
cohere = CohereProvider()

# Step 1: Generate embeddings for documents
documents = [
    "Machine learning is a subset of AI...",
    "Deep learning uses neural networks...",
    "Natural language processing..."
]

embeddings = cohere.generate_embeddings(
    texts=documents,
    input_type='search_document'
)

# Step 2: Generate embedding for query
query = "What is deep learning?"
query_embedding = cohere.generate_embeddings(
    texts=[query],
    input_type='search_query'
)[0]

# Step 3: Find most similar documents (simple cosine similarity)
import numpy as np
similarities = [
    np.dot(query_embedding, doc_emb) / 
    (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
    for doc_emb in embeddings
]
top_doc_idx = np.argmax(similarities)

# Step 4: Use RAG with relevant documents
response = cohere.chat(
    message=query,
    model='command-r',
    documents=[{'text': documents[top_doc_idx]}]
)

print(f"Response: {response}")
```

### Example 2: Fast Code Generation with Together AI

```python
from together_ai_config import TogetherAIProvider

# Initialize provider
together = TogetherAIProvider()

# Generate Python code
code = together.chat(
    prompt="Write a Python function to calculate fibonacci numbers",
    model='wizardcoder-python-34b',
    max_tokens=512
)

print(f"Generated code:\n{code}")
```

### Example 3: Multi-Provider Intelligent Routing

```python
def route_request(task_type: str, complexity: str) -> dict:
    """Intelligent routing across all providers"""
    
    routing_rules = {
        'embeddings': {
            'provider': 'cohere',
            'model': 'embed-english-v3.0'
        },
        'rag': {
            'simple': {'provider': 'cohere', 'model': 'command-r'},
            'complex': {'provider': 'cohere', 'model': 'command-r-plus'}
        },
        'code': {
            'python': {'provider': 'together_ai', 'model': 'wizardcoder-python-34b'},
            'general': {'provider': 'huggingface', 'model': 'qwen-coder-32b'}
        },
        'reasoning': {
            'simple': {'provider': 'gemini', 'model': 'gemini-2.0-flash-exp'},
            'complex': {'provider': 'cohere', 'model': 'command-r-plus'}
        },
        'fast': {
            'provider': 'groq',
            'model': 'llama-3.1-8b-instant'
        }
    }
    
    # Simple routing logic
    if task_type in routing_rules:
        rule = routing_rules[task_type]
        if isinstance(rule, dict) and complexity in rule:
            return rule[complexity]
        return rule
    
    # Default to fast model
    return routing_rules['fast']

# Example usage
task = route_request('code', 'python')
print(f"Route to: {task['provider']} - {task['model']}")
```

---

## Best Practices

### Cohere Best Practices

1. **Embeddings**:
   - Use `search_document` for documents to be searched
   - Use `search_query` for search queries
   - Batch embeddings for efficiency (up to 96 texts per call)

2. **RAG**:
   - Use command-r for most RAG tasks
   - Use command-r+ for complex reasoning
   - Provide relevant documents for better grounding

3. **Rate Limiting**:
   - Respect 100 calls/minute limit
   - Implement exponential backoff
   - Use multiple API keys for higher throughput

### Together AI Best Practices

1. **Model Selection**:
   - Use Llama-3-70b for general chat
   - Use Mixtral-8x22B for long context
   - Use specialized code models for programming

2. **Cost Optimization**:
   - Monitor token usage carefully
   - Use cheaper models when possible
   - Implement caching for repeated queries

3. **Performance**:
   - Together AI is fast (similar to Groq)
   - Use for high-throughput scenarios
   - Combine with caching for best results

---

## Monitoring & Metrics

### Key Metrics to Track

**Cohere**:
- API calls per minute
- Embedding generation time
- RAG response quality
- Token usage

**Together AI**:
- Token usage by model
- Cost per request
- Response time
- Quality metrics

**System-Wide**:
- Provider distribution (which provider used most)
- Cost tracking across all providers
- Performance comparison
- Cache hit rates

### Monitoring Implementation

```python
import time
from collections import defaultdict

class MultiProviderMonitor:
    """Monitor usage across all AI providers"""
    
    def __init__(self):
        self.metrics = defaultdict(lambda: {
            'calls': 0,
            'tokens': 0,
            'cost': 0.0,
            'response_times': []
        })
    
    def log_request(self, provider: str, model: str, 
                   tokens: int, response_time: float, cost: float):
        """Log a request"""
        key = f"{provider}:{model}"
        self.metrics[key]['calls'] += 1
        self.metrics[key]['tokens'] += tokens
        self.metrics[key]['cost'] += cost
        self.metrics[key]['response_times'].append(response_time)
    
    def get_summary(self) -> dict:
        """Get usage summary"""
        import statistics
        
        summary = {}
        for key, data in self.metrics.items():
            avg_response_time = statistics.mean(data['response_times']) if data['response_times'] else 0
            summary[key] = {
                'calls': data['calls'],
                'tokens': data['tokens'],
                'total_cost': data['cost'],
                'avg_response_time': avg_response_time
            }
        
        return summary

# Global monitor instance
monitor = MultiProviderMonitor()
```

---

## Troubleshooting

### Common Issues

**Cohere**:
- **Rate limit exceeded**: Implement rate limiting, use multiple keys
- **Invalid input**: Check text length (max 512 tokens for embeddings)
- **API key invalid**: Verify key at https://dashboard.cohere.com/

**Together AI**:
- **Out of credits**: Monitor usage, upgrade if needed
- **Model not available**: Check model ID, some models may be deprecated
- **Timeout**: Increase timeout, use smaller models

**Anthropic Claude**:
- **Cost concerns**: Use Haiku for cheaper option, implement budgets
- **API errors**: Check quota limits, verify API key

**Replicate**:
- **Model loading slow**: First run loads model (cold start)
- **Out of credits**: Monitor usage, add payment method

---

## Migration Guide

### From Existing System

If you're currently using only the base 42 models, here's how to integrate the new providers:

**Step 1**: Install required packages
```bash
pip install cohere together
```

**Step 2**: Add API keys to environment
```bash
export COHERE_API_KEY="your_key"
export TOGETHER_API_KEY="your_key"
```

**Step 3**: Update routing logic to include new providers

**Step 4**: Test with small workload

**Step 5**: Monitor costs and performance

**Step 6**: Gradually increase usage

---

## Complete System Summary

### Final Tool Count

**Total Tools**: 115
- **AI Models**: 53 (42 + 11 new)
  - Cohere: 3 (embeddings, RAG)
  - Together AI: 6 (fast inference, code)
  - Anthropic: 2 (optional, reasoning)
  - Replicate: Variable (optional, specialized)
- **MCP Tools**: 18
- **ML/Learning Tools**: 15
- **Infrastructure Tools**: 25
- **Providers**: 9 (5 original + 4 new)

### Cost Breakdown

**FREE Tier** ($0/month):
- 42 original AI models
- 18 MCP tools
- 15 ML/Learning tools
- 25 Infrastructure tools
- 3 Cohere models
- 6 Together AI models

**Optional Paid** ($11-70/month):
- 2 Anthropic Claude models
- Replicate specialized models

**Base System Cost**: **$0/month** ✅

---

## Quick Reference

### API Key Environment Variables

```bash
# Required (FREE)
export COHERE_API_KEY="your_cohere_key"
export TOGETHER_API_KEY="your_together_key"

# Optional (PAID)
export ANTHROPIC_API_KEY="your_anthropic_key"
export REPLICATE_API_TOKEN="your_replicate_token"

# Existing (FREE)
export GEMINI_API_KEY="your_gemini_key"
export MISTRAL_API_KEY="your_mistral_key"
export GROQ_API_KEY="your_groq_key"
export OPENROUTER_API_KEY="your_openrouter_key"
export HUGGINGFACE_API_KEY="your_huggingface_key"
```

### Model Selection Cheat Sheet

| Task | Best Model | Provider | Cost |
|------|-----------|----------|------|
| Embeddings | embed-english-v3.0 | Cohere | FREE |
| RAG | command-r | Cohere | FREE |
| Complex Reasoning | command-r+ | Cohere | FREE |
| Fast Chat | llama-3.1-8b-instant | Groq | FREE |
| Code (Python) | wizardcoder-python-34b | Together AI | FREE |
| Code (General) | qwen-coder-32b | HuggingFace | FREE |
| Long Context | mixtral-8x22b | Together AI | FREE |
| Safety Critical | claude-3-sonnet | Anthropic | PAID |

---

## Conclusion

This integration adds **11 new AI models** across **4 providers**, bringing the total to **53 AI models** across **9 providers**.

**Key Benefits**:
- **Embeddings & RAG**: Cohere provides best-in-class embeddings and RAG
- **Fast Inference**: Together AI offers fast open-source model inference
- **Cost**: Maintains $0 base cost with optional paid features
- **Flexibility**: More model options for different tasks

**Next Steps**:
1. Sign up for free accounts (Cohere, Together AI)
2. Install required packages
3. Configure API keys
4. Test integration with small workload
5. Update routing logic
6. Monitor performance and costs

**Total System**: 115 tools, 53 AI models, 9 providers, $0 base cost! ✅
