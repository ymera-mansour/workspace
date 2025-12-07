# Complete AI Providers Configuration Guide

## 🎯 Overview

This guide covers configuration, setup, and best practices for all supported AI providers. All providers listed have free tiers or are completely free.

## Table of Contents

1. [Mistral AI](#mistral-ai)
2. [Google Gemini](#google-gemini)
3. [Groq](#groq)
4. [OpenRouter](#openrouter)
5. [HuggingFace](#huggingface)
6. [Anthropic Claude](#anthropic-claude)
7. [OpenAI](#openai)
8. [Cohere](#cohere)
9. [AI21 Labs](#ai21-labs)
10. [Model Comparison](#model-comparison)

---

## Mistral AI

### 🌟 Overview
- **Free Tier**: Yes (Limited credits)
- **Best For**: Fast inference, efficient models
- **Context Length**: Up to 32K tokens
- **Pricing**: Starting from $0.0002/1K tokens

### 📝 Setup Steps

1. **Get API Key**
   ```
   Visit: https://console.mistral.ai/
   - Sign up for free account
   - Navigate to API Keys section
   - Create new API key
   - Copy the key (shown only once)
   ```

2. **Configure in .env**
   ```bash
   MISTRAL_API_KEY=your_mistral_api_key_here
   ```

3. **Test Connection**
   ```python
   from mistralai.async_client import MistralAsyncClient
   
   client = MistralAsyncClient(api_key="your_key")
   response = await client.chat(
       model="mistral-small-latest",
       messages=[{"role": "user", "content": "Hello!"}]
   )
   print(response.choices[0].message.content)
   ```

### 🎯 Available Models

| Model | Speed | Quality | Cost/1M tokens | Best For |
|-------|-------|---------|----------------|----------|
| `mistral-small-latest` | ⚡⚡⚡ | ⭐⭐⭐ | $0.20 | Simple tasks, fast responses |
| `mistral-medium-latest` | ⚡⚡ | ⭐⭐⭐⭐ | $0.70 | Balanced tasks |
| `mistral-large-latest` | ⚡ | ⭐⭐⭐⭐⭐ | $2.00 | Complex reasoning |
| `codestral-latest` | ⚡⚡ | ⭐⭐⭐⭐⭐ | $0.30 | Code generation |

### ⚙️ Configuration Options

```python
config = {
    "model": "mistral-small-latest",
    "temperature": 0.7,  # 0.0 = deterministic, 1.0 = creative
    "max_tokens": 1024,
    "top_p": 0.9,
    "safe_mode": True,  # Content filtering
    "random_seed": 42  # For reproducibility
}
```

### 🎓 Best Practices

1. **Use mistral-small for**: Summarization, simple Q&A, data extraction
2. **Use mistral-medium for**: Analysis, complex Q&A, reasoning
3. **Use mistral-large for**: Complex reasoning, creative writing, coding
4. **Use codestral for**: Code generation, refactoring, code review

### 📊 Rate Limits
- Free Tier: 60 requests/minute
- Paid Tier: Unlimited (soft limits apply)

---

## Google Gemini

### 🌟 Overview
- **Free Tier**: Yes (Generous limits)
- **Best For**: Long context, multimodal tasks
- **Context Length**: Up to 1M tokens (Flash), 2M tokens (Pro)
- **Pricing**: Flash starting from $0.00015/1K tokens

### 📝 Setup Steps

1. **Get API Key**
   ```
   Visit: https://ai.google.dev/
   - Click "Get API Key"
   - Sign in with Google account
   - Create API key in Google AI Studio
   - Copy the key
   ```

2. **Configure in .env**
   ```bash
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

3. **Test Connection**
   ```python
   import google.generativeai as genai
   
   genai.configure(api_key="your_key")
   model = genai.GenerativeModel('gemini-1.5-flash')
   response = model.generate_content("Hello!")
   print(response.text)
   ```

### 🎯 Available Models

| Model | Speed | Quality | Context | Cost/1M tokens | Best For |
|-------|-------|---------|---------|----------------|----------|
| `gemini-1.5-flash` | ⚡⚡⚡ | ⭐⭐⭐⭐ | 1M | $0.15 | Fast tasks, long docs |
| `gemini-1.5-pro` | ⚡⚡ | ⭐⭐⭐⭐⭐ | 2M | $0.70 | Complex reasoning |
| `gemini-2.0-flash-exp` | ⚡⚡⚡ | ⭐⭐⭐⭐ | 1M | Free (experimental) | Testing, prototyping |

### ⚙️ Configuration Options

```python
generation_config = {
    "temperature": 0.9,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
```

### 🎓 Best Practices

1. **Long Documents**: Use Flash for documents up to 1M tokens
2. **Multimodal**: Gemini excels at processing images, videos, audio
3. **JSON Output**: Use `response_mime_type="application/json"` for structured data
4. **Function Calling**: Built-in support for tool use
5. **Caching**: Use context caching for repeated long contexts (can reduce costs by 90%)

### 📊 Rate Limits (Free Tier)
- Flash: 15 requests/minute, 1M requests/day
- Pro: 2 requests/minute, 50 requests/day

### 💡 Pro Tips

```python
# Enable caching for long contexts
cached_content = genai.caching.CachedContent.create(
    model='gemini-1.5-flash',
    contents=[large_document],
    ttl=datetime.timedelta(hours=1)
)

# Use with streaming for better UX
response = model.generate_content(
    "Your prompt",
    stream=True
)
for chunk in response:
    print(chunk.text, end='')
```

---

## Groq

### 🌟 Overview
- **Free Tier**: Yes (Extremely generous)
- **Best For**: Ultra-fast inference (fastest in the market)
- **Context Length**: Up to 128K tokens
- **Pricing**: FREE for many models, paid tier from $0.05/1M tokens

### 📝 Setup Steps

1. **Get API Key**
   ```
   Visit: https://console.groq.com/
   - Sign up with email or GitHub
   - Navigate to API Keys
   - Create new API key
   - Copy the key
   ```

2. **Configure in .env**
   ```bash
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. **Test Connection**
   ```python
   from groq import AsyncGroq
   
   client = AsyncGroq(api_key="your_key")
   response = await client.chat.completions.create(
       model="llama-3.1-70b-versatile",
       messages=[{"role": "user", "content": "Hello!"}]
   )
   print(response.choices[0].message.content)
   ```

### 🎯 Available Models (All FREE on Free Tier!)

| Model | Speed | Quality | Context | Tokens/Min (Free) | Best For |
|-------|-------|---------|---------|-------------------|----------|
| `llama-3.1-8b-instant` | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | 128K | 30,000 | Ultra-fast tasks |
| `llama-3.1-70b-versatile` | ⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ | 128K | 6,000 | High-quality, fast |
| `llama-3.3-70b-versatile` | ⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ | 128K | 6,000 | Latest, best |
| `mixtral-8x7b-32768` | ⚡⚡⚡ | ⭐⭐⭐⭐ | 32K | 5,000 | Reasoning |
| `gemma2-9b-it` | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | 8K | 15,000 | Efficient |

### ⚙️ Configuration Options

```python
config = {
    "temperature": 0.7,
    "max_tokens": 4096,
    "top_p": 0.9,
    "stream": True,  # Highly recommended for better UX
    "stop": None,
}
```

### 🎓 Best Practices

1. **Speed Priority**: Use Groq for tasks requiring sub-second responses
2. **Streaming**: Always use streaming for better user experience
3. **70B Model**: Best quality-to-speed ratio in the industry
4. **Batch Requests**: Can handle high throughput
5. **Long Context**: 128K context window is excellent for document analysis

### 📊 Rate Limits (Free Tier)
- 30 requests/minute per model
- 14,400 requests/day
- Token limits per model (see table above)

### 💡 Why Groq is Special

Groq uses custom LPU (Language Processing Unit) chips that make inference 10-20x faster than GPUs:
- **Latency**: 250-500 tokens/second (vs 50-100 for typical GPU)
- **Time-to-First-Token**: < 100ms
- **Cost**: Free tier is extremely generous

---

## OpenRouter

### 🌟 Overview
- **Free Tier**: 40+ completely free models
- **Best For**: Access to all models through one API
- **Context Length**: Varies by model
- **Pricing**: Free models + paid models from $0.0001/1K tokens

### 📝 Setup Steps

1. **Get API Key**
   ```
   Visit: https://openrouter.ai/
   - Sign up with email or OAuth
   - Navigate to Keys section
   - Create new API key
   - Copy the key
   ```

2. **Configure in .env**
   ```bash
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```

3. **Test Connection**
   ```python
   import aiohttp
   
   async with aiohttp.ClientSession() as session:
       async with session.post(
           "https://openrouter.ai/api/v1/chat/completions",
           headers={
               "Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"
           },
           json={
               "model": "meta-llama/llama-3.1-8b-instruct:free",
               "messages": [{"role": "user", "content": "Hello!"}]
           }
       ) as response:
           data = await response.json()
           print(data['choices'][0]['message']['content'])
   ```

### 🎯 Free Models (40+ Available!)

#### **Amazon Nova (NEW - December 2024)**
| Model | Speed | Quality | Context | Best For |
|-------|-------|---------|---------|----------|
| `amazon/nova-2-lite-v1:free` | ⚡⚡⚡ | ⭐⭐⭐ | 300K | Fast, efficient |
| `amazon/nova-micro-v1:free` | ⚡⚡⚡⚡ | ⭐⭐ | 128K | Ultra-light tasks |

#### **Meta Llama Family**
| Model | Speed | Quality | Context | Best For |
|-------|-------|---------|---------|----------|
| `meta-llama/llama-3.2-1b-instruct:free` | ⚡⚡⚡⚡ | ⭐⭐ | 128K | Tiny, fast |
| `meta-llama/llama-3.2-3b-instruct:free` | ⚡⚡⚡⚡ | ⭐⭐⭐ | 128K | Small, efficient |
| `meta-llama/llama-3.1-8b-instruct:free` | ⚡⚡⚡ | ⭐⭐⭐⭐ | 128K | Balanced |
| `meta-llama/llama-3-8b-instruct:free` | ⚡⚡⚡ | ⭐⭐⭐⭐ | 8K | General purpose |

#### **Mistral Family**
| Model | Speed | Quality | Context | Best For |
|-------|-------|---------|---------|----------|
| `mistralai/mistral-7b-instruct:free` | ⚡⚡⚡ | ⭐⭐⭐ | 32K | Efficient |
| `mistralai/mixtral-8x7b-instruct:free` | ⚡⚡ | ⭐⭐⭐⭐ | 32K | High quality |

#### **Google Gemma**
| Model | Speed | Quality | Context | Best For |
|-------|-------|---------|---------|----------|
| `google/gemma-2-9b-it:free` | ⚡⚡⚡ | ⭐⭐⭐⭐ | 8K | Efficient, quality |
| `google/gemma-7b-it:free` | ⚡⚡⚡ | ⭐⭐⭐ | 8K | General tasks |

#### **Microsoft Phi**
| Model | Speed | Quality | Context | Best For |
|-------|-------|---------|---------|----------|
| `microsoft/phi-3-mini-128k-instruct:free` | ⚡⚡⚡⚡ | ⭐⭐⭐ | 128K | Long context |
| `microsoft/phi-3-medium-128k-instruct:free` | ⚡⚡⚡ | ⭐⭐⭐⭐ | 128K | Balanced, long |

#### **Qwen (Multilingual)**
| Model | Speed | Quality | Context | Best For |
|-------|-------|---------|---------|----------|
| `qwen/qwen-2-7b-instruct:free` | ⚡⚡⚡ | ⭐⭐⭐⭐ | 32K | Multilingual |
| `qwen/qwen-2.5-7b-instruct:free` | ⚡⚡⚡ | ⭐⭐⭐⭐ | 32K | Latest, better |

#### **Specialized Models**
| Model | Specialization | Best For |
|-------|---------------|----------|
| `deepseek/deepseek-coder-6.7b-instruct:free` | Code | Code generation, review |
| `nousresearch/hermes-3-llama-3.1-405b:free` | Long context | Document analysis |
| `liquid/lfm-40b:free` | Fast inference | Speed-critical tasks |

### ⚙️ Configuration Options

```python
config = {
    "model": "meta-llama/llama-3.1-8b-instruct:free",
    "temperature": 0.7,
    "max_tokens": 2048,
    "top_p": 0.9,
    "top_k": 40,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "route": "fallback",  # Try free models first, fallback to paid if needed
    "transforms": ["middleware"]  # Enable OpenRouter features
}
```

### 🎓 Best Practices

1. **Start with Free**: Always try free models first
2. **Model Routing**: Use `route: "fallback"` for automatic fallbacks
3. **Monitor Credits**: Free models have usage limits (resets monthly)
4. **Batch Similar Requests**: Group by model for efficiency
5. **Cache Results**: OpenRouter has built-in caching

### 📊 Rate Limits (Free Models)
- Varies by model
- Typically: 20-60 requests/minute
- Daily limits apply (check OpenRouter dashboard)

### 💡 Pro Tips

```python
# Get list of all available models
async with aiohttp.ClientSession() as session:
    async with session.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"}
    ) as response:
        models = await response.json()
        free_models = [m for m in models['data'] if ':free' in m['id']]
        print(f"Found {len(free_models)} free models")

# Use provider preferences
config = {
    "provider": {
        "order": ["Together", "Replicate", "Lepton"],  # Try these providers first
        "allow_fallbacks": True
    }
}
```

---

## HuggingFace

### 🌟 Overview
- **Free Tier**: Yes (Inference API)
- **Best For**: Open-source models, experimentation
- **Context Length**: Varies (typically 2K-8K)
- **Pricing**: Free for Inference API, pay for dedicated endpoints

### 📝 Setup Steps

1. **Get API Token**
   ```
   Visit: https://huggingface.co/settings/tokens
   - Sign up/login
   - Create new token (read access)
   - Copy the token
   ```

2. **Configure in .env**
   ```bash
   HF_API_KEY=your_huggingface_token_here
   ```

3. **Test Connection**
   ```python
   from huggingface_hub import AsyncInferenceClient
   
   client = AsyncInferenceClient(token="your_token")
   response = await client.text_generation(
       "Write a poem about AI",
       model="meta-llama/Meta-Llama-3-8B-Instruct"
   )
   print(response)
   ```

### 🎯 Available Models (Free Inference API)

| Model | Speed | Quality | Best For |
|-------|-------|---------|----------|
| `meta-llama/Meta-Llama-3-8B-Instruct` | ⚡⚡ | ⭐⭐⭐⭐ | General purpose |
| `mistralai/Mistral-7B-Instruct-v0.2` | ⚡⚡ | ⭐⭐⭐ | Fast inference |
| `microsoft/phi-2` | ⚡⚡⚡ | ⭐⭐⭐ | Small, efficient |
| `bigcode/starcoder2-15b` | ⚡⚡ | ⭐⭐⭐⭐ | Code generation |
| `stabilityai/stablelm-2-1_6b` | ⚡⚡⚡ | ⭐⭐ | Lightweight |

### ⚙️ Configuration Options

```python
config = {
    "max_new_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "do_sample": True,
}
```

### 📊 Rate Limits (Free Tier)
- Inference API: Rate-limited, shared compute
- May queue during high usage
- Best for experimentation, not production

### 💡 Pro Tips

```python
# Use dedicated inference endpoints for production
from huggingface_hub import InferenceEndpoint

endpoint = InferenceEndpoint(
    "my-endpoint",
    namespace="my-namespace",
    repository="meta-llama/Meta-Llama-3-8B-Instruct",
    framework="pytorch",
    accelerator="gpu",
    instance_size="medium",
    instance_type="g5.2xlarge",
    token="your_token"
)

# Deploy and use
endpoint.create()
endpoint.wait()
response = endpoint.client.text_generation("Hello!")
```

---

## Model Comparison Table

### By Use Case

| Use Case | Best Free Model | Best Paid Model | Reasoning |
|----------|----------------|-----------------|-----------|
| **Code Generation** | `deepseek/deepseek-coder:free` (OpenRouter) | `claude-3.5-sonnet` | Specialized training |
| **Fast Responses** | `llama-3.1-8b-instant` (Groq) | `gpt-4o-mini` | Ultra-fast inference |
| **Long Documents** | `gemini-1.5-flash` | `gemini-1.5-pro` | 1M+ context |
| **Reasoning** | `llama-3.3-70b-versatile` (Groq) | `o1-preview` | Deep reasoning |
| **Cost Optimization** | `mistral-7b:free` (OpenRouter) | `mistral-small` | Efficiency |
| **Multimodal** | `gemini-1.5-flash` | `gpt-4o` | Image/video support |
| **General Purpose** | `llama-3.1-8b:free` | `claude-3.5-sonnet` | Balanced |

### By Performance

| Metric | Champion | Runner-up | Third Place |
|--------|----------|-----------|-------------|
| **Speed** | Groq (all models) | Mistral Small | Gemini Flash |
| **Quality** | Gemini Pro | Claude 3.5 | GPT-4o |
| **Context** | Gemini (2M) | Claude (200K) | GPT-4 Turbo (128K) |
| **Cost** | OpenRouter Free | Groq Free | Gemini Flash |
| **Multilingual** | Qwen | Gemini | GPT-4 |

---

## Quick Selection Guide

### Choose Groq if:
- ✅ Speed is critical (sub-second responses)
- ✅ You want free, high-quality inference
- ✅ You need consistent low latency

### Choose Gemini if:
- ✅ You have long documents (1M+ tokens)
- ✅ You need multimodal capabilities
- ✅ You want the best free context window

### Choose OpenRouter if:
- ✅ You want to try many models
- ✅ You need automatic fallbacks
- ✅ You want unified API for all providers

### Choose Mistral if:
- ✅ You need efficient European AI
- ✅ You want good quality at low cost
- ✅ Code generation is priority (Codestral)

### Choose HuggingFace if:
- ✅ You want open-source models
- ✅ You're experimenting/researching
- ✅ You need custom fine-tuned models

---

## Cost Optimization Strategies

### 1. **Tier Your Requests**
```python
def select_model(complexity, urgency):
    if urgency == "high":
        return "groq/llama-3.1-8b-instant"  # Fast & free
    elif complexity == "low":
        return "openrouter/mistral-7b:free"  # Free
    elif complexity == "high":
        return "gemini-1.5-pro"  # Best quality
    else:
        return "gemini-1.5-flash"  # Balanced
```

### 2. **Use Caching**
```python
# Implement semantic caching
cache_key = get_embedding(prompt)
if cached := cache.get(cache_key, threshold=0.95):
    return cached
```

### 3. **Batch Similar Requests**
```python
# Group requests by model
batches = group_by_model(requests)
for model, batch in batches:
    results = await provider.batch_complete(model, batch)
```

### 4. **Fallback Chain**
```python
models = [
    "free_model_1",
    "free_model_2",
    "paid_cheap_model",
    "paid_premium_model"
]

for model in models:
    try:
        result = await execute(model, prompt)
        if quality_check(result):
            return result
    except:
        continue
```

---

## Troubleshooting

### Common Issues

#### "Invalid API Key"
- Verify key is copied correctly (no spaces)
- Check if key is active in provider dashboard
- Ensure environment variable is loaded

#### "Rate Limit Exceeded"
- Wait for rate limit reset (usually 60 seconds)
- Upgrade to paid tier
- Implement request queuing

#### "Model Not Found"
- Check model ID spelling
- Verify model is available in your region
- Some models require special access

#### "Context Length Exceeded"
- Use model with larger context window
- Implement context truncation
- Use summarization for long inputs

---

## Next Steps

1. [Model Selection Strategy](./model_selection.md)
2. [Multi-Model Execution](../architecture/multi_model_documentation.md)
3. [Cost Optimization](./cost_optimization.md)
4. [Performance Benchmarking](../architecture/performance_benchmarking.md)

---

**Updated**: December 2024  
**Maintainer**: YMERA Team
