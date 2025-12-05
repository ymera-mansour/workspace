# Complete MCP (Model Context Protocol) Guide

## 🎯 Overview

Model Context Protocol (MCP) is an open standard for connecting AI models to external tools and data sources. This guide covers all available MCP servers, tools, and integration strategies.

## Table of Contents

1. [What is MCP?](#what-is-mcp)
2. [Official MCP Servers](#official-mcp-servers)
3. [Community MCP Servers](#community-mcp-servers)
4. [Custom MCP Implementation](#custom-mcp-implementation)
5. [Tool Categories](#tool-categories)
6. [Integration Guide](#integration-guide)
7. [Best Practices](#best-practices)

---

## What is MCP?

### Key Concepts

```
┌─────────────────────────────────────────────────────────┐
│                   AI Model (LLM)                        │
└─────────────────┬───────────────────────────────────────┘
                  │ MCP Protocol
                  ▼
┌─────────────────────────────────────────────────────────┐
│              MCP Server (Tool Provider)                 │
│  • Exposes capabilities                                 │
│  • Handles authentication                               │
│  • Manages rate limits                                  │
└─────────────────┬───────────────────────────────────────┘
                  │
    ┌─────────────┼─────────────┬──────────────┐
    ▼             ▼             ▼              ▼
┌────────┐  ┌────────┐  ┌──────────┐  ┌──────────┐
│  Tool  │  │  Tool  │  │   Tool   │  │   Tool   │
│  #1    │  │  #2    │  │   #3     │  │   #4     │
└────────┘  └────────┘  └──────────┘  └──────────┘
```

### Benefits

1. **Standardization**: One protocol for all tools
2. **Composability**: Mix and match tools from different providers
3. **Security**: Fine-grained permission control
4. **Extensibility**: Easy to add new tools
5. **Interoperability**: Works across different AI platforms

---

## Official MCP Servers

### 1. **Brave Search MCP** (Free Tier Available)

**Provider**: Brave  
**Category**: Web Search  
**Free Tier**: 2,000 queries/month

#### Setup

```bash
npm install @modelcontextprotocol/server-brave-search
```

**Configuration**:
```json
{
  "mcpServers": {
    "brave-search": {
      "command": "node",
      "args": [
        "node_modules/@modelcontextprotocol/server-brave-search/dist/index.js"
      ],
      "env": {
        "BRAVE_API_KEY": "your_brave_api_key"
      }
    }
  }
}
```

**Get API Key**: https://brave.com/search/api/

#### Available Tools

| Tool | Description | Cost |
|------|-------------|------|
| `brave_web_search` | Web search with ranking | Free tier: 2K/month |
| `brave_local_search` | Local business search | Free tier: 2K/month |

**Example**:
```python
result = await mcp_client.call_tool(
    "brave_web_search",
    {
        "query": "best practices for API design",
        "count": 10
    }
)
```

---

### 2. **Filesystem MCP** (Free)

**Provider**: Anthropic  
**Category**: File Operations  
**Free Tier**: Unlimited (local)

#### Setup

```bash
npm install @modelcontextprotocol/server-filesystem
```

**Configuration**:
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "node",
      "args": [
        "node_modules/@modelcontextprotocol/server-filesystem/dist/index.js",
        "/path/to/allowed/directory"
      ]
    }
  }
}
```

#### Available Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `read_file` | Read file contents | `path` |
| `write_file` | Write to file | `path`, `content` |
| `list_directory` | List directory contents | `path` |
| `create_directory` | Create new directory | `path` |
| `move_file` | Move/rename file | `source`, `destination` |
| `search_files` | Search for files | `path`, `pattern` |

**Example**:
```python
# Read configuration file
config = await mcp_client.call_tool(
    "read_file",
    {"path": "/path/to/config.json"}
)

# Write results
await mcp_client.call_tool(
    "write_file",
    {
        "path": "/path/to/output.txt",
        "content": "Results..."
    }
)
```

---

### 3. **GitHub MCP** (Free with GitHub Account)

**Provider**: Anthropic  
**Category**: Version Control  
**Free Tier**: GitHub Free/Pro features

#### Setup

```bash
npm install @modelcontextprotocol/server-github
```

**Configuration**:
```json
{
  "mcpServers": {
    "github": {
      "command": "node",
      "args": [
        "node_modules/@modelcontextprotocol/server-github/dist/index.js"
      ],
      "env": {
        "GITHUB_TOKEN": "your_github_token"
      }
    }
  }
}
```

**Get Token**: https://github.com/settings/tokens (use your GitHub Pro account!)

#### Available Tools

| Tool | Description | GitHub Pro Benefits |
|------|-------------|---------------------|
| `create_repository` | Create new repo | Private repos included |
| `create_issue` | Create issue | Advanced features |
| `create_pull_request` | Create PR | Code review tools |
| `fork_repository` | Fork repo | Unlimited forks |
| `create_branch` | Create branch | Branch protection |
| `search_repositories` | Search GitHub | Better rate limits |
| `get_file_contents` | Read files | Private repo access |
| `push_files` | Push commits | Actions minutes |

**Example**:
```python
# Create issue for bug
await mcp_client.call_tool(
    "create_issue",
    {
        "owner": "username",
        "repo": "project",
        "title": "Bug: Login fails",
        "body": "Steps to reproduce...",
        "labels": ["bug", "high-priority"]
    }
)

# Create PR
await mcp_client.call_tool(
    "create_pull_request",
    {
        "owner": "username",
        "repo": "project",
        "title": "Fix: Login validation",
        "body": "This PR fixes...",
        "head": "feature-branch",
        "base": "main"
    }
)
```

---

### 4. **PostgreSQL MCP** (Free - Self-hosted)

**Provider**: Anthropic  
**Category**: Database  
**Free Tier**: Self-hosted PostgreSQL

#### Setup

```bash
npm install @modelcontextprotocol/server-postgres
```

**Configuration**:
```json
{
  "mcpServers": {
    "postgres": {
      "command": "node",
      "args": [
        "node_modules/@modelcontextprotocol/server-postgres/dist/index.js"
      ],
      "env": {
        "POSTGRES_CONNECTION_STRING": "postgresql://user:pass@localhost:5432/db"
      }
    }
  }
}
```

#### Available Tools

| Tool | Description | Use Case |
|------|-------------|----------|
| `query` | Execute SQL query | Read data |
| `execute` | Execute SQL statement | Insert/Update/Delete |
| `list_tables` | List all tables | Schema discovery |
| `describe_table` | Get table schema | Understanding structure |

**Example**:
```python
# Query data
users = await mcp_client.call_tool(
    "query",
    {
        "sql": "SELECT * FROM users WHERE active = true LIMIT 10"
    }
)

# Insert data
await mcp_client.call_tool(
    "execute",
    {
        "sql": "INSERT INTO logs (message, level) VALUES ($1, $2)",
        "params": ["User logged in", "INFO"]
    }
)
```

---

### 5. **Puppeteer MCP** (Free - Self-hosted)

**Provider**: Anthropic  
**Category**: Web Automation  
**Free Tier**: Unlimited (local)

#### Setup

```bash
npm install @modelcontextprotocol/server-puppeteer
```

**Configuration**:
```json
{
  "mcpServers": {
    "puppeteer": {
      "command": "node",
      "args": [
        "node_modules/@modelcontextprotocol/server-puppeteer/dist/index.js"
      ]
    }
  }
}
```

#### Available Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `puppeteer_navigate` | Navigate to URL | `url` |
| `puppeteer_screenshot` | Take screenshot | `selector`, `path` |
| `puppeteer_click` | Click element | `selector` |
| `puppeteer_fill` | Fill form field | `selector`, `value` |
| `puppeteer_evaluate` | Run JavaScript | `script` |

**Example**:
```python
# Navigate and screenshot
await mcp_client.call_tool(
    "puppeteer_navigate",
    {"url": "https://example.com"}
)

screenshot = await mcp_client.call_tool(
    "puppeteer_screenshot",
    {"path": "/tmp/screenshot.png"}
)
```

---

## Community MCP Servers

### 6. **Slack MCP** (Free Tier)

**Provider**: Community  
**Category**: Communication  
**Free Tier**: Slack Free workspace

#### Setup

```bash
npm install @modelcontextprotocol/server-slack
```

**Configuration**:
```json
{
  "mcpServers": {
    "slack": {
      "command": "node",
      "args": ["node_modules/@modelcontextprotocol/server-slack/dist/index.js"],
      "env": {
        "SLACK_BOT_TOKEN": "xoxb-your-token",
        "SLACK_TEAM_ID": "T1234567890"
      }
    }
  }
}
```

#### Tools
- `post_message`: Send message to channel
- `list_channels`: Get all channels
- `get_channel_history`: Read messages

---

### 7. **Memory MCP** (Free - Local)

**Provider**: Community  
**Category**: State Management  
**Free Tier**: Unlimited (local storage)

#### Setup

```bash
npm install @modelcontextprotocol/server-memory
```

**Configuration**:
```json
{
  "mcpServers": {
    "memory": {
      "command": "node",
      "args": ["node_modules/@modelcontextprotocol/server-memory/dist/index.js"]
    }
  }
}
```

#### Tools
- `store_memory`: Store key-value pair
- `retrieve_memory`: Get stored value
- `list_memories`: List all keys
- `delete_memory`: Remove entry

**Example**:
```python
# Store user preference
await mcp_client.call_tool(
    "store_memory",
    {
        "key": "user_123_theme",
        "value": "dark",
        "ttl": 86400  # 24 hours
    }
)

# Retrieve later
theme = await mcp_client.call_tool(
    "retrieve_memory",
    {"key": "user_123_theme"}
)
```

---

### 8. **Time MCP** (Free)

**Provider**: Community  
**Category**: Utility  
**Free Tier**: Unlimited

#### Setup

```bash
npm install @modelcontextprotocol/server-time
```

#### Tools
- `get_current_time`: Current timestamp
- `convert_timezone`: Convert time zones
- `calculate_duration`: Time calculations
- `format_date`: Date formatting

---

### 9. **Google Drive MCP** (Free Tier)

**Provider**: Community  
**Category**: Cloud Storage  
**Free Tier**: 15GB with Google account

#### Setup

```bash
npm install @modelcontextprotocol/server-gdrive
```

**Configuration**:
```json
{
  "mcpServers": {
    "gdrive": {
      "command": "node",
      "args": ["node_modules/@modelcontextprotocol/server-gdrive/dist/index.js"],
      "env": {
        "GOOGLE_CLIENT_ID": "your_client_id",
        "GOOGLE_CLIENT_SECRET": "your_client_secret",
        "GOOGLE_REFRESH_TOKEN": "your_refresh_token"
      }
    }
  }
}
```

#### Tools
- `upload_file`: Upload to Drive
- `download_file`: Download from Drive
- `list_files`: List Drive files
- `share_file`: Set permissions

---

### 10. **Sequential Thinking MCP** (Free)

**Provider**: Community  
**Category**: Reasoning  
**Free Tier**: Unlimited

#### Setup

```bash
npm install @modelcontextprotocol/server-sequential-thinking
```

#### Tools
- `think_step_by_step`: Break down complex problem
- `analyze_dependencies`: Find task dependencies
- `create_plan`: Generate action plan

---

## Tool Categories

### Web & Search Tools
| MCP Server | Primary Use | Free Tier |
|-----------|-------------|-----------|
| Brave Search | Web search, SEO | 2K queries/month |
| Puppeteer | Web scraping, testing | Unlimited |
| Tavily | AI-powered search | 1K/month |

### Development Tools
| MCP Server | Primary Use | Free Tier |
|-----------|-------------|-----------|
| GitHub | Code management, CI/CD | GitHub Free/Pro |
| GitLab | Similar to GitHub | GitLab Free |
| Filesystem | File operations | Unlimited |

### Data Tools
| MCP Server | Primary Use | Free Tier |
|-----------|-------------|-----------|
| PostgreSQL | SQL database | Self-hosted |
| SQLite | Lightweight DB | Unlimited |
| Google Sheets | Spreadsheets | 15GB Google Drive |

### Communication Tools
| MCP Server | Primary Use | Free Tier |
|-----------|-------------|-----------|
| Slack | Team chat | Free workspace |
| Discord | Community chat | Unlimited |
| Email (SMTP) | Email sending | Depends on provider |

### Utility Tools
| MCP Server | Primary Use | Free Tier |
|-----------|-------------|-----------|
| Memory | State management | Unlimited |
| Time | Date/time operations | Unlimited |
| Sequential Thinking | Planning | Unlimited |

---

## Integration Guide

### Step 1: Install MCP Servers

```bash
# Install all desired servers
npm install \
  @modelcontextprotocol/server-brave-search \
  @modelcontextprotocol/server-filesystem \
  @modelcontextprotocol/server-github \
  @modelcontextprotocol/server-postgres \
  @modelcontextprotocol/server-puppeteer
```

### Step 2: Configure MCP

Create `mcp_config.json`:

```json
{
  "mcpServers": {
    "brave-search": {
      "command": "node",
      "args": ["node_modules/@modelcontextprotocol/server-brave-search/dist/index.js"],
      "env": {
        "BRAVE_API_KEY": "${BRAVE_API_KEY}"
      }
    },
    "filesystem": {
      "command": "node",
      "args": [
        "node_modules/@modelcontextprotocol/server-filesystem/dist/index.js",
        "${WORKSPACE_DIR}"
      ]
    },
    "github": {
      "command": "node",
      "args": ["node_modules/@modelcontextprotocol/server-github/dist/index.js"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

### Step 3: Initialize MCP Client

```python
from mcp import Client
import json

# Load configuration
with open('mcp_config.json') as f:
    config = json.load(f)

# Initialize client
mcp_client = Client(config)
await mcp_client.connect()

# List available tools
tools = await mcp_client.list_tools()
print(f"Available tools: {[t['name'] for t in tools]}")
```

### Step 4: Use Tools in Workflows

```python
class WorkflowExecutor:
    def __init__(self, mcp_client):
        self.mcp = mcp_client
    
    async def research_and_code(self, topic):
        # Step 1: Research with Brave
        search_results = await self.mcp.call_tool(
            "brave_web_search",
            {"query": f"best practices {topic}", "count": 5}
        )
        
        # Step 2: Generate code with AI
        code = await self.generate_code(topic, search_results)
        
        # Step 3: Save to file
        await self.mcp.call_tool(
            "write_file",
            {
                "path": f"./{topic}.py",
                "content": code
            }
        )
        
        # Step 4: Commit to GitHub
        await self.mcp.call_tool(
            "push_files",
            {
                "owner": "username",
                "repo": "project",
                "files": {f"{topic}.py": code},
                "message": f"Add {topic} implementation"
            }
        )
        
        return {"status": "success", "file": f"{topic}.py"}
```

---

## Custom MCP Implementation

### Creating Your Own MCP Server

#### 1. Setup Project

```bash
mkdir my-mcp-server
cd my-mcp-server
npm init -y
npm install @modelcontextprotocol/sdk
```

#### 2. Implement Server

```typescript
// index.ts
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';

const server = new Server(
  {
    name: 'my-custom-server',
    version: '1.0.0',
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// Define tools
server.setRequestHandler('tools/list', async () => {
  return {
    tools: [
      {
        name: 'my_custom_tool',
        description: 'Does something useful',
        inputSchema: {
          type: 'object',
          properties: {
            param1: { type: 'string' },
            param2: { type: 'number' }
          },
          required: ['param1']
        }
      }
    ]
  };
});

// Implement tool logic
server.setRequestHandler('tools/call', async (request) => {
  if (request.params.name === 'my_custom_tool') {
    // Your tool logic here
    return {
      content: [
        {
          type: 'text',
          text: 'Tool executed successfully'
        }
      ]
    };
  }
});

// Start server
const transport = new StdioServerTransport();
await server.connect(transport);
```

#### 3. Build and Test

```bash
npm run build
node dist/index.js
```

---

## Best Practices

### 1. **Security**

```python
# Always validate tool inputs
def validate_file_path(path: str) -> bool:
    # Prevent directory traversal
    if '..' in path:
        return False
    # Ensure path is within allowed directory
    if not path.startswith(ALLOWED_DIR):
        return False
    return True

# Use environment variables for secrets
API_KEY = os.getenv('API_KEY')
if not API_KEY:
    raise ValueError("API_KEY not set")
```

### 2. **Error Handling**

```python
async def safe_tool_call(tool_name, params):
    try:
        result = await mcp_client.call_tool(tool_name, params)
        return {"success": True, "data": result}
    except ToolNotFoundError:
        logger.error(f"Tool {tool_name} not available")
        return {"success": False, "error": "Tool not found"}
    except ToolExecutionError as e:
        logger.error(f"Tool execution failed: {e}")
        return {"success": False, "error": str(e)}
```

### 3. **Rate Limiting**

```python
from asyncio import Semaphore

class RateLimitedMCP:
    def __init__(self, mcp_client, max_concurrent=5):
        self.mcp = mcp_client
        self.semaphore = Semaphore(max_concurrent)
    
    async def call_tool(self, name, params):
        async with self.semaphore:
            return await self.mcp.call_tool(name, params)
```

### 4. **Caching**

```python
from functools import lru_cache
import hashlib

class CachedMCP:
    def __init__(self, mcp_client):
        self.mcp = mcp_client
        self.cache = {}
    
    async def call_tool(self, name, params):
        # Create cache key
        cache_key = f"{name}:{hashlib.md5(str(params).encode()).hexdigest()}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        result = await self.mcp.call_tool(name, params)
        self.cache[cache_key] = result
        return result
```

### 5. **Monitoring**

```python
import time
from dataclasses import dataclass

@dataclass
class ToolMetrics:
    tool_name: str
    call_count: int = 0
    total_time: float = 0.0
    error_count: int = 0

class MonitoredMCP:
    def __init__(self, mcp_client):
        self.mcp = mcp_client
        self.metrics = {}
    
    async def call_tool(self, name, params):
        if name not in self.metrics:
            self.metrics[name] = ToolMetrics(tool_name=name)
        
        start = time.time()
        try:
            result = await self.mcp.call_tool(name, params)
            self.metrics[name].call_count += 1
            self.metrics[name].total_time += time.time() - start
            return result
        except Exception as e:
            self.metrics[name].error_count += 1
            raise
```

---

## Tool Selection Strategy

### For Research Tasks
```python
tools = ["brave_web_search", "puppeteer_navigate", "puppeteer_screenshot"]
```

### For Development Tasks
```python
tools = ["filesystem", "github", "postgres"]
```

### For Data Analysis
```python
tools = ["postgres", "google_sheets", "sequential_thinking"]
```

### For Communication
```python
tools = ["slack", "email", "github"]
```

---

## Next Steps

1. [Workflow Implementation](./workflow_guide.md)
2. [Tool Orchestration](./tool_orchestration.md)
3. [Custom Tool Development](./custom_tools.md)

---

**Updated**: December 2024  
**Maintainer**: YMERA Team
