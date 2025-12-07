========================================
PHASE 6 - GEMINI: FRONTEND ANALYSIS & INTEGRATION PLAN
========================================

=== YOUR IDENTITY ===
Your name: GEMINI
Your role: Frontend analyst and integration architect
Your phase: 6 (NEW - FRONTEND DISCOVERY)
Your workspace: C:\Users\Mohamed Mansour\Desktop\YmeraRefactor\

=== CONTEXT ===
✅ Phase 2B: API Gateway created
⚠️ MISSING: Frontend inventory, React integration plan

**CRITICAL**: Need to analyze existing React frontend and plan integration with new API!

=== YOUR MISSION ===
1. **Inventory existing React frontend** from SOURCE_DIR
2. **Analyze frontend architecture** and dependencies
3. **Create integration plan** between React and new API
4. **Document API contract** (endpoints, schemas, auth flow)
5. **Plan frontend refactoring** if needed
6. **Create migration strategy** for frontend

This is a **DISCOVERY AND PLANNING** phase - no code yet, just comprehensive analysis.

=== SOURCE DIRECTORY (ANALYZE) ===
Location: C:\Users\Mohamed Mansour\Desktop\QoderAgentFiles\

Analyze the React frontend (if exists):
```
QoderAgentFiles\
├── frontend\
│   ├── src\
│   │   ├── components\
│   │   ├── pages\
│   │   ├── services\
│   │   ├── utils\
│   │   ├── hooks\
│   │   ├── contexts\
│   │   └── App.jsx
│   ├── public\
│   ├── package.json
│   └── ... (other frontend files)
```

=== STEP-BY-STEP INSTRUCTIONS ===

## STEP 1: FRONTEND INVENTORY (30 minutes)

### 1.1 Scan Frontend Directory Structure

Create comprehensive inventory:

**File: _reports/gemini/phase6_frontend_inventory.md**

```markdown
# YMERA Frontend Inventory Report
Phase: 6 | Agent: gemini | Created: 2024-11-30

## Executive Summary
- **Frontend Framework**: React [version]
- **Build Tool**: Vite/Webpack/CRA
- **State Management**: Redux/Context/Zustand/None
- **Routing**: React Router [version]
- **UI Library**: Material-UI/Ant Design/Tailwind/Custom
- **Total Components**: [X] components
- **Total Pages**: [Y] pages
- **API Integration**: Axios/Fetch/Custom

---

## Directory Structure

```
frontend/
├── src/
│   ├── components/          ([X] components)
│   │   ├── Header.jsx
│   │   ├── Sidebar.jsx
│   │   ├── AgentCard.jsx
│   │   └── ... (list all)
│   │
│   ├── pages/               ([Y] pages)
│   │   ├── Dashboard.jsx
│   │   ├── AgentExecution.jsx
│   │   ├── Settings.jsx
│   │   └── ... (list all)
│   │
│   ├── services/            (API services)
│   │   ├── api.js           (API client)
│   │   ├── agentService.js  (Agent API calls)
│   │   ├── authService.js   (Auth API calls)
│   │   └── ... (list all)
│   │
│   ├── hooks/               (Custom hooks)
│   │   ├── useAgent.js
│   │   ├── useAuth.js
│   │   └── ... (list all)
│   │
│   ├── contexts/            (React contexts)
│   │   ├── AuthContext.jsx
│   │   ├── AgentContext.jsx
│   │   └── ... (list all)
│   │
│   ├── utils/               (Utility functions)
│   │   ├── formatters.js
│   │   ├── validators.js
│   │   └── ... (list all)
│   │
│   ├── styles/              (CSS/SCSS files)
│   │   ├── global.css
│   │   ├── variables.css
│   │   └── ... (list all)
│   │
│   ├── App.jsx              (Main app component)
│   ├── main.jsx             (Entry point)
│   └── routes.jsx           (Route definitions)
│
├── public/
│   ├── index.html
│   ├── favicon.ico
│   └── ... (list all)
│
├── package.json
├── vite.config.js / webpack.config.js
├── .env.example
└── README.md
```

---

## Technology Stack Analysis

### Core Technologies
| Technology | Version | Purpose | Status |
|------------|---------|---------|--------|
| React | [version] | UI Framework | ✅ Current/⚠️ Outdated |
| React Router | [version] | Routing | ✅ Current/⚠️ Outdated |
| [State Mgmt] | [version] | State Management | ✅/⚠️/❌ |
| [UI Library] | [version] | UI Components | ✅/⚠️/❌ |
| Axios/Fetch | [version] | HTTP Client | ✅/⚠️/❌ |

### Dependencies Analysis
**Production Dependencies**: [X] packages
**Dev Dependencies**: [Y] packages
**Outdated Packages**: [Z] packages
**Security Vulnerabilities**: [A] vulnerabilities

**Critical Dependencies**:
```json
{
  "react": "^18.x.x",
  "react-router-dom": "^6.x.x",
  "axios": "^1.x.x",
  // ... list key dependencies with versions
}
```

---

## Component Inventory

### Layout Components
| Component | File | Props | State | API Calls | Notes |
|-----------|------|-------|-------|-----------|-------|
| Header | Header.jsx | {...} | {...} | Yes/No | [notes] |
| Sidebar | Sidebar.jsx | {...} | {...} | Yes/No | [notes] |
| Footer | Footer.jsx | {...} | {...} | Yes/No | [notes] |

### Feature Components
| Component | File | Purpose | Complexity | Reusability |
|-----------|------|---------|------------|-------------|
| AgentCard | AgentCard.jsx | Display agent info | Low/Med/High | High/Med/Low |
| AgentExecutor | AgentExecutor.jsx | Execute agents | Low/Med/High | High/Med/Low |
| ResultDisplay | ResultDisplay.jsx | Show results | Low/Med/High | High/Med/Low |

### Form Components
[List all form components with validation logic]

### Utility Components
[List all utility/helper components]

---

## Page Inventory

| Page | Route | Components Used | API Endpoints | Auth Required |
|------|-------|----------------|---------------|---------------|
| Dashboard | / | Header, Sidebar, [...] | /api/v1/... | Yes/No |
| Agent Execution | /agents/execute | [...] | /api/v1/agents/execute | Yes/No |
| Settings | /settings | [...] | /api/v1/... | Yes |

---

## API Integration Analysis

### Current API Endpoints Used
```javascript
// Example from agentService.js
const endpoints = {
  listAgents: 'GET /agents',
  executeAgent: 'POST /agents/execute',
  getAgentStatus: 'GET /agents/status/:id',
  // ... list all current endpoints
}
```

### API Client Configuration
```javascript
// Current API base URL
const API_BASE_URL = 'http://localhost:5000/api';  // OLD

// Authentication method
// - Bearer token? API key? Session?

// Error handling
// - How are errors currently handled?

// Request interceptors
// - What preprocessing happens?

// Response interceptors
// - What postprocessing happens?
```

---

## State Management Analysis

### Current State Structure
[If using Redux/Zustand/etc, document state shape]

```javascript
// Example state structure
{
  auth: {
    user: null,
    token: null,
    isAuthenticated: false
  },
  agents: {
    list: [],
    current: null,
    loading: false,
    error: null
  },
  // ... document all state slices
}
```

### Data Flow
```
User Action → Component → Service → API → Response → State Update → Re-render
```

---

## Routing Analysis

### Current Routes
```javascript
const routes = [
  { path: '/', component: 'Dashboard', protected: true },
  { path: '/login', component: 'Login', protected: false },
  { path: '/agents', component: 'AgentList', protected: true },
  { path: '/agents/:id', component: 'AgentDetail', protected: true },
  // ... list all routes
]
```

### Navigation Flow
```
Login → Dashboard → Agents → Agent Detail → Execution → Results
```

---

## Authentication Flow

### Current Auth Implementation
1. Login process: [describe]
2. Token storage: LocalStorage/SessionStorage/Cookie
3. Token refresh: Yes/No/How
4. Protected routes: How implemented
5. Logout process: [describe]

### Auth State Management
[How is auth state managed across app]

---

## Data Fetching Patterns

### Pattern Used
- [ ] useEffect + useState
- [ ] React Query / SWR
- [ ] Redux Thunk / Saga
- [ ] Custom hooks
- [ ] Other: [specify]

### Example Data Fetching
```javascript
// Example from useAgent.js or similar
const useAgent = (agentId) => {
  const [agent, setAgent] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    // ... fetch logic
  }, [agentId]);
  
  return { agent, loading, error };
}
```

---

## UI/UX Analysis

### Design System
- **Color Palette**: [primary, secondary, etc.]
- **Typography**: [fonts used]
- **Spacing System**: [margin/padding scale]
- **Breakpoints**: [responsive breakpoints]
- **Component Library**: Material-UI/Ant Design/Custom

### Accessibility
- [ ] Semantic HTML
- [ ] ARIA labels
- [ ] Keyboard navigation
- [ ] Screen reader support
- [ ] Color contrast compliance

### Performance
- **Bundle Size**: [X]MB
- **First Contentful Paint**: [X]ms
- **Time to Interactive**: [Y]ms
- **Lighthouse Score**: [Z]/100

---

## Code Quality

### Linting/Formatting
- **ESLint**: Yes/No - Config: [...]
- **Prettier**: Yes/No - Config: [...]
- **TypeScript**: Yes/No - Coverage: [X]%

### Testing
- **Unit Tests**: [X] tests - Framework: Jest/Vitest
- **Integration Tests**: [Y] tests
- **E2E Tests**: [Z] tests - Framework: Cypress/Playwright
- **Test Coverage**: [X]%

### Code Standards
- **Component Structure**: Functional/Class/Mixed
- **Props Validation**: PropTypes/TypeScript/None
- **File Naming**: kebab-case/camelCase/PascalCase
- **Folder Structure**: Feature-based/Type-based/Mixed

---

## Issues Identified

### Critical Issues
1. [Issue 1]
   - **Impact**: High/Med/Low
   - **Description**: [...]
   - **Recommendation**: [...]

### Medium Priority Issues
[List issues]

### Low Priority Issues
[List issues]

---

## Dependencies on Old Backend

### Hardcoded References
```javascript
// List all places where old API is referenced
const OLD_API_URL = 'http://localhost:5000';  // Line X in file Y
// ... list all occurrences
```

### Breaking Changes Expected
1. [Change 1]
   - **Old**: [old behavior]
   - **New**: [new behavior]
   - **Impact**: [files affected]

---

## Recommendations

### Immediate Actions
1. [ ] Update API base URL
2. [ ] Implement new auth flow (JWT)
3. [ ] Update all API service calls
4. [ ] Add error handling for new API errors
5. [ ] Update request/response models

### Short-term Improvements
1. [ ] Add TypeScript
2. [ ] Implement React Query for data fetching
3. [ ] Add loading states
4. [ ] Improve error handling
5. [ ] Add WebSocket support for real-time updates

### Long-term Enhancements
1. [ ] Migrate to Next.js (if applicable)
2. [ ] Implement proper testing
3. [ ] Add Storybook for component documentation
4. [ ] Improve accessibility
5. [ ] Optimize performance

---

## Statistics

- **Total Files**: [X] files
- **Total Lines of Code**: ~[Y] lines
- **Total Components**: [Z] components
- **Total Pages**: [A] pages
- **Total Services**: [B] service files
- **API Endpoints Used**: [C] endpoints
- **Dependencies**: [D] packages
- **Bundle Size**: [E]MB
```

## STEP 2: API CONTRACT DOCUMENTATION (45 minutes)

**File: _reports/gemini/phase6_api_contract.md**

```markdown
# API Contract: Frontend ↔ Backend Integration
Phase: 6 | Agent: gemini | Created: 2024-11-30

## Overview
This document defines the contract between the React frontend and the new FastAPI backend.

---

## Base URLs

### Development
```
Frontend: http://localhost:3000
Backend API: http://localhost:8000
WebSocket: ws://localhost:8000
```

### Production
```
Frontend: https://ymera-app.com
Backend API: https://api.ymera-app.com
WebSocket: wss://api.ymera-app.com
```

---

## Authentication

### Login Flow
```
1. User enters credentials on /login
2. POST /api/v1/auth/login
3. Receive JWT tokens
4. Store tokens in memory (not localStorage for security)
5. Add Bearer token to all subsequent requests
6. Refresh token before expiry
```

### Auth Headers
```javascript
headers: {
  'Authorization': `Bearer ${accessToken}`,
  'Content-Type': 'application/json'
}
```

### Token Refresh
```
Access Token: 30 minutes
Refresh Token: 7 days

Before access token expires:
POST /api/v1/auth/refresh
Body: { "refresh_token": "..." }
Response: { "access_token": "...", "refresh_token": "..." }
```

---

## API Endpoints

### Authentication Endpoints

#### POST /api/v1/auth/register
**Purpose**: Register new user

**Request**:
```json
{
  "username": "string",
  "email": "string",
  "password": "string"
}
```

**Response** (201):
```json
{
  "user_id": 1,
  "username": "string",
  "email": "string"
}
```

**Errors**:
- 400: Username/email already exists
- 422: Validation error

---

#### POST /api/v1/auth/login
**Purpose**: Login and get tokens

**Request**:
```json
{
  "username": "string",
  "password": "string"
}
```

**Response** (200):
```json
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer"
}
```

**Errors**:
- 401: Invalid credentials

---

### Agent Endpoints

#### GET /api/v1/agents/
**Purpose**: List all available agents

**Auth**: Required

**Response** (200):
```json
{
  "success": true,
  "agents": [
    {
      "name": "coding_agent",
      "version": "2.0.0",
      "description": "Code generation and analysis",
      "capabilities": ["code_generation", "refactoring"],
      "status": "initialized"
    }
  ],
  "total": 5
}
```

---

#### POST /api/v1/agents/execute
**Purpose**: Execute an agent synchronously

**Auth**: Required

**Request**:
```json
{
  "agent_name": "coding_agent",
  "task_id": "optional-custom-id",
  "task_type": "code_generation",
  "parameters": {
    "language": "python",
    "prompt": "Create a FastAPI endpoint"
  }
}
```

**Response** (200):
```json
{
  "success": true,
  "task_id": "task_123",
  "status": "success",
  "result": {
    "code": "from fastapi import FastAPI...",
    "language": "python",
    "metadata": {}
  },
  "error": null,
  "metadata": {
    "execution_time": 2.5,
    "agent": "coding_agent"
  }
}
```

**Errors**:
- 404: Agent not found
- 422: Invalid parameters
- 500: Execution failed

---

#### POST /api/v1/agents/execute/async
**Purpose**: Execute an agent asynchronously (long-running tasks)

**Auth**: Required

**Request**: Same as synchronous execution

**Response** (202):
```json
{
  "success": true,
  "task_id": "celery-task-uuid",
  "message": "Task queued successfully",
  "status_url": "/api/v1/agents/status/celery-task-uuid"
}
```

---

#### GET /api/v1/agents/status/{task_id}
**Purpose**: Get status of async task

**Auth**: Required

**Response** (200):
```json
{
  "task_id": "celery-task-uuid",
  "status": "PENDING|SUCCESS|FAILURE",
  "result": {...},  // If SUCCESS
  "error": "..."    // If FAILURE
}
```

---

### Health Check Endpoints

#### GET /api/v1/health
**Purpose**: Basic health check

**Auth**: Not required

**Response** (200):
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "environment": "production",
  "timestamp": 1234567890.123
}
```

---

#### GET /api/v1/health/detailed
**Purpose**: Detailed health check

**Auth**: Required (admin only)

**Response** (200):
```json
{
  "status": "healthy|degraded",
  "services": {
    "api": "healthy",
    "database": "healthy",
    "redis": "healthy",
    "celery": "healthy"
  },
  "version": "2.0.0",
  "timestamp": 1234567890.123
}
```

---

## WebSocket Endpoints

### WS /ws/agent-updates/{client_id}
**Purpose**: Real-time agent execution updates

**Connection**:
```javascript
const ws = new WebSocket(`ws://localhost:8000/ws/agent-updates/${clientId}`);

ws.onmessage = (event) => {
  const update = JSON.parse(event.data);
  // Handle update
};
```

**Message Format**:
```json
{
  "type": "agent_started|agent_progress|agent_completed|agent_failed",
  "task_id": "task_123",
  "data": {...},
  "timestamp": 1234567890.123
}
```

---

## Error Handling

### Standard Error Response
```json
{
  "detail": "Error message",
  "status_code": 400,
  "timestamp": 1234567890.123
}
```

### Error Codes
- **400**: Bad Request - Invalid input
- **401**: Unauthorized - Missing/invalid token
- **403**: Forbidden - Insufficient permissions
- **404**: Not Found - Resource doesn't exist
- **422**: Validation Error - Pydantic validation failed
- **429**: Too Many Requests - Rate limit exceeded
- **500**: Internal Server Error - Server error
- **503**: Service Unavailable - Service down

---

## Rate Limiting

- **Default**: 100 requests per minute per IP
- **Headers**:
  ```
  X-RateLimit-Limit: 100
  X-RateLimit-Remaining: 95
  X-RateLimit-Reset: 1234567890
  ```
- **429 Response** when exceeded

---

## CORS Configuration

**Allowed Origins**:
- http://localhost:3000 (development)
- http://localhost:5173 (Vite)
- https://ymera-app.com (production)

**Allowed Methods**: GET, POST, PUT, DELETE, OPTIONS

**Allowed Headers**: Authorization, Content-Type

---

## Data Models (TypeScript Interfaces)

```typescript
// Auth Models
interface LoginRequest {
  username: string;
  password: string;
}

interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

interface User {
  user_id: number;
  username: string;
  email: string;
}

// Agent Models
interface Agent {
  name: string;
  version: string;
  description: string;
  capabilities: string[];
  status: string;
}

interface AgentExecutionRequest {
  agent_name: string;
  task_id?: string;
  task_type: string;
  parameters: Record<string, any>;
}

interface AgentExecutionResponse {
  success: boolean;
  task_id: string;
  status: 'success' | 'error' | 'partial';
  result?: any;
  error?: string;
  metadata?: Record<string, any>;
}

// WebSocket Models
interface WebSocketMessage {
  type: 'agent_started' | 'agent_progress' | 'agent_completed' | 'agent_failed';
  task_id: string;
  data: any;
  timestamp: number;
}
```

---

## Frontend Implementation Guide

### 1. API Client Setup
```javascript
// api/client.js
import axios from 'axios';

const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Request interceptor - add auth token
apiClient.interceptors.request.use(
  (config) => {
    const token = getAccessToken(); // From auth context
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor - handle errors
apiClient.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.response?.status === 401) {
      // Token expired, try refresh
      await refreshToken();
      return apiClient.request(error.config);
    }
    return Promise.reject(error);
  }
);

export default apiClient;
```

### 2. Agent Service
```javascript
// services/agentService.js
import apiClient from './client';

export const agentService = {
  listAgents: () => apiClient.get('/api/v1/agents/'),
  
  executeAgent: (request) => 
    apiClient.post('/api/v1/agents/execute', request),
  
  executeAgentAsync: (request) => 
    apiClient.post('/api/v1/agents/execute/async', request),
  
  getTaskStatus: (taskId) => 
    apiClient.get(`/api/v1/agents/status/${taskId}`)
};
```

### 3. Auth Service
```javascript
// services/authService.js
import apiClient from './client';

export const authService = {
  login: (credentials) => 
    apiClient.post('/api/v1/auth/login', credentials),
  
  register: (userData) => 
    apiClient.post('/api/v1/auth/register', userData),
  
  refreshToken: (refreshToken) => 
    apiClient.post('/api/v1/auth/refresh', { refresh_token: refreshToken })
};
```

### 4. WebSocket Hook
```javascript
// hooks/useWebSocket.js
import { useEffect, useRef, useState } from 'react';

export const useWebSocket = (clientId) => {
  const ws = useRef(null);
  const [messages, setMessages] = useState([]);
  const [connected, setConnected] = useState(false);
  
  useEffect(() => {
    const wsUrl = `ws://localhost:8000/ws/agent-updates/${clientId}`;
    ws.current = new WebSocket(wsUrl);
    
    ws.current.onopen = () => setConnected(true);
    ws.current.onclose = () => setConnected(false);
    ws.current.onmessage = (event) => {
      const message = JSON.parse(event.data);
      setMessages(prev => [...prev, message]);
    };
    
    return () => ws.current?.close();
  }, [clientId]);
  
  const sendMessage = (data) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(data));
    }
  };
  
  return { messages, connected, sendMessage };
};
```

---

## Migration Checklist

### Phase 1: Update API Client
- [ ] Update base URL to new API
- [ ] Implement JWT token handling
- [ ] Add request/response interceptors
- [ ] Update error handling

### Phase 2: Update Services
- [ ] Update all API endpoint paths
- [ ] Update request/response models
- [ ] Add async task support
- [ ] Implement WebSocket connection

### Phase 3: Update Components
- [ ] Update auth flow in Login component
- [ ] Update agent execution in Agent components
- [ ] Add loading states for async operations
- [ ] Add real-time updates via WebSocket

### Phase 4: Testing
- [ ] Test all API endpoints
- [ ] Test authentication flow
- [ ] Test error handling
- [ ] Test WebSocket connection

### Phase 5: Deploy
- [ ] Update environment variables
- [ ] Deploy frontend
- [ ] Verify production API connection
- [ ] Monitor for errors
```

## STEP 3: INTEGRATION PLAN (30 minutes)

## STEP 4: FRONTEND REFACTORING RECOMMENDATIONS (20 minutes)

## STEP 5: MIGRATION STRATEGY (30 minutes)

## STEP 6: CREATE COMPLETION REPORT (15 minutes)

=== SUCCESS CRITERIA ===

Phase 6 is complete when:
1. ✅ Frontend inventory complete
2. ✅ API contract documented
3. ✅ Integration plan created
4. ✅ Migration strategy defined
5. ✅ Refactoring recommendations provided
6. ✅ All issues identified
7. ✅ Next steps clear
8. ✅ Completion report saved

=== ESTIMATED TIME ===
Total: ~3 hours
- Frontend inventory: 30 min
- API contract: 45 min
- Integration plan: 30 min
- Refactoring recommendations: 20 min
- Migration strategy: 30 min
- Completion report: 15 min
- Review and polish: 10 min

========================================
END OF PHASE 6 - GEMINI PROMPT
========================================