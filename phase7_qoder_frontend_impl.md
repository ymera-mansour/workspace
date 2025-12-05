========================================
PHASE 7 - QODER: FRONTEND INTEGRATION IMPLEMENTATION
========================================

=== YOUR IDENTITY ===
Your name: QODER
Your role: Frontend integration developer
Your phase: 7 (NEW - FRONTEND IMPLEMENTATION)
Your workspace: C:\Users\Mohamed Mansour\Desktop\YmeraRefactor\

=== CONTEXT ===
✅ Phase 2B: API Gateway created
✅ Phase 6: Frontend analyzed and integration plan created

**NOW**: Implement the actual frontend integration with new API!

=== YOUR MISSION ===
Based on Gemini's analysis and integration plan:
1. **Refactor existing React frontend** or create new one
2. **Implement API client** for new FastAPI backend
3. **Update all API service calls** to use new endpoints
4. **Implement JWT authentication** flow
5. **Add WebSocket support** for real-time updates
6. **Update components** to use new API
7. **Add proper error handling** and loading states
8. **Test integration** end-to-end

=== SOURCE DIRECTORY (READ) ===
Location: C:\Users\Mohamed Mansour\Desktop\QoderAgentFiles\frontend\

Read existing frontend code from SOURCE_DIR

=== TARGET DIRECTORY (WRITE) ===
Location: C:\Users\Mohamed Mansour\Desktop\YmeraRefactor\

You will create/update:
```
YmeraRefactor\
└── frontend\  (NEW/UPDATED)
    ├── src\
    │   ├── api\
    │   │   ├── client.js (axios instance with interceptors)
    │   │   ├── endpoints.js (all API endpoints)
    │   │   └── types.ts (TypeScript types for API)
    │   │
    │   ├── services\
    │   │   ├── agentService.js (agent API calls)
    │   │   ├── authService.js (auth API calls)
    │   │   └── healthService.js (health check API calls)
    │   │
    │   ├── hooks\
    │   │   ├── useAuth.js (authentication hook)
    │   │   ├── useAgent.js (agent execution hook)
    │   │   ├── useWebSocket.js (WebSocket connection)
    │   │   └── useApi.js (generic API hook)
    │   │
    │   ├── contexts\
    │   │   ├── AuthContext.jsx (auth state)
    │   │   └── AgentContext.jsx (agent state)
    │   │
    │   ├── components\
    │   │   ├── Auth\
    │   │   │   ├── LoginForm.jsx
    │   │   │   └── ProtectedRoute.jsx
    │   │   ├── Agents\
    │   │   │   ├── AgentList.jsx
    │   │   │   ├── AgentExecutor.jsx
    │   │   │   └── AgentResult.jsx
    │   │   └── Common\
    │   │       ├── LoadingSpinner.jsx
    │   │       └── ErrorBoundary.jsx
    │   │
    │   ├── pages\
    │   │   ├── Login.jsx
    │   │   ├── Dashboard.jsx
    │   │   ├── AgentExecution.jsx
    │   │   └── Settings.jsx
    │   │
    │   ├── utils\
    │   │   ├── tokenManager.js (JWT token management)
    │   │   ├── errorHandler.js (error handling)
    │   │   └── validators.js (input validation)
    │   │
    │   ├── App.jsx
    │   ├── main.jsx
    │   └── routes.jsx
    │
    ├── .env.example
    ├── package.json
    ├── vite.config.js
    └── README.md
```

=== STEP-BY-STEP INSTRUCTIONS ===

## STEP 1: SETUP PROJECT (if creating new) (15 minutes)

If creating new React app:
```bash
cd YmeraRefactor
npm create vite@latest frontend -- --template react
cd frontend
npm install axios react-router-dom jwt-decode
npm install -D @types/node
```

If updating existing:
```bash
cd YmeraRefactor
cp -r ../QoderAgentFiles/frontend .
cd frontend
npm install
```

## STEP 2: CREATE API CLIENT (30 minutes)

**File: frontend/src/api/client.js**
```javascript
// YMERA Refactoring Project
// Phase: 7 | Agent: qoder | Created: 2024-11-30
// Axios API client with interceptors

import axios from 'axios';
import { getAccessToken, getRefreshToken, setTokens, clearTokens } from '../utils/tokenManager';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// Create axios instance
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Request interceptor - add auth token
apiClient.interceptors.request.use(
  (config) => {
    const token = getAccessToken();
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    console.error('Request error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor - handle token refresh
apiClient.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;

    // If 401 and we haven't retried yet
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;

      try {
        const refreshToken = getRefreshToken();
        if (!refreshToken) {
          throw new Error('No refresh token available');
        }

        // Try to refresh token
        const response = await axios.post(`${API_BASE_URL}/api/v1/auth/refresh`, {
          refresh_token: refreshToken
        });

        const { access_token, refresh_token } = response.data;
        setTokens(access_token, refresh_token);

        // Retry original request with new token
        originalRequest.headers.Authorization = `Bearer ${access_token}`;
        return apiClient(originalRequest);
      } catch (refreshError) {
        // Refresh failed, logout user
        clearTokens();
        window.location.href = '/login';
        return Promise.reject(refreshError);
      }
    }

    return Promise.reject(error);
  }
);

export default apiClient;
```

**File: frontend/src/api/endpoints.js**
```javascript
// YMERA Refactoring Project
// Phase: 7 | Agent: qoder | Created: 2024-11-30
// API endpoint definitions

export const endpoints = {
  // Auth
  auth: {
    login: '/api/v1/auth/login',
    register: '/api/v1/auth/register',
    refresh: '/api/v1/auth/refresh',
  },
  
  // Agents
  agents: {
    list: '/api/v1/agents/',
    execute: '/api/v1/agents/execute',
    executeAsync: '/api/v1/agents/execute/async',
    status: (taskId) => `/api/v1/agents/status/${taskId}`,
  },
  
  // Health
  health: {
    basic: '/api/v1/health',
    detailed: '/api/v1/health/detailed',
  },
  
  // WebSocket
  websocket: {
    agentUpdates: (clientId) => `/ws/agent-updates/${clientId}`,
  }
};
```

## STEP 3: CREATE TOKEN MANAGER (20 minutes)

**File: frontend/src/utils/tokenManager.js**
```javascript
// YMERA Refactoring Project
// Phase: 7 | Agent: qoder | Created: 2024-11-30
// JWT token management (in-memory for security)

// Store tokens in memory, not localStorage (more secure)
let accessToken = null;
let refreshToken = null;

export const setTokens = (access, refresh) => {
  accessToken = access;
  refreshToken = refresh;
};

export const getAccessToken = () => accessToken;

export const getRefreshToken = () => refreshToken;

export const clearTokens = () => {
  accessToken = null;
  refreshToken = null;
};

// Decode JWT to check expiry
export const isTokenExpired = (token) => {
  if (!token) return true;
  
  try {
    const payload = JSON.parse(atob(token.split('.')[1]));
    const exp = payload.exp * 1000; // Convert to milliseconds
    return Date.now() >= exp;
  } catch (e) {
    return true;
  }
};

export const needsRefresh = () => {
  const token = getAccessToken();
  if (!token) return false;
  
  try {
    const payload = JSON.parse(atob(token.split('.')[1]));
    const exp = payload.exp * 1000;
    const fiveMinutes = 5 * 60 * 1000;
    return Date.now() >= (exp - fiveMinutes); // Refresh if expires in < 5 min
  } catch (e) {
    return false;
  }
};
```

## STEP 4: CREATE API SERVICES (30 minutes)

**File: frontend/src/services/authService.js**
```javascript
// YMERA Refactoring Project
// Phase: 7 | Agent: qoder | Created: 2024-11-30
// Authentication service

import apiClient from '../api/client';
import { endpoints } from '../api/endpoints';
import { setTokens, clearTokens } from '../utils/tokenManager';

export const authService = {
  /**
   * Login user
   * @param {Object} credentials - { username, password }
   * @returns {Promise<Object>} User data
   */
  async login(credentials) {
    const formData = new FormData();
    formData.append('username', credentials.username);
    formData.append('password', credentials.password);
    
    const response = await apiClient.post(endpoints.auth.login, formData, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
    });
    
    const { access_token, refresh_token } = response.data;
    setTokens(access_token, refresh_token);
    
    return response.data;
  },

  /**
   * Register new user
   * @param {Object} userData - { username, email, password }
   * @returns {Promise<Object>} Created user
   */
  async register(userData) {
    const response = await apiClient.post(endpoints.auth.register, userData);
    return response.data;
  },

  /**
   * Logout user
   */
  logout() {
    clearTokens();
  },

  /**
   * Check if user is authenticated
   * @returns {boolean}
   */
  isAuthenticated() {
    const token = getAccessToken();
    return !!token && !isTokenExpired(token);
  }
};
```

**File: frontend/src/services/agentService.js**
```javascript
// YMERA Refactoring Project
// Phase: 7 | Agent: qoder | Created: 2024-11-30
// Agent service

import apiClient from '../api/client';
import { endpoints } from '../api/endpoints';

export const agentService = {
  /**
   * Get list of all agents
   * @returns {Promise<Array>} List of agents
   */
  async listAgents() {
    const response = await apiClient.get(endpoints.agents.list);
    return response.data.agents;
  },

  /**
   * Execute agent synchronously
   * @param {Object} request - Agent execution request
   * @returns {Promise<Object>} Execution result
   */
  async executeAgent(request) {
    const response = await apiClient.post(endpoints.agents.execute, request);
    return response.data;
  },

  /**
   * Execute agent asynchronously
   * @param {Object} request - Agent execution request
   * @returns {Promise<Object>} Task info with task_id
   */
  async executeAgentAsync(request) {
    const response = await apiClient.post(endpoints.agents.executeAsync, request);
    return response.data;
  },

  /**
   * Get status of async task
   * @param {string} taskId - Task ID
   * @returns {Promise<Object>} Task status
   */
  async getTaskStatus(taskId) {
    const response = await apiClient.get(endpoints.agents.status(taskId));
    return response.data;
  }
};
```

## STEP 5: CREATE REACT HOOKS (40 minutes)

**File: frontend/src/hooks/useAuth.js**
```javascript
// YMERA Refactoring Project
// Phase: 7 | Agent: qoder | Created: 2024-11-30
// Authentication hook

import { useState, useEffect } from 'react';
import { authService } from '../services/authService';

export const useAuth = () => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Check if user is authenticated on mount
    const checkAuth = () => {
      if (authService.isAuthenticated()) {
        // TODO: Fetch user info
        setUser({ username: 'user' }); // Placeholder
      }
      setLoading(false);
    };
    
    checkAuth();
  }, []);

  const login = async (credentials) => {
    try {
      setLoading(true);
      setError(null);
      const data = await authService.login(credentials);
      setUser({ username: credentials.username });
      return data;
    } catch (err) {
      setError(err.response?.data?.detail || 'Login failed');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const logout = () => {
    authService.logout();
    setUser(null);
  };

  const register = async (userData) => {
    try {
      setLoading(true);
      setError(null);
      const data = await authService.register(userData);
      return data;
    } catch (err) {
      setError(err.response?.data?.detail || 'Registration failed');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return {
    user,
    loading,
    error,
    login,
    logout,
    register,
    isAuthenticated: !!user
  };
};
```

**File: frontend/src/hooks/useAgent.js**
```javascript
// YMERA Refactoring Project
// Phase: 7 | Agent: qoder | Created: 2024-11-30
// Agent execution hook

import { useState } from 'react';
import { agentService } from '../services/agentService';

export const useAgent = () => {
  const [agents, setAgents] = useState([]);
  const [executing, setExecuting] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const loadAgents = async () => {
    try {
      setError(null);
      const data = await agentService.listAgents();
      setAgents(data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load agents');
    }
  };

  const executeAgent = async (request) => {
    try {
      setExecuting(true);
      setError(null);
      setResult(null);
      
      const data = await agentService.executeAgent(request);
      setResult(data);
      return data;
    } catch (err) {
      const errorMsg = err.response?.data?.detail || 'Execution failed';
      setError(errorMsg);
      throw new Error(errorMsg);
    } finally {
      setExecuting(false);
    }
  };

  const executeAgentAsync = async (request) => {
    try {
      setExecuting(true);
      setError(null);
      
      const data = await agentService.executeAgentAsync(request);
      return data; // Returns task_id
    } catch (err) {
      const errorMsg = err.response?.data?.detail || 'Failed to queue task';
      setError(errorMsg);
      throw new Error(errorMsg);
    } finally {
      setExecuting(false);
    }
  };

  const checkTaskStatus = async (taskId) => {
    try {
      const data = await agentService.getTaskStatus(taskId);
      if (data.status === 'SUCCESS') {
        setResult(data.result);
      } else if (data.status === 'FAILURE') {
        setError(data.error);
      }
      return data;
    } catch (err) {
      const errorMsg = err.response?.data?.detail || 'Failed to check status';
      setError(errorMsg);
      throw new Error(errorMsg);
    }
  };

  return {
    agents,
    executing,
    result,
    error,
    loadAgents,
    executeAgent,
    executeAgentAsync,
    checkTaskStatus
  };
};
```

**File: frontend/src/hooks/useWebSocket.js**
```javascript
// YMERA Refactoring Project
// Phase: 7 | Agent: qoder | Created: 2024-11-30
// WebSocket hook for real-time updates

import { useEffect, useRef, useState, useCallback } from 'react';

const WS_BASE_URL = import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8000';

export const useWebSocket = (clientId) => {
  const ws = useRef(null);
  const [messages, setMessages] = useState([]);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!clientId) return;

    const wsUrl = `${WS_BASE_URL}/ws/agent-updates/${clientId}`;
    
    try {
      ws.current = new WebSocket(wsUrl);

      ws.current.onopen = () => {
        console.log('WebSocket connected');
        setConnected(true);
        setError(null);
      };

      ws.current.onclose = () => {
        console.log('WebSocket disconnected');
        setConnected(false);
      };

      ws.current.onerror = (event) => {
        console.error('WebSocket error:', event);
        setError('WebSocket connection error');
      };

      ws.current.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          setMessages(prev => [...prev, message]);
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      };
    } catch (err) {
      console.error('Failed to create WebSocket:', err);
      setError('Failed to establish WebSocket connection');
    }

    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, [clientId]);

  const sendMessage = useCallback((data) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(data));
    } else {
      console.warn('WebSocket not connected');
    }
  }, []);

  const clearMessages = useCallback(() => {
    setMessages([]);
  }, []);

  return { messages, connected, error, sendMessage, clearMessages };
};
```

## STEP 6: CREATE AUTH CONTEXT (25 minutes)

**File: frontend/src/contexts/AuthContext.jsx**
```javascript
// YMERA Refactoring Project
// Phase: 7 | Agent: qoder | Created: 2024-11-30
// Authentication context

import React, { createContext, useContext } from 'react';
import { useAuth as useAuthHook } from '../hooks/useAuth';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const auth = useAuthHook();

  return (
    <AuthContext.Provider value={auth}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
};
```

## STEP 7: CREATE COMPONENTS (45 minutes)

**File: frontend/src/components/Auth/LoginForm.jsx**
```javascript
// YMERA Refactoring Project
// Phase: 7 | Agent: qoder | Created: 2024-11-30
// Login form component

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';

export const LoginForm = () => {
  const [credentials, setCredentials] = useState({ username: '', password: '' });
  const { login, loading, error } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await login(credentials);
      navigate('/dashboard');
    } catch (err) {
      // Error is handled by useAuth hook
    }
  };

  return (
    <form onSubmit={handleSubmit} className="login-form">
      <h2>Login to YMERA</h2>
      
      {error && <div className="error">{error}</div>}
      
      <input
        type="text"
        placeholder="Username"
        value={credentials.username}
        onChange={(e) => setCredentials({ ...credentials, username: e.target.value })}
        required
      />
      
      <input
        type="password"
        placeholder="Password"
        value={credentials.password}
        onChange={(e) => setCredentials({ ...credentials, password: e.target.value })}
        required
      />
      
      <button type="submit" disabled={loading}>
        {loading ? 'Logging in...' : 'Login'}
      </button>
    </form>
  );
};
```

**File: frontend/src/components/Auth/ProtectedRoute.jsx**
```javascript
// YMERA Refactoring Project
// Phase: 7 | Agent: qoder | Created: 2024-11-30
// Protected route component

import React from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';

export const ProtectedRoute = ({ children }) => {
  const { isAuthenticated, loading } = useAuth();

  if (loading) {
    return <div>Loading...</div>;
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return children;
};
```

**File: frontend/src/components/Agents/AgentExecutor.jsx**
```javascript
// YMERA Refactoring Project
// Phase: 7 | Agent: qoder | Created: 2024-11-30
// Agent executor component

import React, { useState, useEffect } from 'react';
import { useAgent } from '../../hooks/useAgent';
import { useWebSocket } from '../../hooks/useWebSocket';

export const AgentExecutor = () => {
  const [selectedAgent, setSelectedAgent] = useState('');
  const [taskType, setTaskType] = useState('');
  const [parameters, setParameters] = useState('{}');
  const [clientId] = useState(() => `client_${Date.now()}`);
  
  const { agents, loadAgents, executeAgent, executing, result, error } = useAgent();
  const { messages, connected } = useWebSocket(clientId);

  useEffect(() => {
    loadAgents();
  }, []);

  const handleExecute = async () => {
    try {
      const request = {
        agent_name: selectedAgent,
        task_type: taskType,
        parameters: JSON.parse(parameters)
      };
      await executeAgent(request);
    } catch (err) {
      console.error('Execution failed:', err);
    }
  };

  return (
    <div className="agent-executor">
      <h2>Execute Agent</h2>
      
      <div className="form-group">
        <label>Agent:</label>
        <select 
          value={selectedAgent} 
          onChange={(e) => setSelectedAgent(e.target.value)}
        >
          <option value="">Select Agent</option>
          {agents.map(agent => (
            <key={agent.name} value={agent.name}>
              {agent.name} - {agent.description}
            </option>
          ))}
        </select>
      </div>

      <div className="form-group">
        <label>Task Type:</label>
        <input 
          type="text" 
          value={taskType}
          onChange={(e) => setTaskType(e.target.value)}
          placeholder="e.g., code_generation"
        />
      </div>

      <div className="form-group">
        <label>Parameters (JSON):</label>
        <textarea 
          value={parameters}
          onChange={(e) => setParameters(e.target.value)}
          rows={6}
          placeholder='{"language": "python", "prompt": "..."}'
        />
      </div>

      <button onClick={handleExecute} disabled={executing}>
        {executing ? 'Executing...' : 'Execute Agent'}
      </button>

      {error && <div className="error">{error}</div>}
      
      {result && (
        <div className="result">
          <h3>Result:</h3>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}

      {connected && messages.length > 0 && (
        <div className="realtime-updates">
          <h3>Real-time Updates:</h3>
          {messages.map((msg, idx) => (
            <div key={idx} className="message">
              {msg.type}: {JSON.stringify(msg.data)}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
```

## STEP 8: UPDATE APP.JSX AND ROUTES (20 minutes)

## STEP 9: ADD ENVIRONMENT VARIABLES (10 minutes)

## STEP 10: CREATE PACKAGE.JSON SCRIPTS (10 minutes)

## STEP 11: TEST INTEGRATION (30 minutes)

## STEP 12: CREATE COMPLETION REPORT (15 minutes)

=== CRITICAL REQUIREMENTS ===

1. **JWT AUTH** - Proper JWT token handling (in-memory, not localStorage)
2. **AXIOS INTERCEPTORS** - Handle token refresh automatically
3. **ERROR HANDLING** - Global error handling for API calls
4. **WEBSOCKET** - Real-time updates for agent execution
5. **TYPE SAFETY** - Use TypeScript or PropTypes
6. **LOADING STATES** - Proper loading indicators
7. **PROTECTED ROUTES** - Auth-required routes
8. **RESPONSIVE** - Mobile-friendly UI
9. **ACCESSIBILITY** - WCAG compliance
10. **PERFORMANCE** - Optimize bundle size

=== SUCCESS CRITERIA ===

Phase 7 is complete when:
1. ✅ Frontend connects to new API
2. ✅ Authentication working (login/logout)
3. ✅ Agent execution working
4. ✅ Real-time updates via WebSocket
5. ✅ Error handling implemented
6. ✅ Loading states added
7. ✅ All pages functional
8. ✅ Integration tested end-to-end
9. ✅ Documentation updated
10. ✅ Completion report saved

=== ESTIMATED TIME ===
Total: ~5 hours
- Setup: 15 min
- API client: 30 min
- Token manager: 20 min
- Services: 30 min
- Hooks: 40 min
- Context: 25 min
- Components: 45 min
- App/Routes: 20 min
- Environment: 10 min
- Package scripts: 10 min
- Testing: 30 min
- Report: 15 min

========================================
END OF PHASE 7 - QODER PROMPT
========================================