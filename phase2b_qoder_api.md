========================================
PHASE 2B - QODER: API GATEWAY & BACKEND SERVICES
========================================

=== YOUR IDENTITY ===
Your name: QODER
Your role: Backend API and infrastructure architect
Your phase: 2B (NEW - CRITICAL MISSING COMPONENT)
Your workspace: C:\Users\Mohamed Mansour\Desktop\YmeraRefactor\

=== CONTEXT ===
✅ Phase 2: core_services\ created
⚠️ MISSING: API Gateway, REST endpoints, middleware, auth, message queue

**CRITICAL**: The system needs a complete API layer to connect frontend to backend!

=== YOUR MISSION ===
Create the complete API infrastructure including:
1. **API Gateway** (FastAPI/Flask)
2. **REST API Endpoints** (CRUD operations)
3. **Authentication & Authorization** (JWT)
4. **Middleware** (CORS, rate limiting, logging, error handling)
5. **Message Queue/Broker** (Redis/Celery for async tasks)
6. **WebSocket Support** (real-time updates)
7. **API Documentation** (OpenAPI/Swagger)

=== SOURCE DIRECTORY (READ-ONLY) ===
Location: C:\Users\Mohamed Mansour\Desktop\QoderAgentFiles\

Analyze if API components exist in:
```
QoderAgentFiles\
├── api\
├── routes\
├── middleware\
├── auth\
└── ... (any existing API code)
```

=== TARGET DIRECTORY (WRITE) ===
Location: C:\Users\Mohamed Mansour\Desktop\YmeraRefactor\

You will create:
```
YmeraRefactor\
├── api\  (NEW - YOU CREATE THIS)
│   ├── __init__.py
│   ├── main.py (FastAPI application)
│   ├── config.py (API configuration)
│   ├── dependencies.py (dependency injection)
│   │
│   ├── routes\
│   │   ├── __init__.py
│   │   ├── agents.py (agent execution endpoints)
│   │   ├── health.py (health check endpoints)
│   │   ├── auth.py (authentication endpoints)
│   │   └── websocket.py (WebSocket endpoints)
│   │
│   ├── middleware\
│   │   ├── __init__.py
│   │   ├── cors.py (CORS configuration)
│   │   ├── rate_limiter.py (rate limiting)
│   │   ├── auth_middleware.py (JWT validation)
│   │   ├── error_handler.py (global error handling)
│   │   └── logging_middleware.py (request/response logging)
│   │
│   ├── auth\
│   │   ├── __init__.py
│   │   ├── jwt_handler.py (JWT creation/validation)
│   │   ├── password_handler.py (password hashing)
│   │   └── permissions.py (role-based access control)
│   │
│   ├── schemas\
│   │   ├── __init__.py
│   │   ├── agent_schema.py (Pydantic models for agents)
│   │   ├── auth_schema.py (Pydantic models for auth)
│   │   └── response_schema.py (API response models)
│   │
│   └── utils\
│       ├── __init__.py
│       ├── response_builder.py (standardized responses)
│       └── validators.py (input validation)
│
├── workers\  (NEW - MESSAGE QUEUE WORKERS)
│   ├── __init__.py
│   ├── celery_app.py (Celery configuration)
│   ├── tasks.py (async task definitions)
│   └── worker_config.py
│
└── websockets\  (NEW - REAL-TIME COMMUNICATION)
    ├── __init__.py
    ├── connection_manager.py
    └── handlers.py
```

=== STEP-BY-STEP INSTRUCTIONS ===

## STEP 1: ANALYZE EXISTING API CODE (15 minutes)

Scan SOURCE_DIR for any existing API components:

```bash
qoder analyze --directory "C:\Users\Mohamed Mansour\Desktop\QoderAgentFiles" --filter "api,routes,middleware,auth"
```

Create discovery report:
**File: _reports/qoder/phase2b_api_discovery.json**
```json
{
  "existing_api": {
    "found": true/false,
    "framework": "FastAPI/Flask/Django/None",
    "endpoints": ["list of existing endpoints"],
    "auth_method": "JWT/OAuth/None",
    "middleware": ["list of middleware"]
  },
  "missing_components": ["list what needs to be created"]
}
```

## STEP 2: CREATE FASTAPI APPLICATION (30 minutes)

### 2.1 Main Application

**File: api/main.py**
```python
# YMERA Refactoring Project
# Phase: 2B | Agent: qoder | Created: 2024-11-30
# FastAPI main application

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging

from api.routes import agents, health, auth, websocket
from api.middleware.rate_limiter import RateLimitMiddleware
from api.middleware.logging_middleware import LoggingMiddleware
from api.middleware.error_handler import ErrorHandlerMiddleware
from api.config import settings
from shared.config.environment import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for startup and shutdown"""
    # Startup
    logger.info("Starting YMERA API Gateway...")
    logger.info(f"API Version: {settings.VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    
    # Initialize connections
    # TODO: Initialize database, Redis, etc.
    
    yield
    
    # Shutdown
    logger.info("Shutting down YMERA API Gateway...")
    # TODO: Close connections

# Create FastAPI application
app = FastAPI(
    title="YMERA API",
    description="Multi-Agent System API Gateway",
    version=settings.VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

# CORS Middleware (MUST be first)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom Middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(ErrorHandlerMiddleware)

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(agents.router, prefix="/api/v1/agents", tags=["Agents"])
app.include_router(websocket.router, prefix="/ws", tags=["WebSocket"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "YMERA API Gateway",
        "version": settings.VERSION,
        "docs": "/api/docs",
        "status": "operational"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
```

### 2.2 API Configuration

**File: api/config.py**
```python
# YMERA Refactoring Project
# Phase: 2B | Agent: qoder | Created: 2024-11-30
# API configuration

from pydantic_settings import BaseSettings
from typing import List
from shared.config.environment import get_config

class Settings(BaseSettings):
    """API Configuration"""
    
    # Application
    APP_NAME: str = "YMERA API"
    VERSION: str = "2.0.0"
    ENVIRONMENT: str = get_config("ENVIRONMENT", default="development")
    DEBUG: bool = get_config("DEBUG", default=True, cast=bool)
    
    # Server
    HOST: str = get_config("API_HOST", default="0.0.0.0")
    PORT: int = get_config("API_PORT", default=8000, cast=int)
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:3000",
        get_config("FRONTEND_URL", default="http://localhost:3000")
    ]
    
    # Authentication
    JWT_SECRET_KEY: str = get_config("JWT_SECRET_KEY", default="your-secret-key-change-in-production")
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = get_config("JWT_EXPIRE_MINUTES", default=30, cast=int)
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = get_config("RATE_LIMIT_ENABLED", default=True, cast=bool)
    RATE_LIMIT_REQUESTS: int = get_config("RATE_LIMIT_REQUESTS", default=100, cast=int)
    RATE_LIMIT_WINDOW: int = 60  # seconds
    
    # Redis (for rate limiting and caching)
    REDIS_HOST: str = get_config("REDIS_HOST", default="localhost")
    REDIS_PORT: int = get_config("REDIS_PORT", default=6379, cast=int)
    REDIS_DB: int = 0
    
    # Celery (message queue)
    CELERY_BROKER_URL: str = get_config(
        "CELERY_BROKER_URL",
        default=f"redis://{REDIS_HOST}:{REDIS_PORT}/1"
    )
    CELERY_RESULT_BACKEND: str = get_config(
        "CELERY_RESULT_BACKEND",
        default=f"redis://{REDIS_HOST}:{REDIS_PORT}/2"
    )
    
    # Agent Settings
    AGENT_EXECUTION_TIMEOUT: int = 300  # seconds
    MAX_CONCURRENT_AGENTS: int = 10
    
    class Config:
        case_sensitive = True

settings = Settings()
```

## STEP 3: CREATE API ROUTES (45 minutes)

### 3.1 Agent Execution Routes

**File: api/routes/agents.py**
```python
# YMERA Refactoring Project
# Phase: 2B | Agent: qoder | Created: 2024-11-30
# Agent execution API routes

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status
from typing import Dict, Any, List
import logging

from api.schemas.agent_schema import (
    AgentExecutionRequest,
    AgentExecutionResponse,
    AgentListResponse,
    AgentStatusResponse
)
from api.auth.jwt_handler import get_current_user
from agents.registry import AgentRegistry
from agents.base.base_agent import AgentRequest
from workers.tasks import execute_agent_async

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/", response_model=AgentListResponse)
async def list_agents():
    """
    List all available agents
    
    Returns list of agent names and their capabilities
    """
    try:
        agent_names = AgentRegistry.list_agents()
        agents_info = []
        
        for name in agent_names:
            agent_class = AgentRegistry.get_agent_class(name)
            # Create temporary instance to get info
            temp_agent = agent_class()
            info = temp_agent.get_info()
            agents_info.append(info)
        
        return AgentListResponse(
            success=True,
            agents=agents_info,
            total=len(agents_info)
        )
    except Exception as e:
        logger.error(f"Error listing agents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list agents: {str(e)}"
        )

@router.post("/execute", response_model=AgentExecutionResponse)
async def execute_agent(
    request: AgentExecutionRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Execute an agent synchronously
    
    Parameters:
    - agent_name: Name of the agent to execute
    - task_type: Type of task
    - parameters: Task parameters
    
    Returns execution result
    """
    try:
        # Create agent
        agent = AgentRegistry.create_agent(request.agent_name)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent '{request.agent_name}' not found"
            )
        
        # Initialize agent
        await agent.initialize()
        
        # Create agent request
        agent_request = AgentRequest(
            task_id=request.task_id or f"task_{current_user['user_id']}_{request.agent_name}",
            task_type=request.task_type,
            parameters=request.parameters,
            context={"user_id": current_user["user_id"]}
        )
        
        # Execute
        response = await agent.execute(agent_request)
        
        # Cleanup
        await agent.shutdown()
        
        return AgentExecutionResponse(
            success=response.status == "success",
            task_id=response.task_id,
            status=response.status,
            result=response.result,
            error=response.error,
            metadata=response.metadata
        )
        
    except Exception as e:
        logger.error(f"Error executing agent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failed: {str(e)}"
        )

@router.post("/execute/async", response_model=Dict[str, str])
async def execute_agent_async_endpoint(
    request: AgentExecutionRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user)
):
    """
    Execute an agent asynchronously using Celery
    
    Returns task ID for status checking
    """
    try:
        # Queue task with Celery
        task = execute_agent_async.delay(
            agent_name=request.agent_name,
            task_type=request.task_type,
            parameters=request.parameters,
            user_id=current_user["user_id"]
        )
        
        return {
            "success": True,
            "task_id": task.id,
            "message": "Task queued successfully",
            "status_url": f"/api/v1/agents/status/{task.id}"
        }
        
    except Exception as e:
        logger.error(f"Error queuing agent task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue task: {str(e)}"
        )

@router.get("/status/{task_id}", response_model=AgentStatusResponse)
async def get_task_status(
    task_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get status of an async task
    
    Returns task status and result (if complete)
    """
    from celery.result import AsyncResult
    
    try:
        task = AsyncResult(task_id)
        
        response = {
            "task_id": task_id,
            "status": task.state,
            "result": None,
            "error": None
        }
        
        if task.ready():
            if task.successful():
                response["result"] = task.result
            else:
                response["error"] = str(task.info)
        
        return AgentStatusResponse(**response)
        
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task status: {str(e)}"
        )
```

### 3.2 Authentication Routes

**File: api/routes/auth.py**
```python
# YMERA Refactoring Project
# Phase: 2B | Agent: qoder | Created: 2024-11-30
# Authentication API routes

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from typing import Dict

from api.schemas.auth_schema import (
    TokenResponse,
    UserRegisterRequest,
    UserResponse
)
from api.auth.jwt_handler import create_access_token, create_refresh_token
from api.auth.password_handler import verify_password, hash_password
from shared.database.db_manager import DatabaseManager

router = APIRouter()

@router.post("/register", response_model=UserResponse)
async def register(request: UserRegisterRequest):
    """
    Register a new user
    
    Parameters:
    - username: Unique username
    - email: User email
    - password: User password
    
    Returns user information
    """
    try:
        db = DatabaseManager.get_instance()
        await db.connect()
        
        # Check if user exists
        existing = await db.execute_query(
            "SELECT * FROM users WHERE username = ? OR email = ?",
            (request.username, request.email)
        )
        
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username or email already registered"
            )
        
        # Hash password
        hashed_password = hash_password(request.password)
        
        # Insert user
        user_id = await db.execute_insert(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (request.username, request.email, hashed_password)
        )
        
        return UserResponse(
            user_id=user_id,
            username=request.username,
            email=request.email
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )

@router.post("/login", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login and get access token
    
    Parameters:
    - username: Username
    - password: Password
    
    Returns JWT access and refresh tokens
    """
    try:
        db = DatabaseManager.get_instance()
        await db.connect()
        
        # Get user
        users = await db.execute_query(
            "SELECT * FROM users WHERE username = ?",
            (form_data.username,)
        )
        
        if not users:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password"
            )
        
        user = users[0]
        
        # Verify password
        if not verify_password(form_data.password, user["password_hash"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password"
            )
        
        # Create tokens
        access_token = create_access_token({"sub": user["username"], "user_id": user["id"]})
        refresh_token = create_refresh_token({"sub": user["username"], "user_id": user["id"]})
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )
```

### 3.3 Health Check Routes

**File: api/routes/health.py**
```python
# YMERA Refactoring Project
# Phase: 2B | Agent: qoder | Created: 2024-11-30
# Health check API routes

from fastapi import APIRouter, status
from typing import Dict
import time

from shared.database.db_manager import DatabaseManager
from api.config import settings

router = APIRouter()

@router.get("/health")
async def health_check() -> Dict:
    """
    Basic health check
    
    Returns API status
    """
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "timestamp": time.time()
    }

@router.get("/health/detailed")
async def detailed_health_check() -> Dict:
    """
    Detailed health check including dependencies
    
    Returns status of all services
    """
    health_status = {
        "api": "healthy",
        "database": "unknown",
        "redis": "unknown",
        "celery": "unknown"
    }
    
    # Check database
    try:
        db = DatabaseManager.get_instance()
        await db.connect()
        await db.execute_query("SELECT 1")
        health_status["database"] = "healthy"
    except Exception as e:
        health_status["database"] = f"unhealthy: {str(e)}"
    
    # Check Redis
    try:
        import redis
        r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT)
        r.ping()
        health_status["redis"] = "healthy"
    except Exception as e:
        health_status["redis"] = f"unhealthy: {str(e)}"
    
    # Check Celery
    try:
        from workers.celery_app import celery_app
        inspect = celery_app.control.inspect()
        if inspect.active():
            health_status["celery"] = "healthy"
        else:
            health_status["celery"] = "no workers"
    except Exception as e:
        health_status["celery"] = f"unhealthy: {str(e)}"
    
    overall_healthy = all(v == "healthy" for v in health_status.values())
    
    return {
        "status": "healthy" if overall_healthy else "degraded",
        "services": health_status,
        "version": settings.VERSION,
        "timestamp": time.time()
    }
```

### 3.4 WebSocket Route

**File: api/routes/websocket.py**
```python
# YMERA Refactoring Project
# Phase: 2B | Agent: qoder | Created: 2024-11-30
# WebSocket API routes for real-time communication

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from websockets.connection_manager import ConnectionManager
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

manager = ConnectionManager()

@router.websocket("/agent-updates/{client_id}")
async def agent_updates_websocket(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time agent execution updates
    
    Parameters:
    - client_id: Unique client identifier
    """
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            logger.info(f"Received from {client_id}: {data}")
            
            # Echo or process message
            await manager.send_personal_message(f"Echo: {data}", client_id)
            
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        logger.info(f"Client {client_id} disconnected")
```

## STEP 4: CREATE MIDDLEWARE (30 minutes)

[Continue with middleware implementations...]

## STEP 5: CREATE AUTH SYSTEM (30 minutes)

## STEP 6: CREATE PYDANTIC SCHEMAS (20 minutes)

## STEP 7: CREATE MESSAGE QUEUE WORKERS (30 minutes)

## STEP 8: CREATE WEBSOCKET HANDLERS (20 minutes)

## STEP 9: CREATE API DOCUMENTATION (15 minutes)

## STEP 10: INTEGRATION & TESTING (20 minutes)

## STEP 11: CREATE COMPLETION REPORT (15 minutes)

=== CRITICAL REQUIREMENTS ===

1. **FASTAPI** - Use FastAPI framework (modern, async, auto-docs)
2. **JWT AUTH** - Implement proper JWT authentication
3. **RATE LIMITING** - Protect API from abuse
4. **CORS** - Allow frontend connections
5. **WEBSOCKETS** - Enable real-time updates
6. **MESSAGE QUEUE** - Use Celery for async tasks
7. **ERROR HANDLING** - Global error handling middleware
8. **API DOCS** - Auto-generated OpenAPI/Swagger docs
9. **TYPE SAFETY** - Pydantic schemas for all requests/responses
10. **SECURITY** - Follow OWASP best practices

=== SUCCESS CRITERIA ===

Phase 2B is complete when:
1. ✅ FastAPI application running
2. ✅ All API endpoints functional
3. ✅ Authentication working
4. ✅ Middleware configured
5. ✅ Message queue operational
6. ✅ WebSocket support enabled
7. ✅ API documentation generated
8. ✅ Integration with core_services verified
9. ✅ All endpoints tested
10. ✅ Completion report saved

=== ESTIMATED TIME ===
Total: ~4 hours
- Analysis: 15 min
- FastAPI app: 30 min
- API routes: 45 min
- Middleware: 30 min
- Auth system: 30 min
- Schemas: 20 min
- Message queue: 30 min
- WebSocket: 20 min
- Documentation: 15 min
- Integration: 20 min
- Report: 15 min

========================================
END OF PHASE 2B - QODER PROMPT
========================================