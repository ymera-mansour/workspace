"""
YMERA Multi-Agent Workspace Platform
FastAPI Application Entry Point
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import os
import logging

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="YMERA Multi-Agent Workspace Platform",
    description="Intelligent AI orchestration with multi-model execution",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class CompletionRequest(BaseModel):
    """Request model for completions"""
    prompt: str = Field(..., min_length=1, max_length=10000, description="The prompt to process")
    user_id: str = Field(default="anonymous", description="User identifier")
    agent: Optional[str] = Field(default=None, description="Specific agent to use")
    max_tokens: Optional[int] = Field(default=1024, ge=1, le=8192, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=0.7, ge=0, le=2, description="Sampling temperature")
    
class CompletionResponse(BaseModel):
    """Response model for completions"""
    status: str
    result: Any
    model_used: Optional[str] = None
    execution_time: Optional[float] = None
    cost: Optional[float] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    services: Dict[str, str]

# Routes
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "name": "YMERA Multi-Agent Workspace Platform",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    
    services = {
        "api": "healthy",
        "redis": "unknown",
        "ai_providers": "unknown"
    }
    
    # Check Redis
    try:
        import redis
        r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
        r.ping()
        services["redis"] = "connected"
    except Exception as e:
        logger.warning(f"Redis check failed: {e}")
        services["redis"] = "disconnected"
    
    # Check AI providers
    providers = []
    if os.getenv("GROQ_API_KEY"):
        providers.append("groq")
    if os.getenv("GEMINI_API_KEY"):
        providers.append("gemini")
    if os.getenv("OPENROUTER_API_KEY"):
        providers.append("openrouter")
    if os.getenv("MISTRAL_API_KEY"):
        providers.append("mistral")
    
    services["ai_providers"] = ", ".join(providers) if providers else "none_configured"
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        services=services
    )

@app.post("/v1/completions", response_model=CompletionResponse, tags=["Completions"])
async def create_completion(request: CompletionRequest):
    """
    Create a completion using AI models
    
    This endpoint orchestrates multiple AI models to complete the given prompt.
    """
    
    try:
        # For now, return a simple response
        # In production, this would call the multi_model_executor
        logger.info(f"Completion request from user {request.user_id}: {request.prompt[:50]}...")
        
        # Placeholder response
        result = {
            "content": "This is a placeholder response. The full multi-model executor will process this request.",
            "note": "Configure AI provider keys in .env to enable full functionality"
        }
        
        return CompletionResponse(
            status="success",
            result=result,
            model_used="placeholder",
            execution_time=0.1,
            cost=0.0
        )
        
    except Exception as e:
        logger.error(f"Error processing completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models", tags=["Models"])
async def list_models():
    """List available AI models"""
    
    models = {
        "groq": [
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile",
            "llama-3.3-70b-versatile"
        ],
        "gemini": [
            "gemini-1.5-flash",
            "gemini-1.5-pro"
        ],
        "openrouter": [
            "meta-llama/llama-3.1-8b-instruct:free",
            "mistralai/mistral-7b-instruct:free",
            "google/gemma-2-9b-it:free"
        ]
    }
    
    return {
        "models": models,
        "total": sum(len(v) for v in models.values()),
        "note": "Configure API keys in .env to enable models"
    }

@app.get("/v1/agents", tags=["Agents"])
async def list_agents():
    """List available agents"""
    
    agents = {
        "coding_agent": {
            "description": "Generates and refactors code",
            "capabilities": ["code_generation", "code_review", "refactoring"]
        },
        "analysis_agent": {
            "description": "Analyzes data and provides insights",
            "capabilities": ["data_analysis", "visualization", "reporting"]
        },
        "research_agent": {
            "description": "Researches topics and gathers information",
            "capabilities": ["web_search", "summarization", "synthesis"]
        },
        "documentation_agent": {
            "description": "Creates documentation",
            "capabilities": ["api_docs", "user_guides", "technical_writing"]
        }
    }
    
    return {
        "agents": agents,
        "total": len(agents)
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if os.getenv("DEBUG") == "true" else "An error occurred"
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("YMERA Platform starting...")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    logger.info(f"Port: {os.getenv('PORT', '8000')}")
    
    # Check critical configuration
    if not os.getenv("GROQ_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        logger.warning("No AI provider keys configured. Add GROQ_API_KEY or GEMINI_API_KEY to .env")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("YMERA Platform shutting down...")

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("AUTO_RELOAD", "false").lower() == "true"
    )
