# Security Best Practices Guide

## 🔒 Overview

This guide covers security best practices for deploying and operating the YMERA Multi-Agent Workspace Platform.

## Table of Contents

1. [API Key Management](#api-key-management)
2. [Authentication & Authorization](#authentication--authorization)
3. [Input Validation](#input-validation)
4. [Rate Limiting](#rate-limiting)
5. [Data Protection](#data-protection)
6. [Network Security](#network-security)
7. [Monitoring & Auditing](#monitoring--auditing)
8. [Incident Response](#incident-response)

---

## API Key Management

### Storage Best Practices

**Never hardcode API keys in code:**
```python
# ❌ BAD - Hardcoded key
api_key = "sk-1234567890abcdef"

# ✅ GOOD - From environment
api_key = os.getenv("OPENAI_API_KEY")

# ✅ BETTER - With validation
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set")
```

### Environment Variable Management

```bash
# Use .env file (never commit to git)
echo ".env" >> .gitignore

# Secure permissions
chmod 600 .env

# Encrypt for backup
gpg --symmetric --cipher-algo AES256 .env
```

### API Key Rotation

```python
class APIKeyManager:
    """Manage API key rotation"""
    
    def __init__(self):
        self.keys = self.load_keys()
        self.rotation_interval = 90  # days
    
    def load_keys(self) -> Dict[str, str]:
        """Load keys from secure storage"""
        return {
            "primary": os.getenv("API_KEY_PRIMARY"),
            "secondary": os.getenv("API_KEY_SECONDARY")
        }
    
    def get_active_key(self) -> str:
        """Get currently active key"""
        # Check if primary key needs rotation
        if self.should_rotate():
            self.rotate_keys()
        return self.keys["primary"]
    
    def rotate_keys(self):
        """Rotate to secondary key"""
        self.keys["primary"], self.keys["secondary"] = \
            self.keys["secondary"], self.keys["primary"]
        self.notify_rotation()
```

### Secrets Management with Hashicorp Vault

```python
import hvac

class VaultSecretManager:
    """Manage secrets with Hashicorp Vault"""
    
    def __init__(self, vault_url: str, token: str):
        self.client = hvac.Client(url=vault_url, token=token)
    
    def get_secret(self, path: str, key: str) -> str:
        """Retrieve secret from Vault"""
        secret = self.client.secrets.kv.v2.read_secret_version(path=path)
        return secret['data']['data'][key]
    
    def set_secret(self, path: str, key: str, value: str):
        """Store secret in Vault"""
        self.client.secrets.kv.v2.create_or_update_secret(
            path=path,
            secret={key: value}
        )
```

---

## Authentication & Authorization

### JWT Implementation

```python
from datetime import datetime, timedelta
import jwt

class AuthManager:
    """JWT-based authentication"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.algorithm = "HS256"
    
    def create_token(self, user_id: str, roles: List[str]) -> str:
        """Generate JWT token"""
        payload = {
            "user_id": user_id,
            "roles": roles,
            "exp": datetime.utcnow() + timedelta(hours=24),
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Dict:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")
```

### Role-Based Access Control (RBAC)

```python
from enum import Enum
from typing import Set

class Role(Enum):
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"

class Permission(Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"

class RBAC:
    """Role-Based Access Control"""
    
    ROLE_PERMISSIONS = {
        Role.ADMIN: {Permission.READ, Permission.WRITE, Permission.EXECUTE, Permission.ADMIN},
        Role.USER: {Permission.READ, Permission.WRITE, Permission.EXECUTE},
        Role.READONLY: {Permission.READ}
    }
    
    @classmethod
    def has_permission(cls, roles: List[Role], required: Permission) -> bool:
        """Check if user has required permission"""
        for role in roles:
            if required in cls.ROLE_PERMISSIONS.get(role, set()):
                return True
        return False
    
    @classmethod
    def require_permission(cls, required: Permission):
        """Decorator to require permission"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                user_roles = kwargs.get("user_roles", [])
                if not cls.has_permission(user_roles, required):
                    raise PermissionError(f"Permission {required} required")
                return await func(*args, **kwargs)
            return wrapper
        return decorator

# Usage
@RBAC.require_permission(Permission.EXECUTE)
async def execute_task(task: Task, user_roles: List[Role]):
    """Only users with EXECUTE permission can call this"""
    pass
```

---

## Input Validation

### Pydantic Models

```python
from pydantic import BaseModel, Field, validator
from typing import Optional

class TaskRequest(BaseModel):
    """Validated task request"""
    
    prompt: str = Field(..., min_length=1, max_length=10000)
    user_id: str = Field(..., regex=r'^[a-zA-Z0-9_-]+$')
    max_tokens: Optional[int] = Field(default=1024, ge=1, le=8192)
    temperature: Optional[float] = Field(default=0.7, ge=0, le=2)
    
    @validator('prompt')
    def validate_prompt(cls, v):
        """Validate prompt content"""
        # Check for SQL injection patterns
        sql_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER']
        if any(keyword in v.upper() for keyword in sql_keywords):
            raise ValueError("Suspicious content detected")
        
        # Check for command injection
        dangerous_chars = [';', '&&', '||', '`', '$', '(', ')']
        if any(char in v for char in dangerous_chars):
            raise ValueError("Invalid characters in prompt")
        
        return v

# Usage
try:
    task = TaskRequest(
        prompt=user_input,
        user_id=current_user
    )
except ValidationError as e:
    logger.error(f"Validation failed: {e}")
    raise
```

### SQL Injection Prevention

```python
from sqlalchemy import text

# ❌ BAD - SQL injection vulnerability
def get_user_bad(user_id: str):
    query = f"SELECT * FROM users WHERE id = '{user_id}'"
    return db.execute(query)

# ✅ GOOD - Parameterized query
def get_user_good(user_id: str):
    query = text("SELECT * FROM users WHERE id = :user_id")
    return db.execute(query, {"user_id": user_id})
```

### XSS Prevention

```python
import bleach
from markupsafe import escape

def sanitize_output(text: str) -> str:
    """Sanitize text for HTML output"""
    # Allow only safe HTML tags
    allowed_tags = ['p', 'br', 'strong', 'em', 'code']
    cleaned = bleach.clean(
        text,
        tags=allowed_tags,
        strip=True
    )
    return cleaned

def sanitize_strict(text: str) -> str:
    """Escape all HTML"""
    return escape(text)
```

---

## Rate Limiting

### Implementation with Redis

```python
from redis import Redis
import time

class RateLimiter:
    """Redis-based rate limiter"""
    
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
    
    async def check_rate_limit(
        self,
        user_id: str,
        max_requests: int = 60,
        window_seconds: int = 60
    ) -> bool:
        """Check if user is within rate limit"""
        key = f"rate_limit:{user_id}"
        current = int(time.time())
        window_start = current - window_seconds
        
        # Remove old entries
        self.redis.zremrangebyscore(key, 0, window_start)
        
        # Count requests in window
        request_count = self.redis.zcard(key)
        
        if request_count >= max_requests:
            return False
        
        # Add current request
        self.redis.zadd(key, {str(current): current})
        self.redis.expire(key, window_seconds)
        
        return True

# Usage with FastAPI
from fastapi import HTTPException

async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware"""
    user_id = request.headers.get("X-User-ID", request.client.host)
    
    if not await rate_limiter.check_rate_limit(user_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again later."
        )
    
    return await call_next(request)
```

### Token Bucket Algorithm

```python
import asyncio
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TokenBucket:
    """Token bucket rate limiter"""
    capacity: int
    refill_rate: float  # tokens per second
    tokens: float
    last_refill: datetime
    
    async def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens"""
        self.refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def refill(self):
        """Refill tokens based on time passed"""
        now = datetime.now()
        time_passed = (now - self.last_refill).total_seconds()
        
        new_tokens = time_passed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now

# Usage
bucket = TokenBucket(
    capacity=100,
    refill_rate=10,  # 10 tokens per second
    tokens=100,
    last_refill=datetime.now()
)

if await bucket.consume(1):
    # Process request
    pass
else:
    # Rate limited
    raise HTTPException(429, "Too many requests")
```

---

## Data Protection

### Encryption at Rest

```python
from cryptography.fernet import Fernet

class DataEncryption:
    """Encrypt sensitive data"""
    
    def __init__(self, key: bytes = None):
        self.key = key or Fernet.generate_key()
        self.cipher = Fernet(self.key)
    
    def encrypt(self, data: str) -> bytes:
        """Encrypt data"""
        return self.cipher.encrypt(data.encode())
    
    def decrypt(self, encrypted: bytes) -> str:
        """Decrypt data"""
        return self.cipher.decrypt(encrypted).decode()

# Usage
encryptor = DataEncryption()

# Encrypt sensitive data before storing
api_key = "sk-1234567890"
encrypted_key = encryptor.encrypt(api_key)
db.store("api_key", encrypted_key)

# Decrypt when needed
decrypted_key = encryptor.decrypt(encrypted_key)
```

### Personal Data Handling (GDPR)

```python
from datetime import datetime, timedelta

class GDPRCompliance:
    """GDPR compliance helpers"""
    
    def anonymize_user_data(self, user_id: str):
        """Anonymize user data"""
        # Replace personal info with anonymized versions
        db.update_user(user_id, {
            "name": f"user_{hash(user_id)}",
            "email": f"{hash(user_id)}@anonymized.local",
            "phone": None,
            "address": None,
            "anonymized_at": datetime.now()
        })
    
    def delete_user_data(self, user_id: str):
        """Delete all user data"""
        # Delete user records
        db.delete_user(user_id)
        db.delete_user_tasks(user_id)
        db.delete_user_logs(user_id)
        
        # Remove from caches
        cache.delete(f"user:{user_id}")
    
    def export_user_data(self, user_id: str) -> Dict:
        """Export all user data (data portability)"""
        return {
            "user": db.get_user(user_id),
            "tasks": db.get_user_tasks(user_id),
            "logs": db.get_user_logs(user_id),
            "exported_at": datetime.now().isoformat()
        }
```

---

## Network Security

### HTTPS Configuration

```nginx
# Nginx configuration for HTTPS
server {
    listen 80;
    server_name ymera.example.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name ymera.example.com;
    
    ssl_certificate /etc/ssl/certs/ymera.crt;
    ssl_certificate_key /etc/ssl/private/ymera.key;
    
    # Strong SSL settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    
    # HSTS
    add_header Strict-Transport-Security "max-age=31536000" always;
    
    # Security headers
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### CORS Configuration

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://ymera.example.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    max_age=3600
)
```

---

## Monitoring & Auditing

### Audit Logging

```python
import logging
from datetime import datetime

class AuditLogger:
    """Comprehensive audit logging"""
    
    def __init__(self):
        self.logger = logging.getLogger("audit")
    
    def log_api_call(
        self,
        user_id: str,
        endpoint: str,
        method: str,
        status_code: int,
        response_time: float,
        ip_address: str
    ):
        """Log API call"""
        self.logger.info({
            "timestamp": datetime.now().isoformat(),
            "event_type": "api_call",
            "user_id": user_id,
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "response_time": response_time,
            "ip_address": ip_address
        })
    
    def log_auth_event(
        self,
        user_id: str,
        event: str,
        success: bool,
        ip_address: str
    ):
        """Log authentication event"""
        self.logger.warning({
            "timestamp": datetime.now().isoformat(),
            "event_type": "auth",
            "user_id": user_id,
            "event": event,
            "success": success,
            "ip_address": ip_address
        })
```

### Intrusion Detection

```python
class IntrusionDetector:
    """Detect suspicious activity"""
    
    def __init__(self):
        self.failed_attempts = {}
        self.max_attempts = 5
        self.lockout_duration = 300  # 5 minutes
    
    def record_failed_attempt(self, user_id: str) -> bool:
        """Record failed login attempt"""
        now = time.time()
        
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = []
        
        # Remove old attempts
        self.failed_attempts[user_id] = [
            t for t in self.failed_attempts[user_id]
            if now - t < self.lockout_duration
        ]
        
        # Add new attempt
        self.failed_attempts[user_id].append(now)
        
        # Check if locked out
        if len(self.failed_attempts[user_id]) >= self.max_attempts:
            logger.warning(f"Account locked: {user_id}")
            return True
        
        return False
```

---

## Next Steps

1. [Deployment Guide](./local_deployment_windows.md)
2. [Monitoring Setup](../architecture/monitoring.md)
3. [Incident Response Plan](./incident_response.md)

---

**Updated**: December 2024  
**Maintainer**: YMERA Team
