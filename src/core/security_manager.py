# YMERA Refactoring Project
# Phase: 2E Enhanced | Agent: qoder | Created: 2024-12-05
# Security Manager for AI System

from typing import Dict, Any, List, Optional
import re
import hashlib
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from cryptography.fernet import Fernet
import base64

logger = logging.getLogger(__name__)

@dataclass
class SecurityEvent:
    """Security event for audit logging"""
    event_id: str
    event_type: str
    severity: str  # info, warning, critical
    timestamp: str
    user_id: Optional[str]
    details: Dict[str, Any]
    action_taken: str

class SecurityManager:
    """
    Comprehensive security manager for AI system
    
    Features:
    - Input sanitization
    - Output validation
    - API key encryption
    - Rate limiting
    - Audit logging
    - Injection detection
    - Content filtering
    """
    
    def __init__(self):
        self.audit_log: List[SecurityEvent] = []
        self.rate_limits: Dict[str, List[float]] = {}
        self.blocked_patterns = self._initialize_blocked_patterns()
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        logger.info("Security Manager initialized")
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for API keys"""
        import os
        
        key_file = "_data/security/encryption.key"
        
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            os.makedirs(os.path.dirname(key_file), exist_ok=True)
            with open(key_file, 'wb') as f:
                f.write(key)
            return key
    
    def _initialize_blocked_patterns(self) -> List[re.Pattern]:
        """Initialize patterns for malicious input detection"""
        return [
            # Code injection patterns
            re.compile(r'(?i)(exec|eval|__import__|compile)\s*\(', re.IGNORECASE),
            re.compile(r'(?i)(subprocess|os\.system|os\.popen)', re.IGNORECASE),
            
            # SQL injection patterns
            re.compile(r"(?i)(union\s+select|drop\s+table|insert\s+into|delete\s+from)", re.IGNORECASE),
            re.compile(r"(?i)(--|;|\/\*|\*\/|xp_)", re.IGNORECASE),
            
            # Command injection patterns
            re.compile(r'[;&|`$]', re.IGNORECASE),
            re.compile(r'(?i)(rm\s+-rf|mkfs|dd\s+if)', re.IGNORECASE),
            
            # Path traversal patterns
            re.compile(r'\.\.\/|\.\.\\', re.IGNORECASE),
            
            # Script injection patterns
            re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
            re.compile(r'javascript:', re.IGNORECASE),
            
            # Credential patterns (to block)
            re.compile(r'(?i)(password|api[_-]?key|secret|token)\s*[:=]\s*["\']?[\w\-]+', re.IGNORECASE),
        ]
    
    def sanitize_input(self, user_input: str, context: str = "general") -> Dict[str, Any]:
        """
        Sanitize user input
        
        Returns:
            Dict with 'safe': bool, 'sanitized': str, 'issues': List[str]
        """
        issues = []
        
        # Check for empty input
        if not user_input or not user_input.strip():
            return {
                "safe": False,
                "sanitized": "",
                "issues": ["Empty input"]
            }
        
        # Check length
        if len(user_input) > 100000:  # 100KB limit
            issues.append("Input too long (>100KB)")
            user_input = user_input[:100000]
        
        # Check for malicious patterns
        for pattern in self.blocked_patterns:
            if pattern.search(user_input):
                match = pattern.search(user_input)
                issues.append(f"Potentially malicious pattern detected: {match.group()[:50]}")
                
                # Log security event
                self._log_security_event(
                    event_type="injection_attempt",
                    severity="warning",
                    details={
                        "pattern": pattern.pattern[:100],
                        "matched": match.group()[:100],
                        "context": context
                    },
                    action_taken="Input flagged"
                )
        
        # Check for excessive special characters (possible obfuscation)
        special_char_ratio = sum(1 for c in user_input if not c.isalnum() and not c.isspace()) / len(user_input)
        if special_char_ratio > 0.3:
            issues.append(f"High special character ratio: {special_char_ratio:.2%}")
        
        # Basic sanitization
        sanitized = user_input.strip()
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        # Determine if safe
        safe = len(issues) == 0 or all("too long" in issue.lower() for issue in issues)
        
        return {
            "safe": safe,
            "sanitized": sanitized,
            "issues": issues,
            "severity": "high" if not safe else "low"
        }
    
    def validate_output(self, ai_output: str, expected_type: str = "text") -> Dict[str, Any]:
        """
        Validate AI output
        
        Checks for:
        - Sensitive data leaks
        - Harmful content
        - Format correctness
        """
        issues = []
        
        # Check for potential API keys or secrets in output
        api_key_pattern = re.compile(r'(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*["\']?[\w\-]{20,}', re.IGNORECASE)
        if api_key_pattern.search(ai_output):
            issues.append("Potential API key or secret in output")
            
            # Log security event
            self._log_security_event(
                event_type="secret_leak",
                severity="critical",
                details={"output_length": len(ai_output)},
                action_taken="Output flagged for review"
            )
        
        # Check for excessive code execution in output
        if expected_type == "text":
            code_blocks = re.findall(r'```[\s\S]*?```', ai_output)
            if len(code_blocks) > 10:
                issues.append(f"Excessive code blocks: {len(code_blocks)}")
        
        # Check for harmful instructions
        harmful_patterns = [
            r'(?i)how to (hack|crack|break into|exploit)',
            r'(?i)(bomb|weapon|explosive) (making|creation|build)',
            r'(?i)(steal|theft|fraud) (guide|tutorial|method)'
        ]
        
        for pattern in harmful_patterns:
            if re.search(pattern, ai_output):
                issues.append(f"Potentially harmful content detected")
                
                self._log_security_event(
                    event_type="harmful_content",
                    severity="critical",
                    details={"pattern": pattern[:100]},
                    action_taken="Output flagged"
                )
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "severity": "critical" if issues else "low"
        }
    
    def encrypt_api_key(self, api_key: str) -> str:
        """Encrypt API key"""
        if not api_key:
            return ""
        
        encrypted = self.cipher_suite.encrypt(api_key.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt_api_key(self, encrypted_key: str) -> str:
        """Decrypt API key"""
        if not encrypted_key:
            return ""
        
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_key.encode())
            decrypted = self.cipher_suite.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Failed to decrypt API key: {e}")
            return ""
    
    def check_rate_limit(
        self,
        user_id: str,
        limit: int = 100,
        window: int = 3600  # 1 hour
    ) -> Dict[str, Any]:
        """
        Check if user has exceeded rate limit
        
        Args:
            user_id: User identifier
            limit: Maximum requests per window
            window: Time window in seconds
        
        Returns:
            Dict with 'allowed': bool, 'remaining': int, 'reset_time': float
        """
        current_time = datetime.now().timestamp()
        
        # Initialize if not exists
        if user_id not in self.rate_limits:
            self.rate_limits[user_id] = []
        
        # Remove old requests outside window
        self.rate_limits[user_id] = [
            req_time for req_time in self.rate_limits[user_id]
            if current_time - req_time < window
        ]
        
        # Check limit
        request_count = len(self.rate_limits[user_id])
        allowed = request_count < limit
        
        if allowed:
            # Add current request
            self.rate_limits[user_id].append(current_time)
            remaining = limit - request_count - 1
        else:
            remaining = 0
            
            # Log rate limit exceeded
            self._log_security_event(
                event_type="rate_limit_exceeded",
                severity="warning",
                details={
                    "user_id": user_id,
                    "limit": limit,
                    "window": window,
                    "request_count": request_count
                },
                action_taken="Request blocked",
                user_id=user_id
            )
        
        # Calculate reset time
        if self.rate_limits[user_id]:
            oldest_request = min(self.rate_limits[user_id])
            reset_time = oldest_request + window
        else:
            reset_time = current_time + window
        
        return {
            "allowed": allowed,
            "remaining": remaining,
            "reset_time": reset_time,
            "reset_in_seconds": int(reset_time - current_time)
        }
    
    def _log_security_event(
        self,
        event_type: str,
        severity: str,
        details: Dict[str, Any],
        action_taken: str,
        user_id: Optional[str] = None
    ):
        """Log security event"""
        import uuid
        
        event = SecurityEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            severity=severity,
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            details=details,
            action_taken=action_taken
        )
        
        self.audit_log.append(event)
        
        # Log to file
        self._write_audit_log(event)
        
        # Alert if critical
        if severity == "critical":
            logger.critical(f"Security event: {event_type} - {action_taken}")
    
    def _write_audit_log(self, event: SecurityEvent):
        """Write event to audit log file"""
        import os
        
        log_dir = "_data/security/audit_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Organize by date
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(log_dir, f"audit_{date_str}.jsonl")
        
        with open(log_file, 'a') as f:
            event_dict = {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "severity": event.severity,
                "timestamp": event.timestamp,
                "user_id": event.user_id,
                "details": event.details,
                "action_taken": event.action_taken
            }
            f.write(json.dumps(event_dict) + '\n')
    
    def get_security_report(self, hours: int = 24) -> Dict[str, Any]:
        """Get security report for last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_events = [
            event for event in self.audit_log
            if datetime.fromisoformat(event.timestamp) > cutoff_time
        ]
        
        # Count by type
        events_by_type = {}
        for event in recent_events:
            events_by_type[event.event_type] = events_by_type.get(event.event_type, 0) + 1
        
        # Count by severity
        events_by_severity = {}
        for event in recent_events:
            events_by_severity[event.severity] = events_by_severity.get(event.severity, 0) + 1
        
        return {
            "period_hours": hours,
            "total_events": len(recent_events),
            "events_by_type": events_by_type,
            "events_by_severity": events_by_severity,
            "critical_events": [
                {
                    "event_id": event.event_id,
                    "type": event.event_type,
                    "timestamp": event.timestamp,
                    "details": event.details
                }
                for event in recent_events
                if event.severity == "critical"
            ]
        }
    
    def validate_tool_execution(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate if tool execution is safe
        
        Returns:
            Dict with 'allowed': bool, 'reason': str
        """
        # Dangerous tools that require extra validation
        dangerous_tools = [
            "execute_python",
            "execute_javascript",
            "execute_command",
            "write_file",
            "delete_file",
            "database_query",
            "database_update"
        ]
        
        if tool_name in dangerous_tools:
            # Check parameters for malicious content
            params_str = json.dumps(parameters)
            sanitized = self.sanitize_input(params_str, context=f"tool:{tool_name}")
            
            if not sanitized["safe"]:
                self._log_security_event(
                    event_type="dangerous_tool_blocked",
                    severity="warning",
                    details={
                        "tool": tool_name,
                        "issues": sanitized["issues"]
                    },
                    action_taken="Tool execution blocked",
                    user_id=user_id
                )
                
                return {
                    "allowed": False,
                    "reason": f"Unsafe parameters detected: {', '.join(sanitized['issues'])}"
                }
        
        return {
            "allowed": True,
            "reason": "Tool execution approved"
        }


# Singleton
_security_manager_instance = None

def get_security_manager() -> SecurityManager:
    """Get singleton security manager instance"""
    global _security_manager_instance
    if _security_manager_instance is None:
        _security_manager_instance = SecurityManager()
    return _security_manager_instance
