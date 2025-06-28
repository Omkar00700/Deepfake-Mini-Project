
from flask import request, jsonify
import logging
import time
import json
from functools import wraps
from collections import defaultdict
import threading
from backend.config import (
    ENABLE_RATE_LIMITING, RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW
)

# Configure logging
logger = logging.getLogger(__name__)

# Rate limiting storage (in-memory for demonstration)
# In production, use Redis or another distributed storage
class RateLimitStore:
    def __init__(self):
        self.request_counts = defaultdict(list)
        self.lock = threading.Lock()
    
    def add_request(self, key):
        """Add a request to the store"""
        with self.lock:
            current_time = time.time()
            # Add current timestamp to the list
            self.request_counts[key].append(current_time)
            # Clean up old timestamps
            self.request_counts[key] = [t for t in self.request_counts[key] 
                                       if current_time - t < RATE_LIMIT_WINDOW]
            return len(self.request_counts[key])
    
    def get_request_count(self, key):
        """Get the number of requests for a key within the window"""
        with self.lock:
            current_time = time.time()
            # Clean up old timestamps
            self.request_counts[key] = [t for t in self.request_counts[key] 
                                       if current_time - t < RATE_LIMIT_WINDOW]
            return len(self.request_counts[key])

# Initialize rate limit store
rate_limit_store = RateLimitStore()

def get_rate_limit_key():
    """Get the key to use for rate limiting"""
    # If authenticated, use user ID
    if hasattr(request, 'user') and request.user:
        return f"user:{request.user.get('sub', 'anonymous')}"
    
    # Otherwise use IP address
    return f"ip:{request.remote_addr}"

def rate_limit(f):
    """
    Decorator for API endpoints that enforces rate limiting
    
    Usage:
        @app.route('/api/endpoint')
        @rate_limit
        def endpoint():
            # Function body
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        # Skip rate limiting if disabled
        if not ENABLE_RATE_LIMITING:
            return f(*args, **kwargs)
        
        # Get rate limit key
        key = get_rate_limit_key()
        
        # Add request to store
        request_count = rate_limit_store.add_request(key)
        
        # Check if limit exceeded
        if request_count > RATE_LIMIT_REQUESTS:
            # Log rate limit breach with detailed information
            log_data = {
                "event": "rate_limit_exceeded",
                "key": key,
                "request_count": request_count,
                "limit": RATE_LIMIT_REQUESTS,
                "window_seconds": RATE_LIMIT_WINDOW,
                "path": request.path,
                "method": request.method,
                "ip": request.remote_addr,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Add user info if available
            if hasattr(request, 'user') and request.user:
                log_data["user_id"] = request.user.get('sub')
            
            # Add request data for forensics (but be careful with sensitive data)
            if request.method == 'GET':
                log_data["query_params"] = dict(request.args)
            elif request.content_type == 'application/json':
                try:
                    # Try to log request JSON but be careful with sensitive data
                    json_data = request.get_json(silent=True)
                    if json_data:
                        # Redact sensitive fields
                        sanitized_data = json_data.copy() if isinstance(json_data, dict) else {"data": str(json_data)}
                        for key in ['password', 'token', 'key', 'secret']:
                            if key in sanitized_data:
                                sanitized_data[key] = "[REDACTED]"
                        log_data["request_data"] = sanitized_data
                except Exception:
                    pass
            
            # Log as JSON for easier parsing
            logger.warning(f"RATE_LIMIT_BREACH: {json.dumps(log_data)}")
            
            # Return rate limit exceeded response
            return jsonify({
                "success": False,
                "message": "Rate limit exceeded",
                "retry_after": RATE_LIMIT_WINDOW
            }), 429
        
        # Add rate limit headers to response
        response = f(*args, **kwargs)
        
        # If response is a tuple, extract the actual response object
        if isinstance(response, tuple):
            actual_response = response[0]
        else:
            actual_response = response
        
        # Add rate limit headers
        actual_response.headers['X-RateLimit-Limit'] = str(RATE_LIMIT_REQUESTS)
        actual_response.headers['X-RateLimit-Remaining'] = str(max(0, RATE_LIMIT_REQUESTS - request_count))
        actual_response.headers['X-RateLimit-Reset'] = str(int(time.time() + RATE_LIMIT_WINDOW))
        
        return response
    
    return decorated
