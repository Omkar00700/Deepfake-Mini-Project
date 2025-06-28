
import jwt
import time
import logging
import uuid
from typing import Dict, Any, Optional, Callable, List, Set
from flask import request, jsonify
from functools import wraps
from backend.config import (
    JWT_SECRET,
    JWT_ALGORITHM,
    JWT_EXPIRATION_MINUTES,
    JWT_REFRESH_EXPIRATION_DAYS,
    REQUIRE_AUTH
)

# Configure logging
logger = logging.getLogger(__name__)

# Store for revoked tokens (in-memory for demonstration)
# In production, use Redis or a database
REVOKED_TOKENS: Set[str] = set()

# Store for refresh tokens (in-memory for demonstration)
# In production, use a secure database
REFRESH_TOKENS: Dict[str, Dict[str, Any]] = {}

def generate_token(user_id: str, additional_claims: Optional[Dict[str, Any]] = None, 
                   token_type: str = "access") -> str:
    """
    Generate a JWT token for a user
    
    Args:
        user_id: User identifier
        additional_claims: Additional claims to include in the token
        token_type: Type of token (access or refresh)
        
    Returns:
        JWT token string
    """
    # Set token expiration based on type
    if token_type == "refresh":
        expiration = int(time.time()) + (JWT_REFRESH_EXPIRATION_DAYS * 24 * 60 * 60)
    else:  # "access" token
        expiration = int(time.time()) + (JWT_EXPIRATION_MINUTES * 60)
    
    # Generate unique token ID
    token_id = str(uuid.uuid4())
    
    # Create payload
    payload = {
        "sub": user_id,
        "exp": expiration,
        "iat": int(time.time()),
        "jti": token_id,  # JWT ID for revocation
        "type": token_type
    }
    
    # Add additional claims if provided
    if additional_claims:
        payload.update(additional_claims)
    
    # Generate token
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    
    # Store refresh token if applicable
    if token_type == "refresh":
        REFRESH_TOKENS[token_id] = {
            "user_id": user_id,
            "expiration": expiration,
            "created_at": int(time.time())
        }
    
    return token

def verify_token(token: str) -> Dict[str, Any]:
    """
    Verify a JWT token
    
    Args:
        token: JWT token string
        
    Returns:
        Dict containing token claims if valid
    
    Raises:
        jwt.PyJWTError: If the token is invalid
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        
        # Check if token has been revoked
        if payload.get("jti") in REVOKED_TOKENS:
            raise ValueError("Token has been revoked")
        
        return payload
    except jwt.ExpiredSignatureError:
        raise ValueError("Token expired")
    except jwt.InvalidTokenError:
        raise ValueError("Invalid token")

def revoke_token(token: str) -> bool:
    """
    Revoke a JWT token
    
    Args:
        token: JWT token string
        
    Returns:
        True if token was revoked, False otherwise
    """
    try:
        # Decode without verification for speed
        payload = jwt.decode(token, options={"verify_signature": False})
        
        # Get token ID
        token_id = payload.get("jti")
        if not token_id:
            logger.warning("Token has no JTI claim, cannot revoke")
            return False
        
        # Add to revoked tokens
        REVOKED_TOKENS.add(token_id)
        
        # Remove refresh token if applicable
        if payload.get("type") == "refresh" and token_id in REFRESH_TOKENS:
            del REFRESH_TOKENS[token_id]
        
        logger.info(f"Revoked token with ID {token_id}")
        return True
    except Exception as e:
        logger.error(f"Error revoking token: {str(e)}")
        return False

def clean_expired_tokens():
    """
    Remove expired tokens from storage to prevent memory leaks
    """
    current_time = int(time.time())
    
    # Clean expired refresh tokens
    expired_tokens = []
    for token_id, token_data in REFRESH_TOKENS.items():
        if token_data["expiration"] < current_time:
            expired_tokens.append(token_id)
    
    # Remove expired tokens
    for token_id in expired_tokens:
        del REFRESH_TOKENS[token_id]
        # Also remove from revoked tokens if present
        if token_id in REVOKED_TOKENS:
            REVOKED_TOKENS.remove(token_id)
    
    if expired_tokens:
        logger.info(f"Cleaned {len(expired_tokens)} expired tokens")

def auth_required(f: Callable) -> Callable:
    """
    Decorator for API endpoints that require authentication
    
    Usage:
        @app.route('/protected-endpoint')
        @auth_required
        def protected_endpoint():
            # Function body
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        # Skip auth check if auth is not required globally
        if not REQUIRE_AUTH:
            return f(*args, **kwargs)
        
        # Get token from header
        auth_header = request.headers.get('Authorization')
        
        # Check if header exists
        if not auth_header:
            logger.warning("Missing Authorization header")
            return jsonify({"success": False, "message": "Authentication required"}), 401
        
        # Check header format
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != 'bearer':
            logger.warning("Invalid Authorization header format")
            return jsonify({"success": False, "message": "Invalid authorization format"}), 401
        
        token = parts[1]
        
        try:
            # Verify token
            payload = verify_token(token)
            
            # Check token type
            if payload.get("type") != "access":
                logger.warning(f"Invalid token type: {payload.get('type')}")
                return jsonify({"success": False, "message": "Invalid token type"}), 401
            
            # Attach user info to request
            request.user = payload
            
            # Continue to the protected function
            return f(*args, **kwargs)
            
        except Exception as e:
            logger.warning(f"Authentication failed: {str(e)}")
            return jsonify({"success": False, "message": f"Authentication failed: {str(e)}"}), 401
            
    return decorated

def role_required(roles: List[str]):
    """
    Decorator for API endpoints that require specific roles
    
    Usage:
        @app.route('/admin-endpoint')
        @role_required(['admin'])
        def admin_endpoint():
            # Function body
    """
    def decorator(f):
        @wraps(f)
        @auth_required  # Apply auth_required first
        def decorated(*args, **kwargs):
            # Check if user has required role
            if not hasattr(request, 'user') or not request.user:
                return jsonify({"success": False, "message": "Authentication required"}), 401
            
            # Get user roles
            user_roles = request.user.get("roles", [])
            if isinstance(user_roles, str):
                user_roles = [user_roles]
            
            # Check if user has any of the required roles
            if not any(role in user_roles for role in roles):
                logger.warning(f"User {request.user.get('sub')} does not have required roles: {roles}")
                return jsonify({"success": False, "message": "Insufficient permissions"}), 403
            
            # Continue to the protected function
            return f(*args, **kwargs)
        return decorated
    return decorator

def auth_optional(f: Callable) -> Callable:
    """
    Decorator for API endpoints where authentication is optional
    User info will be available if authenticated, but endpoint still works without auth
    
    Usage:
        @app.route('/optional-auth-endpoint')
        @auth_optional
        def optional_auth_endpoint():
            if hasattr(request, 'user'):
                # User is authenticated
            else:
                # Anonymous access
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        # Get token from header
        auth_header = request.headers.get('Authorization')
        
        # If header exists, try to verify
        if auth_header:
            parts = auth_header.split()
            if len(parts) == 2 and parts[0].lower() == 'bearer':
                token = parts[1]
                
                try:
                    # Verify token
                    payload = verify_token(token)
                    
                    # Check token type
                    if payload.get("type") == "access":
                        # Attach user info to request
                        request.user = payload
                    else:
                        logger.debug(f"Invalid token type for optional auth: {payload.get('type')}")
                    
                except Exception as e:
                    logger.debug(f"Optional authentication failed: {str(e)}")
                    # Continue without authentication
        
        # Continue to the function
        return f(*args, **kwargs)
            
    return decorated
