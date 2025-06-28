
import os
import base64
import json
import logging
from typing import Any, Dict, Optional, Union
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from backend.config import DB_ENCRYPTION_KEY

# Configure logging
logger = logging.getLogger(__name__)

class EncryptionManager:
    """
    Handles encryption and decryption of sensitive data
    using Fernet (symmetric encryption)
    """
    
    def __init__(self):
        self.fernet = None
        self._initialize_encryption()
    
    def _initialize_encryption(self):
        """Initialize the encryption key"""
        try:
            if not DB_ENCRYPTION_KEY:
                logger.warning("No encryption key provided. Database encryption is disabled.")
                return
                
            # Generate a key from the provided encryption key
            salt = b'deepdefend_salt'  # Fixed salt for consistency
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(DB_ENCRYPTION_KEY.encode()))
            self.fernet = Fernet(key)
            logger.info("Database encryption initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {str(e)}")
            self.fernet = None
    
    def encrypt(self, data: Union[str, Dict[str, Any], None]) -> Optional[str]:
        """
        Encrypt data (string or JSON-serializable dict)
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data as a base64 string, or None if encryption failed
        """
        if data is None:
            return None
            
        if not self.fernet:
            logger.warning("Encryption not initialized. Returning unencrypted data.")
            return data if isinstance(data, str) else json.dumps(data)
            
        try:
            # Convert dict to JSON string if needed
            if not isinstance(data, str):
                data = json.dumps(data)
                
            # Encrypt the data
            encrypted_data = self.fernet.encrypt(data.encode())
            
            # Return as base64 string
            return base64.urlsafe_b64encode(encrypted_data).decode()
            
        except Exception as e:
            logger.error(f"Encryption failed: {str(e)}")
            return None
    
    def decrypt(self, encrypted_data: Optional[str]) -> Optional[Union[str, Dict[str, Any]]]:
        """
        Decrypt data
        
        Args:
            encrypted_data: Encrypted data as a base64 string
            
        Returns:
            Decrypted data (string or dict), or None if decryption failed
        """
        if encrypted_data is None:
            return None
            
        if not self.fernet:
            logger.warning("Encryption not initialized. Returning data as-is.")
            try:
                # Try to parse as JSON
                return json.loads(encrypted_data)
            except json.JSONDecodeError:
                # Return as string if not valid JSON
                return encrypted_data
        
        try:
            # Decode base64
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data)
            
            # Decrypt the data
            decrypted_data = self.fernet.decrypt(encrypted_bytes).decode()
            
            # Try to parse as JSON
            try:
                return json.loads(decrypted_data)
            except json.JSONDecodeError:
                # Return as string if not valid JSON
                return decrypted_data
                
        except InvalidToken:
            logger.error("Invalid token or incorrect key")
            return None
        except Exception as e:
            logger.error(f"Decryption failed: {str(e)}")
            return None

# Initialize encryption manager
encryption_manager = EncryptionManager()
