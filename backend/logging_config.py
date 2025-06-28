
import os
import sys
import logging
import json
import traceback
import socket
import requests
from typing import Dict, Any, Optional
from logging.handlers import RotatingFileHandler, SMTPHandler
from backend.config import (
    LOG_LEVEL, LOG_FILE, LOG_FORMAT, ENVIRONMENT,
    VERSION, CENTRALIZED_LOGGING_ENABLED, CENTRALIZED_LOGGING_URL,
    ALERT_EMAIL, ENABLE_ERROR_REPORTING, ERROR_REPORTING_EMAIL,
    ERROR_REPORTING_WEBHOOK
)

class JsonFormatter(logging.Formatter):
    """
    Formatter for JSON-structured logs
    """
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "environment": ENVIRONMENT,
            "version": VERSION,
            "host": socket.gethostname()
        }
        
        # Add exception info if available
        if record.exc_info:
            log_record["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exc()
            }
            
        # Add extra attributes
        if hasattr(record, 'request_id'):
            log_record["request_id"] = record.request_id
            
        if hasattr(record, 'user_id'):
            log_record["user_id"] = record.user_id
            
        # Add other custom fields
        for key, value in record.__dict__.items():
            if key not in [
                'args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
                'funcName', 'id', 'levelname', 'levelno', 'lineno', 'module',
                'msecs', 'message', 'msg', 'name', 'pathname', 'process',
                'processName', 'relativeCreated', 'stack_info', 'thread', 'threadName'
            ]:
                log_record[key] = value
                
        return json.dumps(log_record)

class CentralizedLogHandler(logging.Handler):
    """
    Custom handler for sending logs to a centralized logging service
    """
    def __init__(self, url):
        super().__init__()
        self.url = url
        self.buffer = []
        self.buffer_size = 10  # Send logs in batches
        
    def emit(self, record):
        try:
            # Format the record
            log_entry = self.format(record)
            
            # Add to buffer
            self.buffer.append(log_entry)
            
            # Send if buffer is full or record is an error/critical
            if len(self.buffer) >= self.buffer_size or record.levelno >= logging.ERROR:
                self.flush()
                
        except Exception:
            self.handleError(record)
            
    def flush(self):
        if not self.buffer:
            return
            
        try:
            # Send logs to centralized service
            response = requests.post(
                self.url,
                json={"logs": self.buffer},
                headers={"Content-Type": "application/json"},
                timeout=2.0  # Short timeout to avoid blocking
            )
            
            if response.status_code != 200:
                sys.stderr.write(f"Failed to send logs to centralized service: {response.status_code}\n")
                
            # Clear buffer
            self.buffer = []
            
        except Exception as e:
            sys.stderr.write(f"Error sending logs to centralized service: {str(e)}\n")
            
    def close(self):
        self.flush()
        super().close()

class ErrorReportingHandler(logging.Handler):
    """
    Custom handler for reporting critical errors
    """
    def __init__(self, email=None, webhook=None):
        super().__init__()
        self.email = email
        self.webhook = webhook
        self.level = logging.ERROR  # Only handle ERROR and CRITICAL
        
    def emit(self, record):
        if record.levelno < self.level:
            return
            
        try:
            # Format the record
            log_entry = self.format(record)
            
            # Build error report
            error_report = {
                "level": record.levelname,
                "message": record.getMessage(),
                "timestamp": self.formatter.formatTime(record, self.formatter.datefmt),
                "logger": record.name,
                "environment": ENVIRONMENT,
                "version": VERSION,
                "host": socket.gethostname()
            }
            
            # Add exception info if available
            if record.exc_info:
                error_report["exception"] = {
                    "type": record.exc_info[0].__name__,
                    "message": str(record.exc_info[1]),
                    "traceback": traceback.format_exc()
                }
            
            # Send to webhook if configured
            if self.webhook:
                self._send_to_webhook(error_report)
                
            # Send to email if configured
            if self.email:
                self._send_to_email(error_report)
                
        except Exception:
            self.handleError(record)
            
    def _send_to_webhook(self, error_report: Dict[str, Any]):
        """Send error report to webhook (e.g., Slack, Discord)"""
        try:
            # Format for webhook
            webhook_payload = {
                "text": f"*ALERT: {error_report['level']} in {error_report['environment']}*",
                "attachments": [
                    {
                        "color": "#ff0000" if error_report['level'] == "CRITICAL" else "#ffa500",
                        "title": error_report['message'],
                        "fields": [
                            {"title": "Environment", "value": error_report['environment'], "short": True},
                            {"title": "Version", "value": error_report['version'], "short": True},
                            {"title": "Host", "value": error_report['host'], "short": True},
                            {"title": "Logger", "value": error_report['logger'], "short": True}
                        ]
                    }
                ]
            }
            
            # Add exception info if available
            if "exception" in error_report:
                exc = error_report["exception"]
                webhook_payload["attachments"][0]["fields"].append(
                    {"title": "Exception", "value": f"{exc['type']}: {exc['message']}", "short": False}
                )
                webhook_payload["attachments"][0]["fields"].append(
                    {"title": "Traceback", "value": f"```{exc['traceback']}```", "short": False}
                )
            
            # Send to webhook
            requests.post(
                self.webhook,
                json=webhook_payload,
                headers={"Content-Type": "application/json"},
                timeout=5.0
            )
            
        except Exception as e:
            sys.stderr.write(f"Failed to send error report to webhook: {str(e)}\n")
            
    def _send_to_email(self, error_report: Dict[str, Any]):
        """Send error report via email (Note: uses logging's built-in SMTPHandler)"""
        # This is a placeholder - in a real implementation, you'd integrate with an email service
        sys.stderr.write(f"Would send error report to {self.email}\n")

def setup_logging():
    """
    Configure application logging
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    
    # File handler
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    
    # Select formatter based on configuration
    if LOG_FORMAT.lower() == 'json':
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Set formatter for handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Add centralized logging if enabled
    if CENTRALIZED_LOGGING_ENABLED and CENTRALIZED_LOGGING_URL:
        centralized_handler = CentralizedLogHandler(CENTRALIZED_LOGGING_URL)
        centralized_handler.setFormatter(formatter)
        root_logger.addHandler(centralized_handler)
    
    # Add error reporting handler if enabled
    if ENABLE_ERROR_REPORTING:
        error_handler = ErrorReportingHandler(
            email=ERROR_REPORTING_EMAIL,
            webhook=ERROR_REPORTING_WEBHOOK
        )
        error_handler.setFormatter(formatter)
        root_logger.addHandler(error_handler)
    
    # Disable certain noisy loggers
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    # Log startup message
    root_logger.info(f"Logging initialized: level={LOG_LEVEL}, format={LOG_FORMAT}, environment={ENVIRONMENT}")
    
    return root_logger

# Initialize logging
logger = setup_logging()
