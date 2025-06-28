
"""
Advanced monitoring service for DeepDefend
Provides integration with external monitoring services, metrics collection,
and advanced alerting capabilities
"""

import os
import logging
import json
import time
import uuid
import traceback
from typing import Dict, Any, Optional, List, Callable, Union

# Configure logging
logger = logging.getLogger(__name__)

class SentryMonitoring:
    """
    Integration with Sentry.io for error tracking and monitoring
    """
    
    def __init__(self, dsn: Optional[str] = None, environment: str = 'production'):
        self.enabled = False
        self.dsn = dsn or os.environ.get('SENTRY_DSN')
        self.environment = environment
        
        if self.dsn:
            try:
                import sentry_sdk
                from sentry_sdk.integrations.flask import FlaskIntegration
                from sentry_sdk.integrations.celery import CeleryIntegration
                
                sentry_sdk.init(
                    dsn=self.dsn,
                    environment=self.environment,
                    traces_sample_rate=0.3,
                    integrations=[
                        FlaskIntegration(),
                        CeleryIntegration()
                    ]
                )
                self.enabled = True
                logger.info(f"Sentry monitoring initialized for environment: {environment}")
            except ImportError:
                logger.warning("Sentry SDK not installed. Run 'pip install sentry-sdk' to enable Sentry monitoring.")
            except Exception as e:
                logger.error(f"Failed to initialize Sentry: {str(e)}")
    
    def capture_exception(self, exception: Optional[Exception] = None, **kwargs):
        """Capture an exception and send it to Sentry"""
        if not self.enabled:
            return
            
        try:
            import sentry_sdk
            sentry_sdk.capture_exception(exception, **kwargs)
        except Exception as e:
            logger.error(f"Failed to capture exception in Sentry: {str(e)}")
    
    def capture_message(self, message: str, level: str = 'info', **kwargs):
        """Capture a message and send it to Sentry"""
        if not self.enabled:
            return
            
        try:
            import sentry_sdk
            sentry_sdk.capture_message(message, level=level, **kwargs)
        except Exception as e:
            logger.error(f"Failed to capture message in Sentry: {str(e)}")
    
    def start_transaction(self, name: str, op: str = None, **kwargs):
        """Start a Sentry transaction for performance monitoring"""
        if not self.enabled:
            return None
            
        try:
            import sentry_sdk
            return sentry_sdk.start_transaction(name=name, op=op, **kwargs)
        except Exception as e:
            logger.error(f"Failed to start Sentry transaction: {str(e)}")
            return None
    
    def set_user(self, user_info: Dict[str, Any]):
        """Set the current user for Sentry events"""
        if not self.enabled:
            return
            
        try:
            import sentry_sdk
            sentry_sdk.set_user(user_info)
        except Exception as e:
            logger.error(f"Failed to set Sentry user: {str(e)}")
    
    def add_breadcrumb(self, category: str, message: str, level: str = 'info', **kwargs):
        """Add a breadcrumb to the current Sentry event"""
        if not self.enabled:
            return
            
        try:
            import sentry_sdk
            sentry_sdk.add_breadcrumb(
                category=category,
                message=message,
                level=level,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Failed to add Sentry breadcrumb: {str(e)}")


class AlertingService:
    """
    Enhanced alerting service with multiple notification channels and
    configurable alert rules
    """
    
    def __init__(self):
        self.alert_channels: Dict[str, Callable] = {}
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.alert_history: List[Dict[str, Any]] = []
        self.max_history = 100
        
        # Configure default alert thresholds
        self.alert_thresholds = {
            "error_rate": 0.05,  # Alert if error rate exceeds 5%
            "detection_time_avg": 30.0,  # Alert if average detection time exceeds 30 seconds
            "memory_usage": 0.9,  # Alert if memory usage exceeds 90%
            "consecutive_errors": 5,  # Alert if 5 consecutive errors occur
            "api_latency": 2.0,  # Alert if API latency exceeds 2 seconds
            "queue_size": 100,  # Alert if task queue exceeds 100 tasks
            "model_performance": 0.7  # Alert if model performance drops below 70%
        }
        
        # Initialize channels from environment variables
        self._initialize_channels()
    
    def _initialize_channels(self):
        """Initialize notification channels from configuration"""
        # Email alerts
        email_enabled = os.environ.get('ENABLE_EMAIL_ALERTS', 'false').lower() == 'true'
        if email_enabled:
            self.register_email_channel(
                smtp_server=os.environ.get('SMTP_SERVER', 'smtp.example.com'),
                smtp_port=int(os.environ.get('SMTP_PORT', '587')),
                smtp_user=os.environ.get('SMTP_USER', ''),
                smtp_password=os.environ.get('SMTP_PASSWORD', ''),
                from_email=os.environ.get('ALERT_FROM_EMAIL', 'alerts@deepdefend.ai'),
                to_emails=os.environ.get('ALERT_TO_EMAILS', '').split(',')
            )
        
        # Webhook alerts (Slack, Discord, etc.)
        webhook_enabled = os.environ.get('ENABLE_WEBHOOK_ALERTS', 'false').lower() == 'true'
        if webhook_enabled:
            webhook_url = os.environ.get('WEBHOOK_URL', '')
            if webhook_url:
                self.register_webhook_channel(webhook_url)
        
        # Sentry alerts
        sentry_enabled = os.environ.get('ENABLE_SENTRY_ALERTS', 'false').lower() == 'true'
        if sentry_enabled:
            sentry_dsn = os.environ.get('SENTRY_DSN', '')
            if sentry_dsn:
                self.register_sentry_channel(sentry_dsn)
    
    def register_email_channel(self, smtp_server: str, smtp_port: int, 
                              smtp_user: str, smtp_password: str,
                              from_email: str, to_emails: List[str]):
        """Register email notification channel"""
        try:
            def email_sender(alert_data):
                try:
                    import smtplib
                    from email.mime.text import MIMEText
                    from email.mime.multipart import MIMEMultipart
                    
                    # Create message
                    msg = MIMEMultipart()
                    msg['From'] = from_email
                    msg['To'] = ', '.join(to_emails)
                    msg['Subject'] = f"DeepDefend Alert: {alert_data.get('type', 'Notification')} - {alert_data.get('severity', 'info').upper()}"
                    
                    # Create HTML body
                    body = f"""
                    <h2>DeepDefend Alert</h2>
                    <p><strong>Type:</strong> {alert_data.get('type', 'Notification')}</p>
                    <p><strong>Severity:</strong> {alert_data.get('severity', 'info').upper()}</p>
                    <p><strong>Message:</strong> {alert_data.get('message', 'No message provided')}</p>
                    <p><strong>Timestamp:</strong> {alert_data.get('timestamp', time.strftime("%Y-%m-%d %H:%M:%S"))}</p>
                    <h3>Details:</h3>
                    <pre>{json.dumps(alert_data, indent=2)}</pre>
                    """
                    
                    msg.attach(MIMEText(body, 'html'))
                    
                    # Connect to server and send
                    server = smtplib.SMTP(smtp_server, smtp_port)
                    server.starttls()
                    
                    if smtp_user and smtp_password:
                        server.login(smtp_user, smtp_password)
                        
                    server.send_message(msg)
                    server.quit()
                    
                    logger.info(f"Email alert sent to {', '.join(to_emails)}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to send email alert: {str(e)}")
                    return False
            
            self.alert_channels['email'] = email_sender
            logger.info("Email alert channel registered")
        except Exception as e:
            logger.error(f"Failed to register email channel: {str(e)}")
    
    def register_webhook_channel(self, webhook_url: str):
        """Register webhook notification channel (Slack, Discord, etc.)"""
        try:
            def webhook_sender(alert_data):
                try:
                    import requests
                    
                    # Format the data based on severity
                    severity = alert_data.get('severity', 'info')
                    color = {
                        'critical': '#FF0000',  # Red
                        'error': '#FFA500',     # Orange
                        'warning': '#FFFF00',   # Yellow
                        'info': '#0000FF'       # Blue
                    }.get(severity, '#808080')  # Gray default
                    
                    # Create payload (compatible with Slack and similar services)
                    payload = {
                        "attachments": [
                            {
                                "fallback": f"DeepDefend Alert: {alert_data.get('message', 'Notification')}",
                                "color": color,
                                "title": f"DeepDefend Alert: {alert_data.get('type', 'Notification')}",
                                "text": alert_data.get('message', 'No message provided'),
                                "fields": [
                                    {
                                        "title": "Severity",
                                        "value": severity.upper(),
                                        "short": True
                                    },
                                    {
                                        "title": "Timestamp",
                                        "value": alert_data.get('timestamp', time.strftime("%Y-%m-%d %H:%M:%S")),
                                        "short": True
                                    }
                                ],
                                "footer": "DeepDefend Monitoring System"
                            }
                        ]
                    }
                    
                    # Add detailed information for technical alerts
                    if alert_data.get('details'):
                        payload["attachments"][0]["fields"].append({
                            "title": "Details",
                            "value": f"```{json.dumps(alert_data['details'], indent=2)}```",
                            "short": False
                        })
                    
                    # Send the webhook
                    response = requests.post(webhook_url, json=payload)
                    response.raise_for_status()
                    
                    logger.info(f"Webhook alert sent: {response.status_code}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to send webhook alert: {str(e)}")
                    return False
            
            self.alert_channels['webhook'] = webhook_sender
            logger.info("Webhook alert channel registered")
        except Exception as e:
            logger.error(f"Failed to register webhook channel: {str(e)}")
    
    def register_sentry_channel(self, sentry_dsn: str):
        """Register Sentry as a notification channel for critical alerts"""
        try:
            import sentry_sdk
            
            def sentry_sender(alert_data):
                try:
                    if alert_data.get('severity') in ['critical', 'error']:
                        sentry_sdk.capture_message(
                            message=f"Alert: {alert_data.get('message', 'No message')}",
                            level=alert_data.get('severity', 'error'),
                            extras=alert_data
                        )
                        logger.info(f"Sentry alert sent")
                        return True
                    return False  # Don't send non-critical alerts to Sentry
                except Exception as e:
                    logger.error(f"Failed to send Sentry alert: {str(e)}")
                    return False
            
            self.alert_channels['sentry'] = sentry_sender
            logger.info("Sentry alert channel registered")
        except ImportError:
            logger.warning("Sentry SDK not installed. Run 'pip install sentry-sdk' to enable Sentry alerts.")
        except Exception as e:
            logger.error(f"Failed to register Sentry channel: {str(e)}")
    
    def create_alert_rule(self, rule_id: str, condition: Callable, 
                         alert_data: Dict[str, Any], cooldown_seconds: int = 300):
        """
        Create a new alert rule
        
        Args:
            rule_id: Unique identifier for the rule
            condition: Function that returns True when alert should trigger
            alert_data: Default alert data to send
            cooldown_seconds: Minimum time between alerts for this rule
        """
        self.alert_rules[rule_id] = {
            "condition": condition,
            "alert_data": alert_data,
            "cooldown": cooldown_seconds,
            "last_triggered": 0
        }
        logger.info(f"Alert rule created: {rule_id}")
    
    def check_rules(self, context: Dict[str, Any] = None) -> List[str]:
        """
        Check all alert rules with the given context
        
        Args:
            context: Data context for evaluating alert conditions
            
        Returns:
            List of rule IDs that were triggered
        """
        if context is None:
            context = {}
            
        triggered_rules = []
        current_time = time.time()
        
        for rule_id, rule in self.alert_rules.items():
            # Skip if in cooldown period
            if current_time - rule["last_triggered"] < rule["cooldown"]:
                continue
                
            try:
                # Check if condition is met
                if rule["condition"](context):
                    # Update last triggered time
                    rule["last_triggered"] = current_time
                    
                    # Combine default alert data with context
                    alert_data = {**rule["alert_data"], "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
                    if "details" not in alert_data:
                        alert_data["details"] = {}
                    
                    # Add relevant context to details
                    for key in ["error_rate", "memory_usage", "api_latency", "queue_size"]:
                        if key in context:
                            alert_data["details"][key] = context[key]
                    
                    # Generate alert ID
                    alert_id = str(uuid.uuid4())
                    alert_data["alert_id"] = alert_id
                    
                    # Add to history
                    self.add_to_history(alert_data)
                    
                    # Send alert through all channels
                    self.send_alert(alert_data)
                    
                    # Add to triggered rules
                    triggered_rules.append(rule_id)
            except Exception as e:
                logger.error(f"Error checking alert rule {rule_id}: {str(e)}")
        
        return triggered_rules
    
    def add_to_history(self, alert_data: Dict[str, Any]):
        """Add alert to history"""
        self.alert_history.append(alert_data)
        
        # Trim history if needed
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]
    
    def send_alert(self, alert_data: Dict[str, Any]) -> bool:
        """
        Send an alert through all registered channels
        
        Args:
            alert_data: Alert information
            
        Returns:
            True if alert was sent through at least one channel
        """
        if not self.alert_channels:
            logger.warning("No alert channels registered")
            return False
            
        success = False
        
        for channel_name, channel_func in self.alert_channels.items():
            try:
                channel_success = channel_func(alert_data)
                success = success or channel_success
            except Exception as e:
                logger.error(f"Error sending alert through {channel_name}: {str(e)}")
        
        return success
    
    def get_alert_history(self, limit: int = 50, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get alert history
        
        Args:
            limit: Maximum number of alerts to return
            severity: Filter by severity level
            
        Returns:
            List of alert data dictionaries
        """
        if severity:
            filtered_history = [alert for alert in self.alert_history 
                              if alert.get('severity') == severity]
        else:
            filtered_history = self.alert_history
            
        return filtered_history[-limit:]
    
    def get_thresholds(self) -> Dict[str, Any]:
        """Get current alert thresholds"""
        return self.alert_thresholds
    
    def update_threshold(self, name: str, value: Any) -> bool:
        """
        Update an alert threshold
        
        Args:
            name: Threshold name
            value: New threshold value
            
        Returns:
            True if threshold was updated successfully
        """
        if name in self.alert_thresholds:
            self.alert_thresholds[name] = value
            logger.info(f"Updated alert threshold: {name} = {value}")
            return True
        return False


class MetricsManager:
    """
    Advanced metrics collection and analysis with correlation capabilities
    """
    
    def __init__(self):
        self.metrics_data = {}
        self.time_series = {}
        self.correlation_data = {}
    
    def record_metric(self, name: str, value: Union[int, float, str], 
                     tags: Dict[str, str] = None, timestamp: float = None):
        """
        Record a metric value
        
        Args:
            name: Metric name
            value: Metric value
            tags: Associated tags for filtering and grouping
            timestamp: Optional timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()
            
        if tags is None:
            tags = {}
            
        # Create time series entry
        if name not in self.time_series:
            self.time_series[name] = []
            
        # Add to time series
        self.time_series[name].append({
            "timestamp": timestamp,
            "value": value,
            "tags": tags
        })
        
        # Limit time series to 1000 entries per metric
        if len(self.time_series[name]) > 1000:
            self.time_series[name] = self.time_series[name][-1000:]
        
        # Update current value
        if name not in self.metrics_data:
            self.metrics_data[name] = {
                "current": value,
                "min": value if isinstance(value, (int, float)) else None,
                "max": value if isinstance(value, (int, float)) else None,
                "count": 1,
                "last_updated": timestamp
            }
        else:
            self.metrics_data[name]["current"] = value
            self.metrics_data[name]["count"] += 1
            self.metrics_data[name]["last_updated"] = timestamp
            
            if isinstance(value, (int, float)):
                if self.metrics_data[name]["min"] is None or value < self.metrics_data[name]["min"]:
                    self.metrics_data[name]["min"] = value
                    
                if self.metrics_data[name]["max"] is None or value > self.metrics_data[name]["max"]:
                    self.metrics_data[name]["max"] = value
    
    def get_metric(self, name: str) -> Optional[Dict[str, Any]]:
        """Get current value and metadata for a metric"""
        return self.metrics_data.get(name)
    
    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all metrics"""
        return self.metrics_data
    
    def get_time_series(self, name: str, start_time: Optional[float] = None, 
                       end_time: Optional[float] = None, 
                       limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get time series data for a metric
        
        Args:
            name: Metric name
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum number of data points to return
            
        Returns:
            List of time series data points
        """
        if name not in self.time_series:
            return []
            
        series = self.time_series[name]
        
        # Apply time filters
        if start_time is not None:
            series = [point for point in series if point["timestamp"] >= start_time]
            
        if end_time is not None:
            series = [point for point in series if point["timestamp"] <= end_time]
        
        # Return limited number of points
        return series[-limit:]
    
    def calculate_correlation(self, metric1: str, metric2: str, window: int = 100) -> Optional[float]:
        """
        Calculate correlation coefficient between two metrics
        
        Args:
            metric1: First metric name
            metric2: Second metric name
            window: Number of data points to use
            
        Returns:
            Correlation coefficient or None if calculation failed
        """
        try:
            import numpy as np
            
            if metric1 not in self.time_series or metric2 not in self.time_series:
                return None
                
            # Get recent data points
            series1 = self.time_series[metric1][-window:]
            series2 = self.time_series[metric2][-window:]
            
            # Need enough data points
            if len(series1) < 2 or len(series2) < 2:
                return None
                
            # Extract values
            values1 = [point["value"] for point in series1 if isinstance(point["value"], (int, float))]
            values2 = [point["value"] for point in series2 if isinstance(point["value"], (int, float))]
            
            # Make sure we have numeric values
            if len(values1) < 2 or len(values2) < 2:
                return None
                
            # Align the arrays to the same length
            min_len = min(len(values1), len(values2))
            values1 = values1[-min_len:]
            values2 = values2[-min_len:]
            
            # Calculate correlation coefficient
            correlation = np.corrcoef(values1, values2)[0, 1]
            
            # Cache the result
            key = f"{metric1}:{metric2}"
            self.correlation_data[key] = {
                "coefficient": correlation,
                "window": min_len,
                "timestamp": time.time()
            }
            
            return correlation
            
        except Exception as e:
            logger.error(f"Error calculating correlation: {str(e)}")
            return None
    
    def get_correlations(self, min_coefficient: float = 0.5) -> Dict[str, Dict[str, Any]]:
        """
        Get all cached correlations above the minimum coefficient
        
        Args:
            min_coefficient: Minimum absolute correlation coefficient to include
            
        Returns:
            Dictionary of correlation data
        """
        return {
            key: data for key, data in self.correlation_data.items()
            if abs(data["coefficient"]) >= min_coefficient
        }
    
    def analyze_metric_trends(self, metric_name: str, window: int = 100) -> Dict[str, Any]:
        """
        Analyze trends for a specific metric
        
        Args:
            metric_name: Metric to analyze
            window: Number of data points to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        try:
            if metric_name not in self.time_series:
                return {"error": "Metric not found"}
                
            series = self.time_series[metric_name][-window:]
            
            if len(series) < 2:
                return {"error": "Not enough data points"}
                
            # Extract values and timestamps
            values = [point["value"] for point in series if isinstance(point["value"], (int, float))]
            timestamps = [point["timestamp"] for point in series if isinstance(point["value"], (int, float))]
            
            if len(values) < 2:
                return {"error": "Not enough numeric data points"}
                
            import numpy as np
            
            # Calculate basic statistics
            mean = np.mean(values)
            median = np.median(values)
            std_dev = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            
            # Calculate slope (simple linear regression)
            x = np.array(range(len(values)))
            z = np.polyfit(x, values, 1)
            slope = z[0]
            
            # Determine trend direction
            if abs(slope) < 0.0001:
                trend = "stable"
            elif slope > 0:
                trend = "increasing"
            else:
                trend = "decreasing"
            
            # Calculate rate of change
            time_diff = timestamps[-1] - timestamps[0]
            if time_diff > 0:
                rate_of_change = (values[-1] - values[0]) / time_diff
            else:
                rate_of_change = 0
            
            # Return analysis
            return {
                "metric": metric_name,
                "data_points": len(values),
                "time_range": [timestamps[0], timestamps[-1]],
                "statistics": {
                    "mean": mean,
                    "median": median,
                    "std_dev": std_dev,
                    "min": min_val,
                    "max": max_val
                },
                "trend": {
                    "direction": trend,
                    "slope": slope,
                    "rate_of_change": rate_of_change
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing metric trends: {str(e)}")
            return {"error": str(e)}


# Create singleton instances
sentry_monitoring = SentryMonitoring()
alerting_service = AlertingService()
metrics_manager = MetricsManager()

def capture_exception(exception: Optional[Exception] = None, **kwargs):
    """Capture an exception and send it to monitoring services"""
    sentry_monitoring.capture_exception(exception, **kwargs)

def capture_message(message: str, level: str = 'info', **kwargs):
    """Capture a message and send it to monitoring services"""
    sentry_monitoring.capture_message(message, level, **kwargs)

def start_transaction(name: str, op: str = None, **kwargs):
    """Start a transaction for performance monitoring"""
    return sentry_monitoring.start_transaction(name, op, **kwargs)

def send_alert(alert_type: str, message: str, severity: str = 'info', details: Dict[str, Any] = None):
    """Send an alert through all configured channels"""
    alert_data = {
        "type": alert_type,
        "message": message,
        "severity": severity,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "details": details or {}
    }
    return alerting_service.send_alert(alert_data)

def record_metric(name: str, value: Union[int, float, str], tags: Dict[str, str] = None):
    """Record a metric value"""
    metrics_manager.record_metric(name, value, tags)

def get_metrics():
    """Get all metrics"""
    return metrics_manager.get_metrics()

def get_alert_history(limit: int = 50, severity: Optional[str] = None):
    """Get alert history"""
    return alerting_service.get_alert_history(limit, severity)
