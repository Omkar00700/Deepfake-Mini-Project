
#!/bin/bash
set -e

# Wait for Redis if enabled
if [ "$ENABLE_ASYNC_PROCESSING" = "true" ]; then
    echo "Waiting for Redis..."
    
    # Extract host and port from REDIS_URL
    REDIS_HOST=$(echo $REDIS_URL | sed -e 's/^redis:\/\///' | sed -e 's/:.*$//')
    REDIS_PORT=$(echo $REDIS_URL | grep -o ':[0-9]*' | sed -e 's/://')
    
    # Default to standard Redis port if not specified
    if [ -z "$REDIS_PORT" ]; then
        REDIS_PORT=6379
    fi
    
    # Wait for Redis to be ready
    until nc -z $REDIS_HOST $REDIS_PORT; do
        echo "Redis not available yet - sleeping"
        sleep 1
    done
    
    echo "Redis is ready!"
fi

# Start Celery worker if async processing is enabled
if [ "$ENABLE_ASYNC_PROCESSING" = "true" ]; then
    echo "Starting Celery worker..."
    celery -A task_queue.celery_app worker --loglevel=info &
fi

# Start the API server
if [ "$ENVIRONMENT" = "production" ]; then
    echo "Starting DeepDefend API in production mode..."
    exec gunicorn --bind 0.0.0.0:5000 --workers 4 --timeout 120 app:app
else
    echo "Starting DeepDefend API in development mode..."
    exec python app.py
fi
