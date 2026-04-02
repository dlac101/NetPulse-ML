"""Gunicorn production configuration for Linux deployment."""

import multiprocessing

# Server
bind = "0.0.0.0:8000"
workers = min(multiprocessing.cpu_count(), 4)  # Cap at 4 (each loads ML models)
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 120
keepalive = 5

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Security
limit_request_line = 8190
limit_request_fields = 100
