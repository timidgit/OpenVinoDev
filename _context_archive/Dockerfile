# Enhanced OpenVINO GenAI Chat Application
# Production-ready Docker container with NPU support
# Based on strategic roadmap Phase 2.1

FROM openvino/ubuntu22_dev:latest

# Set working directory
WORKDIR /app

# Install system dependencies for NPU and Python packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application source code
COPY app/ ./app/
COPY context/ ./context/
COPY tests/ ./tests/
COPY main.py .
COPY config.json .
COPY config.example.json .
COPY *.md ./
COPY *.ini .
COPY .flake8 .

# Create cache directory
RUN mkdir -p /app/cache

# Expose Gradio port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/healthcheck || exit 1

# Configure Gradio to listen on all interfaces
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Default command with production-ready settings
CMD ["python3", "main.py", "--device", "AUTO", "--debug"]

# Metadata
LABEL maintainer="sbran"
LABEL description="Enhanced Phi-3-mini-128k-instruct Chat Application with OpenVINO GenAI"
LABEL version="2.0.0"