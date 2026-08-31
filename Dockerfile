# Streamlit Frontend for PaddleOCR-VL Document Parser
#
# This is a lightweight container that runs the Streamlit web interface.
# It communicates with the PaddleOCR-VL API service for document processing.
#
# Build: docker build -t paddleocr-vl-streamlit .
# Run:   docker run -p 8501:8501 paddleocr-vl-streamlit

FROM python:3.11-slim

# Host bind-mount owner (./data, ./logs). Override at build with APP_UID/APP_GID.
ARG APP_UID=1000
ARG APP_GID=1000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user matching the host uid/gid so bind-mounted ./data is writable
RUN groupadd --gid ${APP_GID} appuser && \
    useradd --uid ${APP_UID} --gid ${APP_GID} --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create directories for logs, data (results cache), and temp files
RUN mkdir -p /app/logs /app/data /tmp/uploads && \
    chown -R appuser:appuser /app /tmp/uploads

# Switch to non-root user
USER appuser

# Streamlit configuration
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--browser.gatherUsageStats=false"]
