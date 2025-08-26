# Dockerfile-gpu
#FROM cyclegan-turbo:latest
FROM nvcr.io/nvidia/pytorch:23.12-py3

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED=True

# Fix protobuf conflict with ONNX
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Install system dependencies and production dependencies.
RUN apt-get update && apt-get install -y \
    curl \
    libgomp1 \
    libusb-1.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy application dependency manifests to the container image.
# Copying this separately prevents re-running pip install on every code change.
COPY ./requirements.txt ./

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy local code to the container image.
ENV APP_HOME /inference_app
WORKDIR $APP_HOME
COPY ./inference_app ./
COPY ./src ./src

# COPY model checkpoints to the container image.
COPY ./checkpoints ./checkpoints

# Expose the port that the app runs on
# Vertex AI will set AIP_HTTP_PORT environment variable
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${AIP_HTTP_PORT:-8080}${AIP_HEALTH_ROUTE:-/health} || exit 1

# Run the application
CMD ["python", "main.py"]