FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV WORKDIR=/app
ENV HF_HOME=/app/cache/huggingface

WORKDIR $WORKDIR

# Install system dependencies required for OpenCV and compiling C++ native extensions
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies first (leverages Docker caching)
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose API port
EXPOSE 8000

# Run the FastAPI server natively
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
