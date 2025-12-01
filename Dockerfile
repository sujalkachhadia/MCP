# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for some python packages
# libgomp1 is often needed for PyTorch/Sentence-Transformers
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create a temp directory for file uploads to avoid permission issues
RUN mkdir -p temp_uploads && chmod 777 temp_uploads

# The CMD is handled by docker-compose, so we leave it empty or default
CMD ["python", "server.py"]