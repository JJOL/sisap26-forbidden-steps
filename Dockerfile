FROM python:3.9-slim

WORKDIR /app

# Install system dependencies if any
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch CPU only
RUN pip install torch~=2.4.0 --index-url https://download.pytorch.org/whl/cpu

# Copy source code
COPY . .

# Default command
ENTRYPOINT ["python", "search.py"]
