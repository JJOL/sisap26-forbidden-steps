# FROM python:3.9-slim

# WORKDIR /app

# # Install system dependencies if any
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# # Copy requirements
# COPY requirements.txt .

# # Install dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Install PyTorch CPU only
# RUN pip install torch~=2.4.0 --index-url https://download.pytorch.org/whl/cpu

# # Copy source code
# COPY . .



# cpp dockerfile with support of c++17 and HD5F library
FROM ubuntu:20.04

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libhdf5-dev
    # cmake \
    # && rm -rf /var/lib/apt/lists/*

# Copy source code
# COPY . .

# Build the application
COPY main.cpp .
RUN c++ -std=c++17 -O3 \
        -I/usr/include/hdf5/serial \
        main.cpp -o own_main.exe \
        -L/usr/lib/aarch64-linux-gnu/hdf5/serial \
        -lhdf5_cpp -lhdf5
# our compilation: c++ -std=c++17 -I/opt/homebrew/include -L/opt/homebrew/lib -lhdf5_cpp -O3 main.cpp -o own_main.exe


# Run the application
# CMD ["ls", "-l"]

# then to run it interactively: it could be: docker run -it --rm -v $(pwd):/app cpp-hdf5:latest bash