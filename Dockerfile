FROM ubuntu:20.04

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libhdf5-dev

# Build the application
COPY solution .
RUN mkdir -p build

# if platform is linux/amd64
# -L path is /usr/lib/x86_64-linux-gnu/hdf5/serial/
# if platform is linux/arm64
# -L path is /usr/lib/aarch64-linux-gnu/hdf5/serial/ if platform is linux/arm64
RUN c++ -std=c++17 -O3 \
       -fopenmp \
       -I/usr/include/hdf5/serial \
       -Iinclude \
       src/main.cpp -o build/main.exe \
       -L/usr/lib/aarch64-linux-gnu/hdf5/serial/ \
       -lhdf5_cpp -lhdf5

# docker build -t cpp-hdf5:latest .
# then to run it interactively: it could be: docker run -it --rm -v $(pwd)/data:/app/data:ro -v $(pwd)/results:/app/results:rw cpp-hdf5:latest bash