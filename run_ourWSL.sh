#!/bin/bash

# Compile the code with OpenMP and HDF5 for Ubuntu/WSL
g++ -std=c++17 -O3 -fopenmp -I/usr/include/hdf5/serial main.cpp -o own_main -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5_cpp -lhdf5

# Run the executable
./own_main data/nq/nq.h5