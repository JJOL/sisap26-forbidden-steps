#!/bin/bash

c++ -std=c++17 -I/opt/homebrew/include -L/opt/homebrew/lib -lhdf5_cpp -O3 main.cpp -o own_main.exe
./own_main.exe