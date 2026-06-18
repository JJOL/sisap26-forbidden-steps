.PHONY: all clean

# c++ -std=c++17 \
# 	-Xpreprocessor -fopenmp \
# 	main.cpp \
# 	-I/opt/homebrew/opt/libomp/include \
# 	-I/opt/homebrew/include \
# 	-L/opt/homebrew/lib \
# 	-lhdf5_cpp \
# 	-L/opt/homebrew/opt/libomp/lib \
# 	-lomp \
# 	-o main.exe
main.exe:
	mkdir -p build
	clang++ -std=c++17 -O3 \
		-Xpreprocessor -fopenmp \
		src/main.cpp \
		-I/opt/homebrew/opt/libomp/include \
		-I/opt/homebrew/include \
		-Iinclude \
		-L/opt/homebrew/lib \
		-lhdf5_cpp \
		-L/opt/homebrew/opt/libomp/lib \
		-lomp \
		-o build/main.exe
clean:
	rm -rf build

test: main.exe
	./build/main.exe --inputFolder ../data/task-3-spot-check --outputFolder ../results/task-3-spot-check --strategy blocked_inverted --kTop 30 --task task3 --dataset fiqa-dev --threads -1

all: main.exe