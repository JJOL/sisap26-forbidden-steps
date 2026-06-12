#pragma once

#include <chrono>
#include <string>
#include <array>
#include <vector>
#include <cstdlib>
#include <unordered_map>
#include <stdexcept>
#include <filesystem>
#include <fstream>

#include <H5Cpp.h>

struct Clock {
    using time_point = std::chrono::steady_clock::time_point;
    time_point startTime;

    void start() {
        startTime = std::chrono::steady_clock::now();
    }

    long long elapsedMs() const {
        const auto endTime = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    }
};

// Gets the shape value (n,m)
std::array<long long, 2> getH5GroupShape(const H5::Group &group) {
    if (!group.attrExists("shape")) {
        throw std::runtime_error("Attribute 'shape' not found in '" + group.getObjName() + "' group.");
    }
    auto attr = group.openAttribute("shape");
    auto space = attr.getSpace();


    if (space.getSimpleExtentNdims() != 1) {
        throw std::runtime_error("Attribute 'shape' of group '" + group.getObjName() + "' is not a 1-dimensional array");
    }
    hsize_t size;
    space.getSimpleExtentDims(&size);
    if (size != 2) {
        throw std::runtime_error("Attribute 'shape' of group '" + group.getObjName() + "' must have only 2 numbers.");
    }

    std::array<long long, 2> shape;
    attr.read(H5::PredType::NATIVE_LLONG, shape.data());
    
    return shape;
}

template<typename T>
std::vector<T> get1DDataset(const H5::Group &group, const std::string &datasetName, const H5::PredType &type) {
    auto dataset = group.openDataSet(datasetName);
    auto space = dataset.getSpace();
    if (space.getSimpleExtentNdims() != 1) {
        throw std::runtime_error("'indptr' of group '" + group.getObjName() + "' must be 1-dimensional.");
    }
    hsize_t size;
    space.getSimpleExtentDims(&size);

    std::vector<T> indPtrBuf(size);
    dataset.read(indPtrBuf.data(), type);

    return indPtrBuf;
}

std::string findStringFieldInJson(std::ifstream& jsonFile, const std::string& fieldName, const std::string& errorMsg) {
	std::string line;
	while (std::getline(jsonFile, line)) {
		const std::size_t fieldPos = line.find('"' + fieldName + '"');
		if (fieldPos != std::string::npos) {
			const std::size_t colonPos = line.find(':', fieldPos);
			if (colonPos != std::string::npos) {
				const std::size_t quoteStart = line.find('"', colonPos);
				const std::size_t quoteEnd = line.find('"', quoteStart + 1);
				if (quoteStart != std::string::npos && quoteEnd != std::string::npos) {
					return line.substr(quoteStart + 1, quoteEnd - quoteStart - 1);
				}
			}
		}
	}
	throw std::runtime_error(errorMsg);
}

// ./build/main.exe --inputFolder data/fiqa-dev --outputFolder results/task-3-spot-check --retrivalStrategy inverted --k 30 --task task3 --dataset fiqa-dev
void printUsage() {
    std::cout << "Usage: Indexer [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --help, -h       Show this help message and exit" << std::endl;
    std::cout << "  --inputFolder    Path to the input folder containing HDF5 files" << std::endl;
    std::cout << "  --outputFolder   Path to the output folder for results" << std::endl;
    std::cout << "  --kTop           Number of top results to retrieve (default: 30)" << std::endl;
    std::cout << "  --strategy       Retrieval strategy to use (e.g., brute-force, inverted-index, hnsw)" << std::endl;
    std::cout << "  --task           Task name (e.g., task3)" << std::endl;
    std::cout << "  --dataset        Dataset name (e.g., fiqa-dev)" << std::endl;
}

void printArguments(int argc, char **argv, const std::unordered_map<std::string, std::string> &argsMap) {
    std::cout << "Parsed Arguments:" << std::endl;
    for (const auto &[key, value] : argsMap) {
        std::cout << "  " << key << ": " << value << std::endl;
    }
}

using ArgumentsMap = std::unordered_map<std::string, std::string>;

void parseArguments(int argc, char **argv, ArgumentsMap &argsMap) {
    if (argc < 2) {
        printUsage();
        exit(0);
    }

    std::unordered_map<std::string, std::string> optionNameMap = {
        // {"--help", "help"},
        // {"-h", "help"},
        {"--inputFolder", "inputFolder"},
        {"--outputFolder", "outputFolder"},
        {"--kTop", "kTop"},
        {"--strategy", "strategy"},
        {"--task", "task"},
        {"--dataset", "dataset"},
        // Add more options as needed
    };

    if (argv[1] == std::string("--help") || argv[1] == std::string("-h")) {
        printUsage();
        exit(0);
    }

    std::cout << "Parsing arguments..." << std::endl;
    for (int i = 1; i < argc - 1; i += 2) {
        std::cout << "Processing argument pair: " << argv[i] << " " << argv[i + 1] << std::endl;
        std::string key = argv[i];
        std::string value = argv[i + 1];
        if (optionNameMap.find(key) == optionNameMap.end()) {
            std::cerr << "Unknown option: " << key << std::endl;
            printUsage();
            exit(1);
        }
        argsMap[optionNameMap[key]] = value;
    }
    
    // all options are mandatory for now, check that they are all present
    for (const auto &[option, name] : optionNameMap) {
        if (argsMap.find(name) == argsMap.end()) {
            std::cerr << "Missing required option: " << option << std::endl;
            printUsage();
            exit(1);
        }
    }
}


std::string getInputFilePath(std::string inputPath) {
    if (std::filesystem::path(inputPath).extension() == ".h5") {
        std::cout << "Using input file: " << inputPath << '\n';
    } else {
        std::cout << "Input path is not an .h5 file, treating it as a directory: " << inputPath << '\n';
        std::filesystem::path configPath = std::filesystem::path(inputPath) / "config.json";
        if (!std::filesystem::exists(configPath)) {
            std::cerr << "Config file not found in directory: " << configPath << '\n';
            exit(1);
        }
        std::ifstream configFile(configPath);
        if (!configFile.is_open()) {
            std::cerr << "Failed to open config file: " << configPath << '\n';
            exit(1);
        }
        std::string h5Filename = findStringFieldInJson(configFile, "filename", "H5 file path not specified in config");
        std::filesystem::path h5Path = std::filesystem::path(inputPath) / h5Filename;
        if (!std::filesystem::exists(h5Path)) {
            std::cerr << "H5 file specified in config not found: " << h5Path << '\n';
            exit(1);
        }
        std::cout << "Using H5 file from config: " << h5Path << '\n';
        inputPath = h5Path.string();
    }
    return inputPath;
}