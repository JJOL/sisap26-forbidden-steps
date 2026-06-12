#include <iostream>
#include <string>
#include <vector>
#include <array>

#include <stdexcept>
#include <chrono>

#include <H5Cpp.h>

// milisecond-accurate clock class
struct Clock {
    using time_point = std::chrono::steady_clock::time_point;
    time_point startTime;

    void start() {
        startTime = std::chrono::steady_clock::now();
    }

    long long stop() const {
        const auto endTime = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    }
};

enum RetrievalStrategy {
    BRUTE_FORCE
};

struct RetrievalResult {
    std::vector<std::vector<long long>> topIndicesByQuery;
    int kTop;
};

// Compressed Sparse Row Matrix Representation
struct CSRMatrix {
    // indptr store the entries offset for each row.
    //  So indptr[1]=200 means first column entry of row_1 is at position 200 for indices and data.
    std::vector<long long> indptr;
    // indices store the column index of each entry.
    //  So indices[200]=32 means that entry 200 (corresponds to row_1) is for column 32
    std::vector<int> indices;
    // data store the value of each entry
    //  So data[200]=60 means that entry 200 (row_1 col 32) has value 60
    std::vector<float> data;

    long long rows;
    long long cols; // although int would work and corresponds to indices type
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

CSRMatrix loadCSRMatrixFromH5Group(const H5::Group &group) {
    CSRMatrix matrix;

    auto shape = getH5GroupShape(group);
    // std::cout << "Shape = (" << shape[0] << "," << shape[1] << ")" << std::endl;
    matrix.rows = shape[0];
    matrix.cols = shape[1];

    matrix.indptr = get1DDataset<long long>(group, "indptr", H5::PredType::NATIVE_LLONG);
    matrix.indices = get1DDataset<int>(group, "indices", H5::PredType::NATIVE_INT);
    matrix.data = get1DDataset<float>(group, "indptr", H5::PredType::NATIVE_FLOAT);

    if (matrix.indptr.size() != matrix.rows + 1) {
        throw std::runtime_error("indptr is mising some entries to match!");
    }

    return matrix;
}

RetrievalResult bruteForceRetrieval(const CSRMatrix &db, const CSRMatrix &queries, int kTop) {

}

RetrievalResult runRetrieval(const CSRMatrix &db, const CSRMatrix &queries, int kTop, RetrievalStrategy strategy) {
    RetrievalResult results;
    switch (strategy)
    {
    case RetrievalStrategy::BRUTE_FORCE:
        results = bruteForceRetrieval(db, queries, kTop);
        break;
    default:
        break;
    }

    return results;
}

int main (int argc, char **argv) {
    std::string dbFileName = (argc > 1) ? argv[1] : "data/fiqa-dev.h5";
    const RetrievalStrategy strategy = RetrievalStrategy::BRUTE_FORCE;
    const int kTop = 30;

    // ------------------------------------------------------------------------------
    // Load Corpus and Queries
    // ------------------------------------------------------------------------------
    std::cout << "Reading sparse matrices from '" << dbFileName << "'." << std::endl;
    // Load File

    Clock clk;
    clk.start();
    H5::H5File file(dbFileName, H5F_ACC_RDONLY);

    H5::Group trainGroup = file.openGroup("train");
    auto db = loadCSRMatrixFromH5Group(trainGroup);

    H5::Group queriesGroup = file.openGroup("otest/queries");
    auto queries = loadCSRMatrixFromH5Group(queriesGroup);

    auto durationMs = clk.stop(); 

    std::cout << "Train Matrix Shape: (" << db.rows << "," << db.cols << ")" << std::endl;
    std::cout << "Queries Matrix Shape: (" << queries.rows << "," << queries.cols << ")" << std::endl;
    std::cout << "Time to load csr matrices: " << durationMs << " ms." << std::endl;

    if (db.cols != queries.cols) {
        throw std::runtime_error("Train and queries column dimension must match.");
    }

    // ------------------------------------------------------------------------------
    // Process Queries
    // ------------------------------------------------------------------------------
    auto results = runRetrieval(db, queries, kTop, strategy);

    // ------------------------------------------------------------------------------
    // Print Results
    // ------------------------------------------------------------------------------


    return 0;
}