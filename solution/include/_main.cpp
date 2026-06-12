#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <stdexcept>
#include <cstdlib>
#include <cassert>
#include "utils.hpp"

#include <H5Cpp.h>

enum RetrievalStrategy {
    BRUTE_FORCE,
    INVERTED_INDEX
};

using ScoreDoc = std::pair<float, int>;

struct RetrievalResult {
    std::vector<std::vector<int>> topIndicesByQuery;
	std::vector<std::vector<float>> topScoresByQuery;
    long long prepElapsedMs = 0;
	long long elapsedMs = 0;
	long long totalPostingsVisited = 0;
	long long totalTouchedDocs = 0;
	long long maxTouchedDocs = 0;
	long long queryCount = 0;
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

RetrievalStrategy parseStrategy(const std::string& name) {
    if (name == "brute") {
        return RetrievalStrategy::BRUTE_FORCE;
    } else if (name == "inverted") {
        return RetrievalStrategy::INVERTED_INDEX;
    } else {
        throw std::runtime_error("Unknown strategy name: " + name);
    }
}

float dotProductRows(const CSRMatrix& a, long long aRow, const CSRMatrix& b, long long bRow) {
	const long long aStart = a.indptr[static_cast<std::size_t>(aRow)];
	const long long aEnd = a.indptr[static_cast<std::size_t>(aRow + 1)];
	const long long bStart = b.indptr[static_cast<std::size_t>(bRow)];
	const long long bEnd = b.indptr[static_cast<std::size_t>(bRow + 1)];

	long long i = aStart;
	long long j = bStart;
	float sum = 0.0F;

	while (i < aEnd && j < bEnd) {
        assert(i < a.indices.size() && j < b.indices.size());
		const int ai = a.indices[static_cast<std::size_t>(i)];
		const int bj = b.indices[static_cast<std::size_t>(j)];
		if (ai == bj) {
			sum += a.data[i] * b.data[static_cast<std::size_t>(j)];
            sum += 1.0;
			++i;
			++j;
		} else if (ai < bj) {
			++i;
		} else {
			++j;
		}
	}

	return sum;
}

std::vector<ScoreDoc> topKScoredQuickselect(const std::vector<float>& scores, int k) {
    std::vector<ScoreDoc> scored;
	scored.reserve(scores.size());
    for (int i = 0; i < scores.size(); ++i) {
		scored.emplace_back(scores[i], i);
	}

    k = std::min(k, (int)scored.size());

    if (k < scored.size()) {
		std::nth_element(scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(k), scored.end(),
					 [](const ScoreDoc& lhs, const ScoreDoc& rhs) {
						 if (lhs.first == rhs.first) {
							 return lhs.second < rhs.second;
						 }
						 return lhs.first > rhs.first;
					 });
	}
    std::sort(scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(k),
		  [](const ScoreDoc& lhs, const ScoreDoc& rhs) {
			  if (lhs.first == rhs.first) {
				  return lhs.second < rhs.second;
			  }
			  return lhs.first > rhs.first;
		  });

    scored.resize(k);
	return scored;
}

RetrievalResult runBruteForceRetrieval(const CSRMatrix& db, const CSRMatrix& queries, std::size_t kTop) {
	Clock searchClock;
	searchClock.start();

	RetrievalResult result;
	result.topIndicesByQuery.resize(static_cast<std::size_t>(queries.rows));
	result.topScoresByQuery.resize(static_cast<std::size_t>(queries.rows));

	std::vector<float> scores(static_cast<std::size_t>(db.rows), 0.0F);
	for (long long q = 0; q < queries.rows; ++q) {
        std::cout << "Processing query " << (q + 1) << "/" << queries.rows << std::endl;
		for (long long d = 0; d < db.rows; ++d) {
            assert(d < scores.size());
			scores[static_cast<std::size_t>(d)] = dotProductRows(db, d, queries, q);
		}
        std::cout << "Processed query " << (q + 1) << "/" << queries.rows << '\n';
		const std::vector<ScoreDoc> top = topKScoredQuickselect(scores, kTop);
		auto& topIndices = result.topIndicesByQuery[static_cast<std::size_t>(q)];
		auto& topScores = result.topScoresByQuery[static_cast<std::size_t>(q)];
		topIndices.reserve(top.size());
		topScores.reserve(top.size());
		for (const ScoreDoc& item : top) {
			topScores.push_back(item.first);
			topIndices.push_back(item.second);
		}

		if ((q + 1) % 100 == 0 || q + 1 == queries.rows) {
		}
	}

	result.elapsedMs = searchClock.stop();
	return result;
}

RetrievalResult runRetrieval(const CSRMatrix &db, const CSRMatrix &queries, int kTop, RetrievalStrategy strategy) {
    RetrievalResult results;
    switch (strategy)
    {
    case RetrievalStrategy::BRUTE_FORCE:
        results = runBruteForceRetrieval(db, queries, kTop);
        break;
    default:
        throw std::runtime_error("Unknown retrieval strategy.");
    }

    return results;
}

int main (int argc, char **argv) {
    ArgumentsMap argsMap;
    parseArguments(argc, argv, argsMap);

    // if help option is present, print usage and exit
    if (argsMap.find("help") != argsMap.end()) {
        printUsage();
        exit(0);
    }

    // ./build/main.exe --inputFolder data/fiqa-dev --outputFolder results/task-3-spot-check --retrivalStrategy inverted --k 30 --task task3 --dataset fiqa-dev
    std::cout << "Received the following parameters:" << std::endl;
    for (const auto &[key, value] : argsMap) {
        std::cout << key << ": " << value << std::endl;
    }

    std::string filePath = getInputFilePath(argsMap["inputFolder"]);
    int kTop = std::stoi(argsMap["kTop"]);
    std::string datasetName = argsMap["dataset"];
    std::string taskName = argsMap["task"];
    std::string outputPath = argsMap["outputFolder"] + "/" + taskName + "_" + datasetName + "_k=" + std::to_string(kTop) + argsMap["strategy"] +  ".h5";
    RetrievalStrategy strategy = parseStrategy(argsMap["strategy"]);

    // ------------------------------------------------------------------------------
    // Load Corpus and Queries
    // ------------------------------------------------------------------------------
    std::cout << "Reading sparse matrices from '" << filePath << "'." << std::endl;

    Clock clk;
    clk.start();
    H5::H5File file(filePath, H5F_ACC_RDONLY);

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
    auto retrieval = runRetrieval(db, queries, kTop, strategy);
    if (strategy == RetrievalStrategy::BRUTE_FORCE) {
        std::cout << "Brute-force retrieval completed." << std::endl;
    } else if (strategy == RetrievalStrategy::INVERTED_INDEX) {
        std::cout << "Inverted index retrieval completed." << std::endl;
        std::cout << "Inverted index build time: " << retrieval.prepElapsedMs << " ms" << '\n';
        std::cout << "Inverted-index search time: " << retrieval.elapsedMs << " ms" << '\n';
        const double avgTouched = static_cast<double>(retrieval.totalTouchedDocs) /
                            static_cast<double>(retrieval.queryCount);
        const double avgPostingsVisited = static_cast<double>(retrieval.totalPostingsVisited) /
                                static_cast<double>(retrieval.queryCount);
        const double touchedPct = db.rows > 0
            ? (100.0 * avgTouched / static_cast<double>(db.rows))
            : 0.0;

        std::cout << "Avg touched docs/query: " << avgTouched
                    << " (" << touchedPct << "% of db rows)" << '\n';
        std::cout << "Max touched docs/query: " << retrieval.maxTouchedDocs << '\n';
        std::cout << "Avg postings visited/query: " << avgPostingsVisited << '\n';
    } else {
        std::cout << "Unknown retrieval strategy." << std::endl;
    }

    // ------------------------------------------------------------------------------
    // Print Results
    // ------------------------------------------------------------------------------


    return 0;
}