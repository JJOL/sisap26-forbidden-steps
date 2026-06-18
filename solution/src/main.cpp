#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <stdexcept>
#include <cstdlib>
#include <cassert>
#include <omp.h>
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

// Similar to CSRMatrix but with names for inverted index representation
struct InvertedIndex {
    std::vector<long long> termIndptr; // indptr for terms
    std::vector<int> docIds; // indices for documents
    std::vector<float> docValues; // data for documents
    long long terms; // number of terms
};

CSRMatrix loadCSRMatrixFromH5Group(const H5::Group &group) {
    CSRMatrix matrix;

    auto shape = getH5GroupShape(group);
    // std::cout << "Shape = (" << shape[0] << "," << shape[1] << ")" << std::endl;
    matrix.rows = shape[0];
    matrix.cols = shape[1];

    matrix.indptr = get1DDataset<long long>(group, "indptr", H5::PredType::NATIVE_LLONG);
    matrix.indices = get1DDataset<int>(group, "indices", H5::PredType::NATIVE_INT);
    matrix.data = get1DDataset<float>(group, "data", H5::PredType::NATIVE_FLOAT);

    if (matrix.indptr.size() != matrix.rows + 1) {
        throw std::runtime_error("indptr is mising some entries to match!");
    }

    if (matrix.indices.size() != matrix.data.size()) {
        throw std::runtime_error("indices and data must have the same number of entries!");
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
	const long long aStart = a.indptr[(aRow)];
	const long long aEnd = a.indptr[(aRow + 1)];
	const long long bStart = b.indptr[(bRow)];
	const long long bEnd = b.indptr[(bRow + 1)];

	long long i = aStart;
	long long j = bStart;
	float sum = 0.0F;

	while (i < aEnd && j < bEnd) {
		const int ai = a.indices[(i)];
		const int bj = b.indices[(j)];
		if (ai == bj) {
			sum += a.data[i] * b.data[(j)];
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

RetrievalResult runBruteForceRetrieval(const CSRMatrix& db, const CSRMatrix& queries, std::size_t kTop, int numThreads) {
	Clock searchClock;
	searchClock.start();

	RetrievalResult result;
	result.topIndicesByQuery.resize(queries.rows);
	result.topScoresByQuery.resize(queries.rows);

    // set number of threads for OpenMP
    omp_set_num_threads(numThreads);
    #pragma omp parallel
    {
        std::vector<float> scores(db.rows, 0.0F);

        #pragma omp for schedule(static, 200)
        for (long long q = 0; q < queries.rows; ++q) {
            if (q % 100 == 0 || q + 1 == queries.rows) {
                #pragma omp critical
                {
                    std::cout << "Processing query " << (q + 1) << "/" << queries.rows << std::endl;
                }
            }

            // openmp division of work is static or dynamic. By default, it is static. So each thread will get a chunk of queries to process.
            for (long long d = 0; d < db.rows; ++d) {
                scores[d] = dotProductRows(db, d, queries, q);
            }
            // std::cout << "Processed query " << (q + 1) << "/" << queries.rows << '\n';
            const std::vector<ScoreDoc> top = topKScoredQuickselect(scores, kTop);
            // const std::vector<ScoreDoc> top = std::vector<ScoreDoc>(queries.rows, std::make_pair(0.0F, 0));
            auto& topIndices = result.topIndicesByQuery[q];
            auto& topScores = result.topScoresByQuery[q];
            topIndices.reserve(top.size());
            topScores.reserve(top.size());
            for (const ScoreDoc& item : top) {
                topScores.push_back(item.first);
                topIndices.push_back(item.second);
            }
    
        }
    }

	result.elapsedMs = searchClock.elapsedMs();
	return result;
}

InvertedIndex buildSimpleInvertedIndex(const CSRMatrix &db, int numThreads) {
    InvertedIndex index;
    index.terms = db.cols;
    index.termIndptr.assign(db.cols + 1, 0);

    // Count the number of postings for each term, to "relative" offset the "next" term offset in the indptr array.
    // For now termIndptr[i+1] represents the distance to termIndptr[i].
    for (int term : db.indices) {
        if (term < 0 || term >= index.terms) {
            throw std::runtime_error("Term index out of bounds in database: " + std::to_string(term));
        }
        index.termIndptr[term + 1]++;
    }

    // From left to right, Offset each termIndptr by summing the previous to calculate an absolute offset.
    for (int i = 1; i < index.termIndptr.size(); i++) {
        index.termIndptr[i] += index.termIndptr[i - 1];
    }

    index.docIds.assign(db.indices.size(), 0);
    index.docValues.assign(db.data.size(), 0.0F);
    // nextWrite is a copy of termIndptr, it will change to be able to provide next positions to write.
    std::vector<long long> nextWrite = index.termIndptr;
    for (long long row = 0; row < db.rows; row++) {
        const long long rowStart = db.indptr[row];
        const long long rowEnd = db.indptr[row + 1];
        for (long long p = rowStart; p < rowEnd; p++) {
            const int term = db.indices[p];
            const float value = db.data[p];
            const long long writePos = nextWrite[term]++; // get next and increment for to point to next
            index.docIds[writePos] = (int)(row);
            index.docValues[writePos] = value;
        }
    }

    return index;
}

std::vector<ScoreDoc> topKScoredFromTouchedQuickselect(
    const std::vector<int>& touchedDocs,
    const std::vector<float>& scores,
    std::size_t k) {

    std::vector<ScoreDoc> scored;
    // first all scores from touchedDocs
    scored.reserve(touchedDocs.size());
    for (int docId : touchedDocs) {
        scored.emplace_back(scores[docId], docId);
    }

    k = std::min(k, scored.size());
    if (k == 0) {
        return {};
    }

    auto scoredPtrDiff = static_cast<std::ptrdiff_t>(k);

    // if more than k scores, partition putting the top k in the first k postions,
    if (k < scored.size()) {
        std::nth_element(scored.begin(), scored.begin() + scoredPtrDiff, scored.end(),
                        [](const ScoreDoc& lhs, const ScoreDoc& rhs) {
                            // if same scores, order by docId ascending
                            if (lhs.first == rhs.first) {
                                return lhs.second < rhs.second;
                            }
                            // else order by score descending
                            return lhs.first > rhs.first;
                        });
    }

    // sort the top k scores in descending order, and if same score, by docId ascending
    std::sort(scored.begin(), scored.begin() + scoredPtrDiff,
              [](const ScoreDoc& lhs, const ScoreDoc& rhs) {
                  if (lhs.first == rhs.first) {
                      return lhs.second < rhs.second;
                  }
                  return lhs.first > rhs.first;
              });
    
    scored.resize(k);
    return scored;
}

void searchTopKInvertedIndex(const InvertedIndex &index, const CSRMatrix &db, const CSRMatrix &queries, std::size_t kTop, int numThreads, RetrievalResult& result) {
    omp_set_num_threads(numThreads);

    #pragma omp parallel
    {
        std::vector<float> scores(db.rows, 0.0);
        std::vector<unsigned int> seenStamp(db.rows, 0);
        std::vector<int> touchedDocs;
        touchedDocs.reserve(4096); // at least 4k touched (it goes beyond, about 90% which is 57k for the fiqa dataset)
        unsigned int stamp = 1;

        #pragma omp for schedule(dynamic, 200)
        for (long long q = 0; q < queries.rows; q++) {
            touchedDocs.clear();
            long long postingsVisitedForQuery = 0;
            const long long qStart = queries.indptr[q];
            const long long qEnd = queries.indptr[q + 1];

            for (long long tInd = qStart; tInd < qEnd; tInd++) {
                const int term = queries.indices[tInd];
                if (term < 0 || term >= index.terms) {
                    throw std::runtime_error("Term index out of bounds in query: " + std::to_string(term));
                }

                const float qVal = queries.data[tInd];

                const long long postStart = index.termIndptr[term];
                const long long postEnd = index.termIndptr[term + 1];
                postingsVisitedForQuery += (postEnd - postStart);

                for (long long pInd = postStart; pInd < postEnd; pInd++) {
                    const int docId = index.docIds[pInd];
                    if (seenStamp[docId] != stamp) {
                        seenStamp[docId] = stamp;
                        scores[docId] = 0.0F;
                        touchedDocs.push_back(docId);
                    }
                    scores[docId] += qVal * index.docValues[pInd];
                }
            }

            const std::vector<ScoreDoc> top = topKScoredFromTouchedQuickselect(touchedDocs, scores, kTop);
            // make aliases for convenience
            auto& topIndices = result.topIndicesByQuery[q];
            auto& topScores = result.topScoresByQuery[q];
            topIndices.reserve(top.size());
            topScores.reserve(top.size());

            // push top doc/vector rows indices and scores to the result
            for (const ScoreDoc& item : top) {
                topScores.push_back(item.first);
                topIndices.push_back(item.second);
            }

            #pragma omp critical
            {
                result.totalPostingsVisited += postingsVisitedForQuery;
                result.totalTouchedDocs += touchedDocs.size();
                result.maxTouchedDocs = std::max(result.maxTouchedDocs, (long long)touchedDocs.size());
                if ((q + 1) % 100 == 0 || q + 1 == queries.rows) {
                    std::cout << "Processed query " << (q + 1) << "/" << queries.rows
                            << " (touched docs: " << touchedDocs.size()
                            << ", postings visited: " << postingsVisitedForQuery << ")" << std::endl;
                }
            }

            // Update stamp for next query docs reset
            stamp++;
            if (stamp == 0U) {
                std::fill(seenStamp.begin(), seenStamp.end(), 0U);
                stamp = 1U;
            }
        }
    } // End of OpenMP Block
}

RetrievalResult runInvertedIndexRetrieval(const CSRMatrix& db, const CSRMatrix& queries, std::size_t kTop, int numThreads) {
    RetrievalResult result;
    result.topIndicesByQuery.resize(queries.rows);
    result.topScoresByQuery.resize(queries.rows);
    result.queryCount = queries.rows;

    Clock prepClock;
    prepClock.start();
    // build inverted index
    auto invertedIndex = buildSimpleInvertedIndex(db, numThreads);
    result.prepElapsedMs = prepClock.elapsedMs();

    // do search
    Clock searchClock;
    searchClock.start();
    searchTopKInvertedIndex(invertedIndex, db, queries, kTop, numThreads, result);
    result.elapsedMs = searchClock.elapsedMs();

    return result;
}

RetrievalResult runRetrieval(const CSRMatrix &db, const CSRMatrix &queries, int kTop, RetrievalStrategy strategy, int numThreads) {
    RetrievalResult results;
    switch (strategy)
    {
    case RetrievalStrategy::BRUTE_FORCE:
        results = runBruteForceRetrieval(db, queries, kTop, numThreads);
        break;
    case RetrievalStrategy::INVERTED_INDEX:
        results = runInvertedIndexRetrieval(db, queries, kTop, numThreads);
        break;
    default:
        throw std::runtime_error("Unknown retrieval strategy.");
    }

    return results;
}

void printTopKResults(RetrievalResult& retrieval, std::size_t kTop) {
    // print the top-k results for the first 5 queries in a table form.
    // columns are queries and rows are ranks
    for (int i = 0; i < kTop; ++i) {
        if (i == 0) {
            std::cout << "Rank\t";
            for (int j = 0; j < 5; ++j) {
                std::cout << "Query " << j + 1 << "\t\t\t";
            }
            std::cout << std::endl;
        }
        for (int j = 0; j < 5; ++j) {
            if (j == 0) {
                std::cout << (i + 1) << "\t";
            }
            if (j < retrieval.topIndicesByQuery.size() && i < retrieval.topIndicesByQuery[j].size()) {
                // std::string scoreStr = sprintf("%.4f", retrieval.topScoresByQuery[j][i]);

                std::cout << std::fixed << std::setprecision(4) << retrieval.topScoresByQuery[j][i] << "," << retrieval.topIndicesByQuery[j][i] << "\t\t";
            } else {
                std::cout << "N/A\t";
            }
        }
        std::cout << std::endl;
    }
}

void storeResults(const std::string& dst,
			  const std::string& algo,
			  const std::string& dataset,
			  const std::string& task,
			  const RetrievalResult& retrieval,
			  std::size_t kTop,
			  double buildTimeSeconds,
			  double queryTimeSeconds,
			  const std::string& params) {
	if (retrieval.topIndicesByQuery.size() != retrieval.topScoresByQuery.size()) {
		std::cerr << "Error: Retrieval result has mismatched sizes for indices and scores\n";
		// print sizes
		std::cerr << "topIndicesByQuery size: " << retrieval.topIndicesByQuery.size() << '\n';
		std::cerr << "topScoresByQuery size: " << retrieval.topScoresByQuery.size() << '\n';
		throw std::runtime_error("Invalid retrieval result: index and score matrix sizes differ");
	}

	std::filesystem::path outPath(dst);
	if (!outPath.parent_path().empty()) {
		std::filesystem::create_directories(outPath.parent_path());
	}

	const std::size_t nQueries = retrieval.topIndicesByQuery.size();
	std::vector<int> flatKnns(nQueries * kTop, 0);
	std::vector<float> flatDists(nQueries * kTop, 0.0F);

	for (std::size_t q = 0; q < nQueries; ++q) {
		const auto& indices = retrieval.topIndicesByQuery[q];
		const auto& dists = retrieval.topScoresByQuery[q];
		const std::size_t available = std::min({kTop, indices.size(), dists.size()});
		for (std::size_t i = 0; i < available; ++i) {
			// Ground truth and baseline outputs are 1-based.
			flatKnns[q * kTop + i] = indices[i] + 1;
			flatDists[q * kTop + i] = dists[i];
		}
	}

	H5::H5File file(dst, H5F_ACC_TRUNC);
	writeStringAttribute(file, "algo", algo);
	writeStringAttribute(file, "dataset", dataset);
	writeStringAttribute(file, "task", task);
	writeDoubleAttribute(file, "buildtime", buildTimeSeconds);
	writeDoubleAttribute(file, "querytime", queryTimeSeconds);
	writeStringAttribute(file, "params", params);

	hsize_t dims[2] = {static_cast<hsize_t>(nQueries), static_cast<hsize_t>(kTop)};
	H5::DataSpace matrixSpace(2, dims);

	H5::DataSet knnsDs = file.createDataSet("knns", H5::PredType::NATIVE_INT, matrixSpace);
	if (!flatKnns.empty()) {
		knnsDs.write(flatKnns.data(), H5::PredType::NATIVE_INT);
	}

	H5::DataSet distsDs = file.createDataSet("dists", H5::PredType::NATIVE_FLOAT, matrixSpace);
	if (!flatDists.empty()) {
		distsDs.write(flatDists.data(), H5::PredType::NATIVE_FLOAT);
	}
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

    int kTop = std::stoi(argsMap["kTop"]);
    std::string datasetName = argsMap["dataset"];
    std::string taskName = argsMap["task"];
    std::string filePath = getInputFilePath(argsMap["inputFolder"], datasetName, taskName);
    std::string outputPath = argsMap["outputFolder"] + "/" + taskName + "_" + datasetName + "_k=" + std::to_string(kTop) + argsMap["strategy"] +  ".h5";
    RetrievalStrategy strategy = parseStrategy(argsMap["strategy"]);
    int numThreads = std::stoi(argsMap["threads"]);
    if (numThreads == -1) {
        numThreads = omp_get_max_threads();
    } else if (numThreads == 0 || numThreads < -1) {
        std::cerr << "Invalid number of threads specified: " << numThreads << ". Must be -1 or a positive integer." << std::endl;
        exit(1);
    }

    std::cout << "Using dataset: " << datasetName << std::endl;
    std::cout << "Using task: " << taskName << std::endl;

    std::string params = "k=" + std::to_string(kTop) + ", threads=" + std::to_string(numThreads);

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

    auto durationMs = clk.elapsedMs(); 

    std::cout << "Train Matrix Shape: (" << db.rows << "," << db.cols << ")" << std::endl;
    std::cout << "Queries Matrix Shape: (" << queries.rows << "," << queries.cols << ")" << std::endl;
    std::cout << "Time to load csr matrices: " << durationMs << " ms." << std::endl;

    if (db.cols != queries.cols) {
        throw std::runtime_error("Train and queries column dimension must match.");
    }

    std::cout << "Running retrieval with strategy: " << argsMap["strategy"] << std::endl;
    std::cout << "Configured to use " << numThreads << " threads for parallel processing." << std::endl;
    // ------------------------------------------------------------------------------
    // Process Queries
    // ------------------------------------------------------------------------------
    // RetrievalResult retrieval;
    auto retrieval = runRetrieval(db, queries, kTop, strategy, numThreads);

    // ------------------------------------------------------------------------------
    // Print Results
    // ------------------------------------------------------------------------------

    // Print Timing Information
    if (strategy == RetrievalStrategy::BRUTE_FORCE) {
        std::cout << "Brute-force retrieval completed." << std::endl;
        std::cout << "Brute-force search time: " << retrieval.elapsedMs << " ms" << '\n';
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

    // Print Top-K Results for the first query
    printTopKResults(retrieval, kTop);



    // ------------------------------------------------------------------------------
    // Store Results
    // ------------------------------------------------------------------------------
    storeResults(outputPath,
			   argsMap["strategy"],
			   datasetName,
			   taskName,
			   retrieval,
			   kTop,
			   static_cast<double>(retrieval.prepElapsedMs) / 1000.0,
			   static_cast<double>(retrieval.elapsedMs) / 1000.0,
			   params);
    std::cout << "Stored HDF5 results at: " << outputPath << '\n';


    return 0;
}