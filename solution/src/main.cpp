// Load a sparse CSR matrix from an HDF5 file using the conventional
// (indptr, indices, data, shape attr) layout.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>
#include <filesystem>
#include <string>
#include <vector>
#include <omp.h>
#include <fstream>
#include "utils.hpp"

#include <H5Cpp.h>

// Compressed Sparse Row Matrix Representation
struct CsrMatrix {
    // indptr store the entries offset for each row.
    //  So indptr[1]=200 means first column entry of row_1 is at position 200 for indices and data.
	std::vector<long long> indptr;
    // indices store the column index of each entry.
    //  So indices[200]=32 means that entry 200 (corresponds to row_1) is for column 32
	 std::vector<int> indices;
    // data store the value of each entry
    //  So data[200]=60 means that entry 200 (row_1 col 32) has value 60
	std::vector<float> data;

	long long rows = 0;
	long long cols = 0;
};

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

enum class RetrievalStrategy {
	BruteForce,
	InvertedIndex
};

RetrievalStrategy parseStrategy(const std::string& name) {
	if (name == "brute" || name == "bruteforce") {
		return RetrievalStrategy::BruteForce;
	}
	if (name == "inverted" || name == "inverted-index" || name == "index") {
		return RetrievalStrategy::InvertedIndex;
	}
	throw std::runtime_error("Unknown strategy '" + name + "'. Use: brute, inverted");
}

struct InvertedIndex {
	std::vector<long long> termIndptr;
	std::vector<int> docIds;
	std::vector<float> docValues;
	long long terms = 0;
};

using ScoreDoc = std::pair<float, int>;

struct MinScoreComparator {
	bool operator()(const ScoreDoc& lhs, const ScoreDoc& rhs) const {
		if (lhs.first == rhs.first) {
			return lhs.second > rhs.second;
		}
		return lhs.first > rhs.first;
	}
};

float dotProductRows(const CsrMatrix& a, long long aRow, const CsrMatrix& b, long long bRow) {
	const long long aStart = a.indptr[static_cast<std::size_t>(aRow)];
	const long long aEnd = a.indptr[static_cast<std::size_t>(aRow + 1)];
	const long long bStart = b.indptr[static_cast<std::size_t>(bRow)];
	const long long bEnd = b.indptr[static_cast<std::size_t>(bRow + 1)];

	long long i = aStart;
	long long j = bStart;
	float sum = 0.0F;

	while (i < aEnd && j < bEnd) {
		const int ai = a.indices[static_cast<std::size_t>(i)];
		const int bj = b.indices[static_cast<std::size_t>(j)];
		if (ai == bj) {
			sum += a.data[static_cast<std::size_t>(i)] * b.data[static_cast<std::size_t>(j)];
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

std::vector<int> topKIndicesQuickselect(const std::vector<float>& scores, std::size_t k) {
	std::vector<std::pair<float, int>> scored;
	scored.reserve(scores.size());
	for (std::size_t i = 0; i < scores.size(); ++i) {
		scored.emplace_back(scores[i], static_cast<int>(i));
	}

	k = std::min(k, scored.size());
	if (k == 0) {
		return {};
	}

	if (k < scored.size()) {
		std::nth_element(scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(k), scored.end(),
						 [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });
	}
	std::sort(scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(k),
		  [](const auto& lhs, const auto& rhs) {
			  if (lhs.first == rhs.first) {
				  return lhs.second < rhs.second;
			  }
			  return lhs.first > rhs.first;
		  });

	std::vector<int> topIndices;
	topIndices.reserve(k);
	for (std::size_t i = 0; i < k; ++i) {
		topIndices.push_back(scored[i].second);
	}

	return topIndices;
}

std::vector<int> topKFromScoredDocsQuickselect(const std::vector<int>& touchedDocs,
								const std::vector<float>& scores,
								std::size_t k) {
	std::vector<std::pair<float, int>> scored;
	scored.reserve(touchedDocs.size());
	for (int docId : touchedDocs) {
		scored.emplace_back(scores[static_cast<std::size_t>(docId)], docId);
	}

	k = std::min(k, scored.size());
	if (k == 0) {
		return {};
	}

	if (k < scored.size()) {
		std::nth_element(scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(k), scored.end(),
						 [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });
	}
	std::sort(scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(k),
		  [](const auto& lhs, const auto& rhs) {
			  if (lhs.first == rhs.first) {
				  return lhs.second < rhs.second;
			  }
			  return lhs.first > rhs.first;
		  });

	std::vector<int> topIndices;
	topIndices.reserve(k);
	for (std::size_t i = 0; i < k; ++i) {
		topIndices.push_back(scored[i].second);
	}
	return topIndices;
}

std::vector<ScoreDoc> topKScoredFromTouchedDocsQuickselect(const std::vector<int>& touchedDocs,
									   const std::vector<float>& scores,
									   std::size_t k) {
	std::vector<ScoreDoc> scored;
	scored.reserve(touchedDocs.size());
	for (int docId : touchedDocs) {
		scored.emplace_back(scores[static_cast<std::size_t>(docId)], docId);
	}

	k = std::min(k, scored.size());
	if (k == 0) {
		return {};
	}

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

InvertedIndex buildInvertedIndex(const CsrMatrix& db) {
	InvertedIndex index;
	index.terms = db.cols;
	index.termIndptr.assign(static_cast<std::size_t>(db.cols + 1), 0);

	for (int term : db.indices) {
		if (term < 0 || term >= db.cols) {
			throw std::runtime_error("Invalid term index in train matrix");
		}
		++index.termIndptr[static_cast<std::size_t>(term + 1)];
	}

	for (std::size_t t = 1; t < index.termIndptr.size(); ++t) {
		index.termIndptr[t] += index.termIndptr[t - 1];
	}

	index.docIds.assign(db.indices.size(), 0);
	index.docValues.assign(db.data.size(), 0.0F);
	std::vector<long long> nextWrite = index.termIndptr;

	for (long long row = 0; row < db.rows; ++row) {
		const long long start = db.indptr[static_cast<std::size_t>(row)];
		const long long end = db.indptr[static_cast<std::size_t>(row + 1)];
		for (long long p = start; p < end; ++p) {
			const int term = db.indices[static_cast<std::size_t>(p)];
			const long long w = nextWrite[static_cast<std::size_t>(term)]++;
			index.docIds[static_cast<std::size_t>(w)] = static_cast<int>(row);
			index.docValues[static_cast<std::size_t>(w)] = db.data[static_cast<std::size_t>(p)];
		}
	}

	return index;
}

inline void  accumulatePostingRange(const InvertedIndex& index,
					  long long postStart,
					  long long postEnd,
					  float qVal,
					  std::vector<float>& scores,
					  std::vector<unsigned int>& seenStamp,
					  unsigned int stamp,
					  std::vector<int>& touchedDocs) {
	const int* docIds = index.docIds.data();
	const float* docValues = index.docValues.data();
	float* scorePtr = scores.data();
	unsigned int* seenPtr = seenStamp.data();

	long long it = postStart;
	for (; it + 3 < postEnd; it += 4) {
		const int doc0 = docIds[it];
		if (seenPtr[doc0] != stamp) {
			seenPtr[doc0] = stamp;
			scorePtr[doc0] = 0.0F;
			touchedDocs.push_back(doc0);
		}
		scorePtr[doc0] += qVal * docValues[it];

		const int doc1 = docIds[it + 1];
		if (seenPtr[doc1] != stamp) {
			seenPtr[doc1] = stamp;
			scorePtr[doc1] = 0.0F;
			touchedDocs.push_back(doc1);
		}
		scorePtr[doc1] += qVal * docValues[it + 1];

		const int doc2 = docIds[it + 2];
		if (seenPtr[doc2] != stamp) {
			seenPtr[doc2] = stamp;
			scorePtr[doc2] = 0.0F;
			touchedDocs.push_back(doc2);
		}
		scorePtr[doc2] += qVal * docValues[it + 2];

		const int doc3 = docIds[it + 3];
		if (seenPtr[doc3] != stamp) {
			seenPtr[doc3] = stamp;
			scorePtr[doc3] = 0.0F;
			touchedDocs.push_back(doc3);
		}
		scorePtr[doc3] += qVal * docValues[it + 3];
	}

	for (; it < postEnd; ++it) {
		const int docId = docIds[it];
		if (seenPtr[docId] != stamp) {
			seenPtr[docId] = stamp;
			scorePtr[docId] = 0.0F;
			touchedDocs.push_back(docId);
		}
		scorePtr[docId] += qVal * docValues[it];
	}
}

RetrievalResult runBruteForceRetrieval(const CsrMatrix& db, const CsrMatrix& queries, std::size_t kTop) {
	Clock searchClock;
	searchClock.start();

	RetrievalResult result;
	result.topIndicesByQuery.resize(static_cast<std::size_t>(queries.rows));
	result.topScoresByQuery.resize(static_cast<std::size_t>(queries.rows));

	std::vector<float> scores(static_cast<std::size_t>(db.rows), 0.0F);
	for (long long q = 0; q < queries.rows; ++q) {
		for (long long d = 0; d < db.rows; ++d) {
			scores[static_cast<std::size_t>(d)] = dotProductRows(db, d, queries, q);
		}
		result.topIndicesByQuery[static_cast<std::size_t>(q)] = topKIndicesQuickselect(scores, kTop);

		if ((q + 1) % 100 == 0 || q + 1 == queries.rows) {
			std::cout << "Processed query " << (q + 1) << "/" << queries.rows << '\n';
		}
	}

	result.elapsedMs = searchClock.elapsedMs();
	return result;
}

RetrievalResult runInvertedIndexRetrieval(const CsrMatrix& db, const CsrMatrix& queries, std::size_t kTop) {
	RetrievalResult result;
	result.topIndicesByQuery.resize(static_cast<std::size_t>(queries.rows));
	result.topScoresByQuery.resize(static_cast<std::size_t>(queries.rows));
	result.queryCount = queries.rows;

	Clock prepClock;
	prepClock.start();
	const InvertedIndex index = buildInvertedIndex(db);
	result.prepElapsedMs = prepClock.elapsedMs();

	Clock searchClock;
	searchClock.start();

	#pragma omp parallel
	{
		// 1. These variables MUST be declared inside so each thread gets its own private copy
		std::vector<float> scores(static_cast<std::size_t>(db.rows), 0.0F);
		std::vector<unsigned int> seenStamp(static_cast<std::size_t>(db.rows), 0U);
		std::vector<int> touchedDocs;
		touchedDocs.reserve(4096);
		unsigned int stamp = 1U;

		// 2. Split queries into batches of 200 per thread
		#pragma omp for schedule(dynamic, 200)
		for (long long q = 0; q < queries.rows; ++q) {
			touchedDocs.clear();
			long long postingsVisitedForQuery = 0;
			const long long qStart = queries.indptr[static_cast<std::size_t>(q)];
			const long long qEnd = queries.indptr[static_cast<std::size_t>(q + 1)];

			for (long long p = qStart; p < qEnd; ++p) {
				const int term = queries.indices[static_cast<std::size_t>(p)];
				if (term < 0 || term >= index.terms) {
					continue;
				}

				const float qVal = queries.data[static_cast<std::size_t>(p)];
				const long long postStart = index.termIndptr[static_cast<std::size_t>(term)];
				const long long postEnd = index.termIndptr[static_cast<std::size_t>(term + 1)];
				postingsVisitedForQuery += (postEnd - postStart);

				for (long long it = postStart; it < postEnd; ++it) {
					const int docId = index.docIds[static_cast<std::size_t>(it)];
					if (seenStamp[static_cast<std::size_t>(docId)] != stamp) {
						seenStamp[static_cast<std::size_t>(docId)] = stamp;
						scores[static_cast<std::size_t>(docId)] = 0.0F;
						touchedDocs.push_back(docId);
					}
					scores[static_cast<std::size_t>(docId)] += qVal * index.docValues[static_cast<std::size_t>(it)];
				}
			}

			// result.topIndicesByQuery[static_cast<std::size_t>(q)] = topKFromScoredDocsQuickselect(touchedDocs, scores, kTop);

			const std::vector<ScoreDoc> top = topKScoredFromTouchedDocsQuickselect(touchedDocs, scores, kTop);
			auto& topIndices = result.topIndicesByQuery[static_cast<std::size_t>(q)];
			auto& topScores = result.topScoresByQuery[static_cast<std::size_t>(q)];
			topIndices.reserve(top.size());
			topScores.reserve(top.size());
			for (const ScoreDoc& item : top) {
				topScores.push_back(item.first);
				topIndices.push_back(item.second);
			}
			
			// 3. Lock the statistics to accumulate them safely
			#pragma omp critical
			{
				result.totalPostingsVisited += postingsVisitedForQuery;
				result.totalTouchedDocs += static_cast<long long>(touchedDocs.size());
				result.maxTouchedDocs = std::max(result.maxTouchedDocs, static_cast<long long>(touchedDocs.size()));
				if ((q + 1) % 100 == 0 || q + 1 == queries.rows) {
					std::cout << "Processed query " << (q + 1) << "/" << queries.rows
					<< " (touched docs: " << touchedDocs.size()
					<< ", postings visited: " << postingsVisitedForQuery << ")" << '\n';
				}
			}
			++stamp;
			if (stamp == 0U) {
				std::fill(seenStamp.begin(), seenStamp.end(), 0U);
				stamp = 1U;
			}
		}
	} // End of OpenMP Block

	result.elapsedMs = searchClock.elapsedMs();
	return result;
}

RetrievalResult runRetrieval(const CsrMatrix& db, const CsrMatrix& queries, std::size_t kTop,
				 RetrievalStrategy strategy) {
	switch (strategy) {
		case RetrievalStrategy::BruteForce:
			return runBruteForceRetrieval(db, queries, kTop);
		case RetrievalStrategy::InvertedIndex:
			return runInvertedIndexRetrieval(db, queries, kTop);
		default:
			throw std::runtime_error("Unsupported retrieval strategy");
	}
}

template <typename T>
std::vector<T> read1DDataset(const H5::Group& group, const std::string& name, const H5::PredType& type) {
	H5::DataSet dataset = group.openDataSet(name);
	H5::DataSpace space = dataset.getSpace();

	if (space.getSimpleExtentNdims() != 1) {
		throw std::runtime_error("Dataset '" + name + "' is not 1D");
	}

	hsize_t dims = 0;
	space.getSimpleExtentDims(&dims, nullptr);

	std::vector<T> out(static_cast<std::size_t>(dims));
	if (!out.empty()) {
		dataset.read(out.data(), type);
	}
	return out;
}

std::array<long long, 2> readShapeAttribute(const H5::Group& group) {
	H5::Attribute attr = group.openAttribute("shape");
	H5::DataSpace space = attr.getSpace();
	if (space.getSimpleExtentNdims() != 1) {
		throw std::runtime_error("Attribute 'shape' is not 1D");
	}

	hsize_t dims = 0;
	space.getSimpleExtentDims(&dims, nullptr);
	if (dims != 2) {
		throw std::runtime_error("Attribute 'shape' must have exactly 2 values [rows, cols]");
	}

	std::array<long long, 2> shape{};
	attr.read(H5::PredType::NATIVE_LLONG, shape.data());
	return shape;
}

CsrMatrix loadSparseMatrix(const H5::Group& group) {
	CsrMatrix mat;
	mat.indptr = read1DDataset<long long>(group, "indptr", H5::PredType::NATIVE_LLONG);
	mat.indices = read1DDataset<int>(group, "indices", H5::PredType::NATIVE_INT);
	mat.data = read1DDataset<float>(group, "data", H5::PredType::NATIVE_FLOAT);

	const auto shape = readShapeAttribute(group);
	mat.rows = shape[0];
	mat.cols = shape[1];

	if (mat.indptr.size() != static_cast<std::size_t>(mat.rows + 1)) {
		throw std::runtime_error("Invalid CSR: indptr size must be rows + 1");
	}
	if (mat.indices.size() != mat.data.size()) {
		throw std::runtime_error("Invalid CSR: indices and data sizes differ");
	}

	return mat;
}

bool isIntegerString(const std::string& value) {
	if (value.empty()) {
		return false;
	}
	std::size_t start = 0;
	if (value[0] == '+' || value[0] == '-') {
		start = 1;
	}
	if (start >= value.size()) {
		return false;
	}
	for (std::size_t i = start; i < value.size(); ++i) {
		if (value[i] < '0' || value[i] > '9') {
			return false;
		}
	}
	return true;
}

void writeStringAttribute(H5::H5File& file, const std::string& name, const std::string& value) {
	const H5::StrType strType(H5::PredType::C_S1, H5T_VARIABLE);
	const H5::DataSpace scalarSpace(H5S_SCALAR);
	H5::Attribute attr = file.createAttribute(name, strType, scalarSpace);
	attr.write(strType, value);
}

void writeDoubleAttribute(H5::H5File& file, const std::string& name, double value) {
	const H5::DataSpace scalarSpace(H5S_SCALAR);
	H5::Attribute attr = file.createAttribute(name, H5::PredType::NATIVE_DOUBLE, scalarSpace);
	attr.write(H5::PredType::NATIVE_DOUBLE, &value);
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


int main(int argc, char** argv) {
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

	try {
		Clock loadClock;
		loadClock.start();
		H5::H5File file(filePath, H5F_ACC_RDONLY);
		H5::Group train = file.openGroup("train");
		CsrMatrix db = loadSparseMatrix(train);
		H5::Group queryGroup = file.openGroup("otest/queries");
		CsrMatrix queries = loadSparseMatrix(queryGroup);
		const long long loadMs = loadClock.elapsedMs();

		std::cout << "Loaded file: " << filePath << '\n';
		std::cout << "Strategy: "
				  << ((strategy == RetrievalStrategy::InvertedIndex)
					  ? "InvertedIndex"
					  : "BruteForce")
				  << '\n';
		
		std::cout << "CSR load time (train + queries): " << loadMs << " ms" << '\n';
		std::cout << "Train shape: [" << db.rows << ", " << db.cols << "]" << '\n';
		std::cout << "Queries shape: [" << queries.rows << ", " << queries.cols << "]" << '\n';

		if (db.cols != queries.cols) {
			throw std::runtime_error("Dimension mismatch between train and queries");
		}

		const RetrievalResult retrieval = runRetrieval(db, queries, kTop, strategy);
		if (strategy == RetrievalStrategy::InvertedIndex) {
			std::cout << "Inverted index build time: " << retrieval.prepElapsedMs << " ms" << '\n';
			std::cout << "Inverted-index search time: " << retrieval.elapsedMs << " ms" << '\n';
			if (retrieval.queryCount > 0) {
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
			}
		} else if (strategy == RetrievalStrategy::BruteForce) {
			std::cout << "Brute-force search time: " << retrieval.elapsedMs << " ms" << '\n';
		}

		std::string algo = "cpp_sparse_inverted";
		std::string params = "index=(cpp_sparse),query=(default)";
		if (strategy == RetrievalStrategy::BruteForce) {
			algo = "cpp_sparse_bruteforce";
			// params = "index=(none),query=(bruteforce)";
		}
		params = "TBD";

		storeResults(outputPath,
			   algo,
			   datasetName,
			   taskName,
			   retrieval,
			   kTop,
			   static_cast<double>(retrieval.prepElapsedMs) / 1000.0,
			   static_cast<double>(retrieval.elapsedMs) / 1000.0,
			   params);
		std::cout << "Stored HDF5 results at: " << outputPath << '\n';

		if (!retrieval.topIndicesByQuery.empty()) {
			std::cout << "First query top-" << kTop << " indices:";
			for (int idx : retrieval.topIndicesByQuery.front()) {
				std::cout << ' ' << idx;
			}
			std::cout << '\n';
		}

		const bool rowsOk = (db.rows >= 55000 && db.rows <= 60000);
		const bool colsOk = (db.cols >= 30000);
		const bool queriesOk = (queries.rows >= 3000);
		std::cout << "Row check (~57k): " << (rowsOk ? "PASS" : "FAIL") << '\n';
		std::cout << "Component check (>=30k): " << (colsOk ? "PASS" : "FAIL") << '\n';
		std::cout << "Query count check (~3k): " << (queriesOk ? "PASS" : "FAIL") << '\n';

		return (rowsOk && colsOk && queriesOk) ? EXIT_SUCCESS : EXIT_FAILURE;
	} catch (const H5::Exception& e) {
		std::cerr << "HDF5 error: " << e.getDetailMsg() << '\n';
	} catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << '\n';
	}

	return EXIT_FAILURE;
}