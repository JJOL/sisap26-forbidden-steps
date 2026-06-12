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
	InvertedIndex,
	Hnsw
};

struct HnswParams {
	int M = 32;
	int efConstruction = 200;
	int efSearch = 120;
};

RetrievalStrategy parseStrategy(const std::string& name) {
	if (name == "brute" || name == "bruteforce") {
		return RetrievalStrategy::BruteForce;
	}
	if (name == "inverted" || name == "inverted-index" || name == "index") {
		return RetrievalStrategy::InvertedIndex;
	}
	if (name == "hnsw") {
		return RetrievalStrategy::Hnsw;
	}
	throw std::runtime_error("Unknown strategy '" + name + "'. Use: brute, inverted, or hnsw");
}

struct InvertedIndex {
	std::vector<long long> termIndptr;
	std::vector<int> docIds;
	std::vector<float> docValues;
	long long terms = 0;
};

struct HnswIndex {
	std::vector<int> levels;
	std::vector<std::vector<std::vector<int>>> neighborsByNode;
	int entryPoint = -1;
	int maxLevel = -1;
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

struct Clock {
	using time_point = std::chrono::steady_clock::time_point;
	time_point start_time{};

	void start() {
		start_time = std::chrono::steady_clock::now();
	}

	long long elapsedMs() const {
		const auto end_time = std::chrono::steady_clock::now();
		return std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
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


int sampleHnswLevel(std::mt19937& rng, int M) {
	if (M <= 1) {
		return 0;
	}
	std::uniform_real_distribution<float> dist(0.0F, 1.0F);
	const float u = std::max(dist(rng), 1e-12F);
	const float levelMult = 1.0F / std::log(static_cast<float>(M));
	return static_cast<int>(-std::log(u) * levelMult);
}

void pruneNeighborsForNode(const CsrMatrix& db,
				   HnswIndex& index,
				   int node,
				   int level,
				   int maxDegree) {
	auto& adj = index.neighborsByNode[static_cast<std::size_t>(node)][static_cast<std::size_t>(level)];
	if (adj.size() <= static_cast<std::size_t>(maxDegree)) {
		return;
	}

	std::vector<ScoreDoc> scored;
	scored.reserve(adj.size());
	for (int nb : adj) {
		const float score = dotProductRows(db,
						   static_cast<long long>(node),
						   db,
						   static_cast<long long>(nb));
		scored.emplace_back(score, nb);
	}

	std::sort(scored.begin(), scored.end(), [](const ScoreDoc& lhs, const ScoreDoc& rhs) {
		if (lhs.first == rhs.first) {
			return lhs.second < rhs.second;
		}
		return lhs.first > rhs.first;
	});

	adj.clear();
	adj.reserve(static_cast<std::size_t>(maxDegree));
	for (int i = 0; i < maxDegree && i < static_cast<int>(scored.size()); ++i) {
		adj.push_back(scored[static_cast<std::size_t>(i)].second);
	}
}

void addBidirectionalEdge(const CsrMatrix& db,
			 HnswIndex& index,
			 int src,
			 int dst,
			 int level,
			 int maxDegree) {
	if (src == dst) {
		return;
	}

	auto& srcAdj = index.neighborsByNode[static_cast<std::size_t>(src)][static_cast<std::size_t>(level)];
	if (std::find(srcAdj.begin(), srcAdj.end(), dst) == srcAdj.end()) {
		srcAdj.push_back(dst);
	}
	pruneNeighborsForNode(db, index, src, level, maxDegree);

	auto& dstAdj = index.neighborsByNode[static_cast<std::size_t>(dst)][static_cast<std::size_t>(level)];
	if (std::find(dstAdj.begin(), dstAdj.end(), src) == dstAdj.end()) {
		dstAdj.push_back(src);
	}
	pruneNeighborsForNode(db, index, dst, level, maxDegree);
}

int greedySearchAtLevel(const CsrMatrix& db,
			   const CsrMatrix& queryMatrix,
			   long long queryRow,
			   const HnswIndex& index,
			   int entryPoint,
			   int level) {
	int current = entryPoint;
	float currentScore = dotProductRows(db,
					   static_cast<long long>(current),
					   queryMatrix,
					   queryRow);

	bool changed = true;
	while (changed) {
		changed = false;
		const auto& adj = index.neighborsByNode[static_cast<std::size_t>(current)][static_cast<std::size_t>(level)];
		for (int nb : adj) {
			const float s = dotProductRows(db,
						   static_cast<long long>(nb),
						   queryMatrix,
						   queryRow);
			if (s > currentScore) {
				current = nb;
				currentScore = s;
				changed = true;
			}
		}
	}

	return current;
}

std::vector<ScoreDoc> searchLayer(const CsrMatrix& db,
				  const CsrMatrix& queryMatrix,
				  long long queryRow,
				  const HnswIndex& index,
				  const std::vector<int>& entryPoints,
				  int level,
				  std::size_t ef,
				  std::vector<unsigned int>& visitedStamp,
				  unsigned int stamp) {
	std::priority_queue<ScoreDoc> candidateQueue;
	std::priority_queue<ScoreDoc, std::vector<ScoreDoc>, MinScoreComparator> topResults;

	for (int ep : entryPoints) {
		if (ep < 0) {
			continue;
		}
		if (visitedStamp[static_cast<std::size_t>(ep)] == stamp) {
			continue;
		}
		visitedStamp[static_cast<std::size_t>(ep)] = stamp;
		const float score = dotProductRows(db,
						   static_cast<long long>(ep),
						   queryMatrix,
						   queryRow);
		candidateQueue.emplace(score, ep);
		topResults.emplace(score, ep);
	}

	while (!candidateQueue.empty()) {
		const ScoreDoc current = candidateQueue.top();
		candidateQueue.pop();

		if (topResults.size() >= ef && current.first < topResults.top().first) {
			break;
		}

		const int node = current.second;
		const auto& adj = index.neighborsByNode[static_cast<std::size_t>(node)][static_cast<std::size_t>(level)];
		for (int nb : adj) {
			if (visitedStamp[static_cast<std::size_t>(nb)] == stamp) {
				continue;
			}
			visitedStamp[static_cast<std::size_t>(nb)] = stamp;

			const float score = dotProductRows(db,
						   static_cast<long long>(nb),
						   queryMatrix,
						   queryRow);

			if (topResults.size() < ef || score > topResults.top().first) {
				candidateQueue.emplace(score, nb);
				topResults.emplace(score, nb);
				if (topResults.size() > ef) {
					topResults.pop();
				}
			}
		}
	}

	std::vector<ScoreDoc> out;
	out.reserve(topResults.size());
	while (!topResults.empty()) {
		out.push_back(topResults.top());
		topResults.pop();
	}
	std::sort(out.begin(), out.end(), [](const ScoreDoc& lhs, const ScoreDoc& rhs) {
		if (lhs.first == rhs.first) {
			return lhs.second < rhs.second;
		}
		return lhs.first > rhs.first;
	});

	return out;
}

HnswIndex buildHnswIndex(const CsrMatrix& db, const HnswParams& params) {
	HnswIndex index;
	const std::size_t n = static_cast<std::size_t>(db.rows);
	index.levels.assign(n, 0);
	index.neighborsByNode.resize(n);

	Clock buildClock;
	buildClock.start();
	const std::size_t reportEvery = std::max<std::size_t>(1, n / 100);
	std::size_t nextReport = reportEvery;

	std::cout << "HNSW build started: 0/" << n
		  << " nodes (0.0%), elapsed: 0.00s, eta: n/a" << '\n';

	std::mt19937 rng(42U);
	std::vector<unsigned int> visitedStamp(n, 0U);
	unsigned int stamp = 1U;

	for (std::size_t i = 0; i < n; ++i) {
		const int newLevel = sampleHnswLevel(rng, params.M);
		index.levels[i] = newLevel;
		index.neighborsByNode[i].resize(static_cast<std::size_t>(newLevel + 1));

		if (index.entryPoint < 0) {
			index.entryPoint = static_cast<int>(i);
			index.maxLevel = newLevel;
			continue;
		}

		int ep = index.entryPoint;
		const int searchUpper = index.maxLevel;
		const int targetUpper = std::min(newLevel, index.maxLevel);

		for (int level = searchUpper; level > targetUpper; --level) {
			ep = greedySearchAtLevel(db, db, static_cast<long long>(i), index, ep, level);
		}

		for (int level = targetUpper; level >= 0; --level) {
			if (stamp == 0U) {
				std::fill(visitedStamp.begin(), visitedStamp.end(), 0U);
				stamp = 1U;
			}

			const std::vector<ScoreDoc> candidates =
				searchLayer(db,
						db,
						static_cast<long long>(i),
						index,
						std::vector<int>{ep},
						level,
						static_cast<std::size_t>(std::max(params.efConstruction, params.M)),
						visitedStamp,
						stamp);
			++stamp;

			const int maxDegree = (level == 0) ? (2 * params.M) : params.M;
			int linked = 0;
			for (const ScoreDoc& cand : candidates) {
				if (cand.second == static_cast<int>(i)) {
					continue;
				}
				addBidirectionalEdge(db,
						   index,
						   static_cast<int>(i),
						   cand.second,
						   level,
						   maxDegree);
				++linked;
				if (linked >= maxDegree) {
					break;
				}
			}

			if (!candidates.empty()) {
				ep = candidates.front().second;
			}
		}

		if (newLevel > index.maxLevel) {
			index.entryPoint = static_cast<int>(i);
			index.maxLevel = newLevel;
		}

		const std::size_t processed = i + 1;
		if (processed >= nextReport || processed == n) {
			const double elapsedSec = static_cast<double>(buildClock.elapsedMs()) / 1000.0;
			const double progressPct = (n > 0)
				? (100.0 * static_cast<double>(processed) / static_cast<double>(n))
				: 100.0;
			const double nodesPerSec = elapsedSec > 0.0
				? static_cast<double>(processed) / elapsedSec
				: 0.0;
			const std::size_t remaining = n - processed;
			const double etaSec = nodesPerSec > 0.0
				? static_cast<double>(remaining) / nodesPerSec
				: std::numeric_limits<double>::infinity();

			std::cout << std::fixed << std::setprecision(2)
				  << "HNSW build progress: " << processed << "/" << n
				  << " nodes (" << progressPct << "%), elapsed: " << elapsedSec << "s";
			if (std::isfinite(etaSec)) {
				std::cout << ", eta: " << etaSec << "s";
			} else {
				std::cout << ", eta: n/a";
			}
			std::cout << ", speed: " << nodesPerSec << " nodes/s"
				  << ", maxLevel: " << index.maxLevel << '\n';

			nextReport += reportEvery;
		}
	}

	std::cout << "HNSW build completed in " << buildClock.elapsedMs() << " ms" << '\n';

	return index;
}

RetrievalResult runHnswRetrieval(const CsrMatrix& db,
				const CsrMatrix& queries,
				std::size_t kTop,
				const HnswParams& params) {
	if (params.M <= 0) {
		throw std::runtime_error("HNSW parameter M must be > 0");
	}
	if (params.efConstruction <= 0 || params.efSearch <= 0) {
		throw std::runtime_error("HNSW efConstruction and efSearch must be > 0");
	}

	RetrievalResult result;
	result.topIndicesByQuery.resize(static_cast<std::size_t>(queries.rows));
	result.queryCount = queries.rows;

	Clock prepClock;
	prepClock.start();
	const HnswIndex index = buildHnswIndex(db, params);
	result.prepElapsedMs = prepClock.elapsedMs();

	Clock searchClock;
	searchClock.start();

	std::vector<unsigned int> visitedStamp(static_cast<std::size_t>(db.rows), 0U);
	unsigned int stamp = 1U;

	for (long long q = 0; q < queries.rows; ++q) {
		if (index.entryPoint < 0) {
			result.topIndicesByQuery[static_cast<std::size_t>(q)] = {};
			continue;
		}

		int ep = index.entryPoint;
		for (int level = index.maxLevel; level >= 1; --level) {
			ep = greedySearchAtLevel(db, queries, q, index, ep, level);
		}

		if (stamp == 0U) {
			std::fill(visitedStamp.begin(), visitedStamp.end(), 0U);
			stamp = 1U;
		}

		const std::vector<ScoreDoc> candidates =
			searchLayer(db,
					queries,
					q,
					index,
					std::vector<int>{ep},
					0,
					static_cast<std::size_t>(std::max(params.efSearch, static_cast<int>(kTop))),
					visitedStamp,
					stamp);
		++stamp;

		std::vector<int> top;
		top.reserve(kTop);
		for (std::size_t i = 0; i < candidates.size() && i < kTop; ++i) {
			top.push_back(candidates[i].second);
		}
		result.topIndicesByQuery[static_cast<std::size_t>(q)] = std::move(top);

		if ((q + 1) % 100 == 0 || q + 1 == queries.rows) {
			std::cout << "Processed query " << (q + 1) << "/" << queries.rows << '\n';
		}
	}

	result.elapsedMs = searchClock.elapsedMs();
	return result;
}

RetrievalResult runRetrieval(const CsrMatrix& db, const CsrMatrix& queries, std::size_t kTop,
				 RetrievalStrategy strategy,
				 const HnswParams& hnswParams) {
	switch (strategy) {
		case RetrievalStrategy::BruteForce:
			return runBruteForceRetrieval(db, queries, kTop);
		case RetrievalStrategy::InvertedIndex:
			return runInvertedIndexRetrieval(db, queries, kTop);
		case RetrievalStrategy::Hnsw:
			return runHnswRetrieval(db, queries, kTop, hnswParams);
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

std::string findStringFieldInJson(std::ifstream& jsonFile, const std::string& fieldName, const std::string& errorMsg) {
	// jsonFile.clear();
	// jsonFile.seekg(0, std::ios::beg);
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


int main(int argc, char** argv) {
	std::string filePath = (argc > 1) ? argv[1] : "data/fiqa-dev/fiqa-dev.h5";

	if (argc > 1) {
		// if filePath extension is not .h5, then it is a directory that has a config.json. That json has a filename field that has the h5 file inside the directory.
		if (std::filesystem::path(filePath).extension() == ".h5") {
			std::cout << "Using input file: " << filePath << '\n';
		} else {
			std::cout << "Input path is not an .h5 file, treating it as a directory: " << filePath << '\n';
			std::filesystem::path configPath = std::filesystem::path(filePath) / "config.json";
			if (!std::filesystem::exists(configPath)) {
				std::cerr << "Config file not found in directory: " << configPath << '\n';
				return 1;
			}
			std::ifstream configFile(configPath);
			if (!configFile.is_open()) {
				std::cerr << "Failed to open config file: " << configPath << '\n';
				return 1;
			}
			std::string h5Filename = findStringFieldInJson(configFile, "filename", "H5 file path not specified in config");
			std::filesystem::path h5Path = std::filesystem::path(filePath) / h5Filename;
			if (!std::filesystem::exists(h5Path)) {
				std::cerr << "H5 file specified in config not found: " << h5Path << '\n';
				return 1;
			}
			std::cout << "Using H5 file from config: " << h5Path << '\n';
			filePath = h5Path.string();
		}
	} else {
		std::cout << "No input file specified, using default: " << filePath << '\n';
	}

	constexpr std::size_t kTop = 30;
	const RetrievalStrategy strategy = (argc > 2) ? parseStrategy(argv[2]) : RetrievalStrategy::InvertedIndex;
	std::string outputPath = "results/cpp_task3/index=(cpp_sparse),query=(default).h5";
	std::string datasetName = "unknown";
	std::string taskName = "task3";
	HnswParams hnswParams;

	if (strategy == RetrievalStrategy::Hnsw) {
		const bool oldStyleHnswArgs = (argc > 3) && isIntegerString(argv[3]);
		if (oldStyleHnswArgs) {
			hnswParams.M = (argc > 3) ? std::stoi(argv[3]) : 32;
			hnswParams.efConstruction = (argc > 4) ? std::stoi(argv[4]) : 200;
			hnswParams.efSearch = (argc > 5) ? std::stoi(argv[5]) : 120;
			outputPath = (argc > 6) ? argv[6] : outputPath;
			datasetName = (argc > 7) ? argv[7] : datasetName;
			taskName = (argc > 8) ? argv[8] : taskName;
		} else {
			outputPath = (argc > 3) ? argv[3] : outputPath;
			datasetName = (argc > 4) ? argv[4] : datasetName;
			taskName = (argc > 5) ? argv[5] : taskName;
			hnswParams.M = (argc > 6) ? std::stoi(argv[6]) : 32;
			hnswParams.efConstruction = (argc > 7) ? std::stoi(argv[7]) : 200;
			hnswParams.efSearch = (argc > 8) ? std::stoi(argv[8]) : 120;
		}
	} else {
		outputPath = (argc > 3) ? argv[3] : outputPath;
		datasetName = (argc > 4) ? argv[4] : datasetName;
		taskName = (argc > 5) ? argv[5] : taskName;
	}

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
					  : ((strategy == RetrievalStrategy::Hnsw) ? "HNSW" : "BruteForce"))
				  << '\n';
		if (strategy == RetrievalStrategy::Hnsw) {
			std::cout << "HNSW params: M=" << hnswParams.M
					  << ", efConstruction=" << hnswParams.efConstruction
					  << ", efSearch=" << hnswParams.efSearch << '\n';
		}
		std::cout << "CSR load time (train + queries): " << loadMs << " ms" << '\n';
		std::cout << "Train shape: [" << db.rows << ", " << db.cols << "]" << '\n';
		std::cout << "Queries shape: [" << queries.rows << ", " << queries.cols << "]" << '\n';

		if (db.cols != queries.cols) {
			throw std::runtime_error("Dimension mismatch between train and queries");
		}

		const RetrievalResult retrieval = runRetrieval(db, queries, kTop, strategy, hnswParams);
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
		} else {
			std::cout << "HNSW build time: " << retrieval.prepElapsedMs << " ms" << '\n';
			std::cout << "HNSW search time: " << retrieval.elapsedMs << " ms" << '\n';
		}

		std::string algo = "cpp_sparse_inverted";
		std::string params = "index=(cpp_sparse),query=(default)";
		if (strategy == RetrievalStrategy::BruteForce) {
			algo = "cpp_sparse_bruteforce";
			// params = "index=(none),query=(bruteforce)";
		} else if (strategy == RetrievalStrategy::Hnsw) {
			algo = "cpp_sparse_hnsw";
			// params = "index=(hnsw,M=" + std::to_string(hnswParams.M) +
			// 	",efConstruction=" + std::to_string(hnswParams.efConstruction) +
			// 	"),query=(efSearch=" + std::to_string(hnswParams.efSearch) + ")";
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