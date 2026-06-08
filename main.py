import h5py
from scipy.sparse import csr_matrix, vstack
import timeit
import numpy as np
from numba import jit
# import torch
# from transformers import AutoTokenizer, AutoModelForMaskedLM
# from beir.datasets.data_loader import GenericDataLoader

def load_sparse_matrix(h5_group):
    print(f"Loading sparse matrix from group: {h5_group.name}")
    print(f"Shape attribute: {h5_group.attrs['shape']}")
    indptr = h5_group['indptr'][:]
    indices = h5_group['indices'][:]
    data = h5_group['data'][:]
    shape = tuple(h5_group.attrs['shape'])
    return csr_matrix((data, indices, indptr), shape=shape)

# def embed_splade3(data, dimensions=30522, mapper=lambda x: x, model=None, tokenizer=None, batch_size=64):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     if model is None or tokenizer is None:
#         model_name = "naver/splade-cocondenser-ensembledistil"
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         model = AutoModelForMaskedLM.from_pretrained(model_name)
#         model = model.to(device)
#         model.eval()

#     if isinstance(data, dict):
#         texts = [mapper(v) for v in data.values()]
#     else:
#         texts = [mapper(data[i]) for i in range(data.shape[0])]

#     batch_matrices = []

#     with torch.no_grad():
#         for i in range(0, len(texts), batch_size):
#             if i % (batch_size * 100) == 0:
#                 print(f"Embedding {i}/{len(texts)}...")

#             batch_texts = texts[i:i + batch_size]
#             encoded = tokenizer(
#                 batch_texts, padding=True, truncation=True,
#                 max_length=512, return_tensors="pt"
#             )
#             encoded = {k: v.to(device) for k, v in encoded.items()}

#             output = model(**encoded)
#             # SPLADE transformation: max over token dim of log(1 + ReLU(logits))
#             sparse_vecs = torch.log(1 + torch.relu(output.logits))          # (B, seq_len, vocab)
#             sparse_vecs = sparse_vecs * encoded["attention_mask"].unsqueeze(-1)
#             sparse_vecs, _ = torch.max(sparse_vecs, dim=1)                  # (B, vocab)

#             batch_matrices.append(csr_matrix(sparse_vecs.cpu().float().numpy()))

#     return vstack(batch_matrices)


def main():
    # fiqa-dev.h5 is an h5 file

    with h5py.File('data/fiqa-dev.h5', 'r') as f:
        train_matrix = load_sparse_matrix(f["train"])
        # h5_group = f["train"]
        # print(f"Loading sparse matrix from group: {h5_group.name}")
        # print(f"Shape attribute: {h5_group.attrs['shape']}")
        # indptr = h5_group['indptr'][:]
        # indices = h5_group['indices'][:]
        # data = h5_group['data'][:]
        # shape = tuple(h5_group.attrs['shape'])

        # print

        queries = load_sparse_matrix(f["otest/queries"])
        print(f"Train matrix shape: {train_matrix.shape}")
        print(f"Queries matrix shape: {queries.shape}")

    # nq dataset is a directory with corpus.jsonl and queries.jsonl files as well as qrels/test.tsv.
    # to load equivalente train_matrix and queries, we can use the following code:
    # print("Loading NQ dataset...")
    # corpus, queries, qrels = GenericDataLoader(data_folder='data/nq').load(split="test")
    
    # # the corpus is a dictionary with  2M entries labeled "doc0", "doc1", .... and each entry has a "text" and "title" field.
    # # the queries is a dictionary with 3k entries labeled "test0", "test1", ... and each entry is the plain question text.
    # # We now want to embed every text into a vector space using SPLADE-v3, which is a sparse embedding model that produces sparse vectors and want 30,522 dimensions,

    # print("Embedding corpus and queries with SPLADE-v3...")
    # start_time = timeit.default_timer()
    # model_name = "naver/splade-cocondenser-ensembledistil"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # splade_model = AutoModelForMaskedLM.from_pretrained(model_name)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # splade_model = splade_model.to(device)
    # splade_model.eval()

    # train_matrix = embed_splade3(corpus, dimensions=30522, mapper=lambda x: x["text"],
    #                               model=splade_model, tokenizer=tokenizer)
    # queries = embed_splade3(queries, dimensions=30522, mapper=lambda x: x,
    #                          model=splade_model, tokenizer=tokenizer)
    # end_time = timeit.default_timer()
    # print(f"Time taken to embed corpus and queries: {end_time - start_time} seconds")
    print(f"Train matrix shape: {train_matrix.shape}")
    print(f"Queries matrix shape: {queries.shape}")

    # We should save them in h5 files for later use: A nq.h5 will have two groups: "train" and "otest/queries", each containing the sparse matrix in CSR format (indptr, indices, data) and the shape as an attribute.
    # with h5py.File('nq.h5', 'w') as f:
    #     train_group = f.create_group('train')
    #     train_group.create_dataset('indptr', data=train_matrix.indptr)
    #     train_group.create_dataset('indices', data=train_matrix.indices)
    #     train_group.create_dataset('data', data=train_matrix.data)
    #     train_group.attrs['shape'] = train_matrix.shape

    #     queries_group = f.create_group('otest/queries')
    #     queries_group.create_dataset('indptr', data=queries.indptr)
    #     queries_group.create_dataset('indices', data=queries.indices)
    #     queries_group.create_dataset('data', data=queries.data)
    #     queries_group.attrs['shape'] = queries.shape

    

    start_time = timeit.default_timer()
    results = np.empty((queries.shape[0], 30), dtype=object)  # preallocate results array

    @jit(nopython=True)
    def compute_dot_products(train_matrix: np.ndarray, queries: np.ndarray, results: np.ndarray):
        for i in range(queries.shape[0]):
            # if (i + 1) % 1000 == 0: print(f"Processing query {i + 1}/{queries.shape[0]}")
            # print(f"Processing query {i}/{queries.shape[0]}")
            query = queries[i]
            scores = (train_matrix @ query.T).toarray().flatten()
            top_indices = np.argpartition(scores, -30)[-30:] 
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    
            # results[i] = list(zip(top_indices.tolist(), scores[top_indices].tolist()))
            # results[i] = list(zip(top_indices, scores[top_indices]))
            # only store indices of documents/vectors, not scores
            results[i] = top_indices.tolist()
        # dot_products = [(j, query @ train_matrix[j].T) for j in range(train_matrix.shape[0])]
        # dot_products.sort(key=lambda x: x[1], reverse=True)

    compute_dot_products(train_matrix, queries, results)
    end_time = timeit.default_timer()
        # results.append(dot_products[:15])
        # print(f"Time taken: {end_time - start_time} seconds")
    # time per query
    total_time = end_time - start_time
    print(f"Average time per query: {total_time / queries.shape[0] * 1000} ms")
    print(f"Total time taken: {total_time} seconds")



if __name__ == "__main__":
    main()