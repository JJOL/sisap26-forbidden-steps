import argparse
import faiss
import h5py
import numpy as np
import os
from pathlib import Path
import time
from tqdm import tqdm
import torch
from datasets import DATASETS, prepare, get_fn

def store_results(dst, algo, dataset, task, D, I, buildtime, querytime, params):
    os.makedirs(Path(dst).parent, exist_ok=True)
    f = h5py.File(dst, 'w')
    f.attrs['algo'] = algo
    f.attrs['dataset'] = dataset
    f.attrs['task'] = task
    f.attrs['buildtime'] = buildtime
    f.attrs['querytime'] = querytime
    f.attrs['params'] = params
    f.create_dataset('knns', I.shape, dtype=I.dtype)[:] = I
    f.create_dataset('dists', D.shape, dtype=D.dtype)[:] = D
    f.close()


def run_task1(dataset, task, k):
    print(f'Running {task} on {dataset}')

    prepare(dataset, task)

    fn = get_fn(dataset, task)
    f = h5py.File(fn)
    data = np.array(DATASETS[dataset][task]['data'](f), dtype=np.float32)
    f.close()

    n, d = data.shape
    k = k + 1 # need to query for one more point to include self-loop 

    nlist = 1024 # number of clusters/centroids to build the IVF from
    index_identifier = f"IVF{nlist},SQfp16"

    index = faiss.index_factory(d, index_identifier, faiss.METRIC_INNER_PRODUCT)

    print(f"Training index on {data.shape} with {data.dtype}")
    start = time.time()
    index.train(data)
    index.add(data)
    elapsed_build = time.time() - start
    print(f"Done training in {elapsed_build}s.")
    assert index.is_trained

    for nprobe in [1, 2, 5, 10, 100]:
        print(f"Starting search on {data.shape} with nprobe={nprobe}")
        start = time.time()
        index.nprobe = nprobe
        D, I = index.search(data, k)
        elapsed_search = time.time() - start
        print(f"Done searching in {elapsed_search}s.")

        I = I + 1 # FAISS is 0-indexed, groundtruth is 1-indexed

        identifier = f"index=({index_identifier}),query=(nprobe={nprobe})"

        store_results(os.path.join("results/", dataset, task, f"{identifier}.h5"), "faissIVF", 
                      dataset, task, D, I, elapsed_build, elapsed_search, identifier)

def run_task2(dataset, task, k):
    print(f'Running {task} on {dataset}')

    prepare(dataset, task)

    fn = get_fn(dataset, task)
    f = h5py.File(fn)
    data = np.array(DATASETS[dataset][task]['data'](f))
    queries = np.array(DATASETS[dataset][task]['queries'](f))
    f.close()

    n, d = data.shape

    nlist = 1024 # number of clusters/centroids to build the IVF from
    index_identifier = f"IVF{nlist},SQfp16"

    index = faiss.index_factory(d, index_identifier, faiss.METRIC_INNER_PRODUCT)

    print(f"Training index on {data.shape} with {data.dtype}")
    start = time.time()
    index.train(data)
    index.add(data)
    elapsed_build = time.time() - start
    print(f"Done training in {elapsed_build}s.")
    assert index.is_trained

    for nprobe in [1, 2, 5, 10, 100]:
        print(f"Starting search on {queries.shape} with nprobe={nprobe}")
        start = time.time()
        index.nprobe = nprobe
        D, I = index.search(queries, k)
        elapsed_search = time.time() - start
        print(f"Done searching in {elapsed_search}s.")

        I = I + 1 # FAISS is 0-indexed, groundtruth is 1-indexed

        identifier = f"index=({index_identifier}),query=(nprobe={nprobe})"

        store_results(os.path.join("results/", dataset, task, f"{identifier}.h5"), "faissIVF", 
                      dataset, task, D, I, elapsed_build, elapsed_search, identifier)

def run_task3(dataset, task, k):
    print(f'Running {task} on {dataset}')

    prepare(dataset, task)

    fn = get_fn(dataset, task)
    f = h5py.File(fn)
    corpus = DATASETS[dataset][task]['data'](f)
    queries = DATASETS[dataset][task]['queries'](f)
    f.close()

    print(f"Corpus shape: {corpus.shape}")
    print(f"Queries shape: {queries.shape}")

    n_queries = queries.shape[0]

    device = torch.device("cpu")

    # Prepare corpus on device
    # Scipy CSR uses int32 usually, but PyTorch wants int64 for indices
    indptr = torch.from_numpy(corpus.indptr.astype(np.int64)).to(device)
    indices = torch.from_numpy(corpus.indices.astype(np.int64)).to(device)
    data = torch.from_numpy(corpus.data.astype(np.float32)).to(device)
    
    corpus_torch = torch.sparse_csr_tensor(
        indptr, indices, data, size=corpus.shape, device=device
    )
    
    I = np.zeros((n_queries, k), dtype=np.int32)
    D = np.zeros((n_queries, k), dtype=np.float32)

    batch_size = 100
    
    start_time = time.time()
    print(f"Extracting top-{k} neighbors with PyTorch sparse mm...")
    
    # Process in batches
    for i in tqdm(range(0, n_queries, batch_size), desc="Processing Batches"):
        end_idx = min(i + batch_size, n_queries)
        
        # Get query batch (scipy sparse matrix)
        q_batch_scipy = queries[i:end_idx]
        
        # Convert to dense PyTorch tensor on device
        # Note: toarray() creates a dense numpy array
        q_batch_dense = torch.from_numpy(q_batch_scipy.toarray().astype(np.float32)).to(device)
        
        # Matrix multiplication: 
        # corpus (N_docs, N_feat) @ q_batch.T (N_feat, batch_size) -> scores (N_docs, batch_size)
        scores = torch.mm(corpus_torch, q_batch_dense.T)
        
        # Get top-k along dim 0 (for each query in the batch)
        # values, indices shapes: (k, batch_size)
        top_val, top_idx = torch.topk(scores, k, dim=0)
        
        # Transpose to (batch_size, k) and move to CPU/numpy
        # Add 1 to indices for 1-based groundtruth
        I[i:end_idx] = top_idx.t().cpu().numpy() + 1
        D[i:end_idx] = top_val.t().cpu().numpy()

    elapsed_search = time.time() - start_time
    print(f"Extraction completed in {elapsed_search:.2f} seconds.")

    identifier = "pytorch_sparse_mm"
    store_results(os.path.join("results/", dataset, task, f"{identifier}.h5"), identifier, 
                  dataset, task, D, I, 0.0, elapsed_search, identifier)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=['task1', 'task2', 'task3'],
        default='task2'
    )

    parser.add_argument(
        '--dataset',
        choices=DATASETS.keys(),
        default='llama-dev'
    )

    args = parser.parse_args()
    if args.task not in DATASETS[args.dataset]:
        print(f"Task '{args.task}' incompatible with dataset '{args.dataset}'")
        exit(1)    

    if args.task == 'task1':
        run_task1(args.dataset, args.task, DATASETS[args.dataset][args.task]['k'])
    elif args.task == 'task2':
        run_task2(args.dataset, args.task, DATASETS[args.dataset][args.task]['k'])
    elif args.task == 'task3':
        run_task3(args.dataset, args.task, DATASETS[args.dataset][args.task]['k'])


