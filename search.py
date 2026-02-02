import argparse
import faiss
import h5py
import numpy as np
import os
from pathlib import Path
import time
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

def run(dataset, task, k):
    print(f'Running {task} on {dataset}')

    prepare(dataset, task)

    fn, _ = get_fn(dataset, task)
    f = h5py.File(fn)
    data = np.array(DATASETS[dataset][task]['data'](f))
    queries = np.array(DATASETS[dataset][task]['queries'](f))
    f.close()

    n, d = data.shape
    if task == 'task2':
        k = k + 1 # need to search for one more NN since we cannot remove self-loop

    nlist = 1024 # number of clusters/centroids to build the IVF from
    index_identifier = f"IVF{nlist},SQfp16"

    index = faiss.index_factory(d, index_identifier, faiss.METRIC_INNER_PRODUCT)

    print(f"Training index on {data.shape}")
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

        #I = I + 1 # FAISS is 0-indexed, groundtruth is 1-indexed

        identifier = f"index=({index_identifier}),query=(nprobe={nprobe})"

        store_results(os.path.join("results/", dataset, task, f"{identifier}.h5"), "faissIVF", dataset, task, D, I, elapsed_build, elapsed_search, identifier)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=['task1', 'task2'],
        default='task2'
    )

    parser.add_argument(
        '--dataset',
        choices=DATASETS.keys(),
        default='llama'
    )


    args = parser.parse_args()
    run(args.dataset, args.task, DATASETS[args.dataset][args.task]['k'])

