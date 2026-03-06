import h5py
import os
from urllib.request import urlretrieve
from pathlib import Path
from scipy.sparse import csr_matrix

def download(src, dst):
    print(dst)
    if not os.path.exists(dst):
        os.makedirs(Path(dst).parent, exist_ok=True)
        print('downloading %s -> %s...' % (src, dst))
        urlretrieve(src, dst)

def load_sparse_matrix(h5_group):
    """Reconstructs a SciPy CSR matrix from HDF5 datasets."""
    indptr = h5_group['indptr'][:]
    indices = h5_group['indices'][:]
    data = h5_group['data'][:]
    shape = tuple(h5_group.attrs['shape'])
    return csr_matrix((data, indices, indptr), shape=shape)

def get_fn(dataset, task):
    return os.path.join("data", dataset, task, f"{dataset}.h5")

def prepare(dataset, task):
    url = DATASETS[dataset][task]['url']
    fn = get_fn(dataset, task)

    download(url, fn)

def get_query_count(dataset, task):
    fn = get_fn(dataset, task) 
    f = h5py.File(fn)
    qn = len(DATASETS[dataset][task]['queries'](f))
    f.close()
    return qn

DATASETS = {
    'wikipedia-small': {
        'task1': {
            'url': 'https://huggingface.co/datasets/SISAP-Challenges/SISAP2026/resolve/main/benchmark-dev-wikipedia-bge-m3-small.h5',
            'data': lambda x: x['train'],
            'gt_I': lambda x: x['allknn']['knns'],
            'k': 15,
        }
    },
    'llama-dev': {
        'task2' : {
            'url': 'https://huggingface.co/datasets/SISAP-Challenges/SISAP2026/resolve/main/llama-dev.h5',
            'queries': lambda x: x['test']['queries'],
            'data': lambda x: x['train'],
            'gt_I': lambda x: x['test']['knns'],
            'k': 30,
        }
    },
    'nq': {
        'task3': {
            'url': 'https://huggingface.co/datasets/SISAP-Challenges/SISAP2026/resolve/refs%2Fpr%2F3/nq.h5',
            'queries': lambda x: load_sparse_matrix(x['otest']['queries']),
            'data': lambda x: load_sparse_matrix(x['train']),
            'gt_I': lambda x: x['otest']['knns'],
            'k': 30,
        }
    },
    'fiqa-dev': {
        'task3': {
            'url': 'https://huggingface.co/datasets/SISAP-Challenges/SISAP2026/resolve/refs%2Fpr%2F3/fiqa.h5',
            'queries': lambda x: load_sparse_matrix(x['otest']['queries']),
            'data': lambda x: load_sparse_matrix(x['train']),
            'gt_I': lambda x: x['otest']['knns'],
            'k': 30,            
        }
    }
}