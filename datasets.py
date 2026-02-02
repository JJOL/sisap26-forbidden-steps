import h5py
import os 
from urllib.request import urlretrieve
from pathlib import Path

def download(src, dst):
    if not os.path.exists(dst):
        os.makedirs(Path(dst).parent, exist_ok=True)
        print('downloading %s -> %s...' % (src, dst))
        urlretrieve(src, dst)

def get_fn(dataset, task):
    return os.path.join("data", dataset, task, f"{dataset}.h5"), os.path.join('data', dataset, task, 'gt', f'gt_{dataset}.h5')

def prepare(dataset, task):
    url = DATASETS[dataset][task]['url']
    gt_url = DATASETS[dataset][task]['gt_url']
    fn, gt_fn = get_fn(dataset, task)

    download(url, fn)
    download(gt_url, gt_fn)

def get_query_count(dataset, task):
    fn, _ = get_fn(dataset, task) 
    f = h5py.File(fn)
    qn = len(DATASETS[dataset][task]['queries'](f))
    f.close()
    return qn

DATASETS = {
    'llama': {
        'task2' : {
            'url': 'https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/llama-128-ip.hdf5',
            'queries': lambda x: x['test'],
            'data': lambda x: x['train'],
            'gt_url': 'https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/llama-128-ip.hdf5',
            'gt_I': lambda x: x['neighbors'],
            'k': 30,
        }
    }
}