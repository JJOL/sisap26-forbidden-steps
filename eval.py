import argparse
import h5py
import numpy as np
import csv
import glob
from datasets import DATASETS, get_fn, prepare, get_h5_item

def get_all_results(dirname):
    mask = [dirname + "/**/*.h5", dirname + "/**/*/*.h5"]
    print("Searching for results matching:")
    print("\n".join(mask))
    for m in mask:
        for fn in glob.iglob(m):
            print(fn)
            f = h5py.File(fn, "r")
            if "knns" not in f or not ("dataset" in f or "dataset" in f.attrs):
                print("Ignoring " + fn)
                f.close()
                continue
            yield f
            f.close()

def get_recall(I, gt, k):
    assert k <= I.shape[1]
    assert len(I) == len(gt)

    n = len(I)
    recall = 0
    for i in range(n):
        recall += len(set(I[i, :k]) & set(gt[i, :k]))
    return recall / (n * k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        help='directory in which results are stored',
        default="results"
    )
    parser.add_argument(
        '--private',
        help="private queries held out for evaluation",
        action='store_true',
        default=False
    )

    parser.add_argument("csvfile")
    args = parser.parse_args()
    true_I_cache = {}


    columns = ["dataset", "task", "algo", "buildtime", "querytime", "params", "recall"]

    with open(args.csvfile, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for res in get_all_results(args.results):
            dataset = res.attrs["dataset"]
            task = res.attrs["task"]
            assert dataset in DATASETS and task in DATASETS[dataset]
            prepare(dataset, task)
            d = dict(res.attrs)
            # print(d)
            fn = get_fn(dataset, task)
            print(f"Using groundtruth in {fn}")
            f = h5py.File(fn)
            gt_I = np.array(get_h5_item(f, DATASETS[dataset][task]['gt_I']))
            # Task 2 ground-truth uses 0-based indices; shift to 1-based to
            # match the 1-based IDs required in result files by the spec.
            if task == "task2":
                gt_I = gt_I + 1
            f.close()
            recall = get_recall(np.array(res["knns"]), gt_I, DATASETS[dataset][task]['k'])
            d['recall'] = recall
            print(d["dataset"], d["task"], d["algo"], d["params"], "=>", recall)
            writer.writerow(d)