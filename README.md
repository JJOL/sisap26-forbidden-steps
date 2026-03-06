# SISAP 2026 Challenge: Working example in Python 

This repository is a working example for the SISAP 2026 Indexing Challenge <https://sisap-challenges.github.io/>, working with Python and GitHub Actions.

## Installation & Setup

### 1. Clone this repository
```bash
git clone https://github.com/sisap-challenges/sisap26-python-baseline
cd sisap26-python-baseline
```

### 2. Install Dependencies
This repository requires Python 3.8+ and several dependencies. We provide a helper script for easy setup, or you can install manually.

#### Option A: Quick Start (Linux/Mac)
Use the provided install script to set up a virtual environment and install dependencies (including CPU-optimized PyTorch):

```bash
chmod +x install.sh
./install.sh
source venv/bin/activate
```

#### Option B: Manual Installation
1. Install base requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Install CPU-only PyTorch (to avoid large CUDA downloads):
   ```bash
   pip install torch~=2.4.0 --index-url https://download.pytorch.org/whl/cpu
   ```

#### Option C: Docker
Build and run using Docker:
```bash
docker build -t sisap-baseline .
docker run sisap-baseline --task task3 --dataset nq
```

## Running the Code
Run the tasks on an example input using

```bash
python search.py --task {task1, task2, task3}
```
For task3 (approximate nearest neighbor search on sparse data), specifically:
```bash
python search.py --task task3 --dataset nq
```

It will automatically take care of downloading the necessary example dataset.



### Evaluation

```bash
python eval.py results.csv
```
will produce a summary file of the results with the computed recall against the ground truth data. 

This csv file can be further processed to create plots (using `python plot.py --task {task1, task2}`) and show the fastest solutions above a certain recall threshold (using `python show_operating_points.py`).

## How to take this to create my own system
You can fork this repository and polish it to create your solution. Please also take care of the ci workflow (see below).

## GitHub Actions: Continuous integration 

You can monitor your runnings in the "Actions" tab of the GitHub panel: for instance, you can see some runs of this repository:
<https://github.com/sisap-challenges/sisap26-python-baseline/actions>

 
