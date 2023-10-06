# Sparse Multi-Label Classification (spmlbl)

Herein we provide the code to reproduce our results for the paper:

> Taming the Sigmoid Bottleneck: Provably Argmaxable Sparse Multi-Label Classification

# Installation

Below installation is for main library.

## Install Python Dependencies
```bash
python3.8 -m venv .env
source .env/bin/activate
# Adapt cuda in requirements to your use case
pip install -r requirements.txt
pip install -e .
# NOTE: For each specific dataset you may need to install more libraries
# See README.md file in each experiment section
```

## Set Environment Variables

```bash
# Avoid pytorch using too many threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Random seed - needed for exps
export SEED=0
# What device to run models on
export MLBL_DEVICE="cuda:0"
# Number of threads
export MLBL_NUM_PROC=10
```

## Install [Gurobi](https://www.gurobi.com/academia/academic-program-and-licenses/)

The linear programming algorithm that detects (un)argmaxable label assignments depends on Gurobi.
It requires a license, see link above.


## Run Tests

Tests require dependencies and Gurobi.

```bash
py.test tests
```

## Experiments

For each dataset, see the `README.md` file in the corresponding folder under experiments.

* [MIMIC-III](experiments/mimic/README.md)
* [BioASQ Task A](experiments/bioasq/README.md)
* [OpenImages v6](experiments/openimagesv6/README.md)
