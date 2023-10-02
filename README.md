# Taming the Sigmoid Bottleneck: Provably Argmaxable Sparse Multi-Label Classification

Herein we provide the code to reproduce our results for the paper.

# Installation

## Install Python Dependencies
```bash
python3.8 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
pip install -e .
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


# Next Steps

Release this code on github together with exact instructions for reproducibility.
