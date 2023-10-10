We rely on the ASL paper's code here and have included it below just to make the process easier.

# Installation

Install all ASL paper dependencies.
```bash
pip install git+https://github.com/mapillary/inplace_abn.git@v1.0.12
```

## Running the experiments

You can use the scripts `run-bsl.sh` and `run-dft.sh` to train the BSL and DFT models, correspondingly.
Set the `SEED` environment variable to change the random state.
```bash
export SEED=0
mkdir -p logs
./run-bsl.sh
./run-dft.sh
```


## Verifying results

```bash
# Verification via the LP is parallelisable (the larger you can afford to make NUM_PROC, the better)
export MLBL_NUM_PROC=10
./eval.sh
```
