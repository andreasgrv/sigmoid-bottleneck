# Installation

## Accessing the MIMIC-III data
The MIMIC-III dataset contains sensitive patient information, and as such, a training and validation procedure is needed in order to access the dataset. This can be carried out [here](https://mimic.mit.edu/docs/gettingstarted/).
We use the same data preprocessing as the CAML paper, so after obtaining access to the data you can run the following notebook.

## Processing the MIMIC-III data

Follow the notebook [caml/notebooks/dataproc_mimic_III.ipynb](caml/notebooks/dataproc_mimic_III.ipynb).

## Setting environment variables
Within the CAML folder there is a `caml/constants.py` file.
You will need to point those variables that are currently set to `CHANGEME`.
Use absolute paths to point to datasets and point `MODEL_DIR` to the experiments folder, which we create next.

## Dependencies

```bash
pip install -r requirements.txt
```
There may be additional python dependencies you need to install, but we updated the caml code to be compatible with the more recent torch version we installed for spmlbl.

## Running experiments
```bash
export PYTHONPATH="$PWD/caml"
mkdir -p experiments
mkdir -p logs
./run.sh
```

## Verifying models

After running the command below, each model folder will be populated with two analysis files: `test-analysis.json` and `dev-analysis.json`.
```bash
./eval.sh
```
