# Installation

## Accessing the MIMIC-III data
The MIMIC-III dataset contains sensitive patient information, and as such, a training and validation procedure is needed in order to access the dataset. This can be carried out [here](https://mimic.mit.edu/docs/gettingstarted/).
We use the same data preprocessing as the CAML paper, so after obtaining access to the data you can run the following notebook.

## Setting environment variables
Within the CAML folder there is a `caml/constants.py` file.
You will need to point those variables that are currently set to `CHANGEME`.
Use absolute paths to point to datasets and point `MODEL_DIR` to the experiments folder, which we create next.

## Dependencies

There may be additional python dependencies you need to install, but we updated the caml code to be compatible with the more recent torch version we installed for spmlbl.

## Running experiments
```bash
mkdir -p experiments
./run.sh
```
