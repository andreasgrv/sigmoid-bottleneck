# BioASQ Task A

## Getting the data

1. Register for BioASQ here: [http://participants-area.bioasq.org/accounts/register/](http://participants-area.bioasq.org/accounts/register/)
2. Tick task A from the list of Tasks at the bottom of the registration page.
3. Navigate to the datasets: [http://participants-area.bioasq.org/datasets/](http://participants-area.bioasq.org/datasets/)
4. Download `Training v.2021` - the txt version which gives `allMeSH_2021.zip`.
5. Unzip it and place the file in the `experiments/bioasq/data` folder.

## (Re)Creating the dataset

In what follows, we have provided scripts to recreate the dataset.
Each abstract has a unique key, its pmid. We recreate the splits we created for the paper by filtering the downloaded dataset according to the pmids.
For more information on how we created the original splits - e.g. got the pmids, see the appendix in the paper and the scripts under `bin/preprocess`.
The script below assumes you have placed `allMeSH_2021.json` in the `experiments/bioasq/data` folder.

```bash
pip install awscli
mkdir -p data/subsets-v-20000
aws s3 cp s3://sigmoid-bottleneck/bioasq/data/train-100k-part-1.csv --no-sign-request  data/subsets-v-20000
aws s3 cp s3://sigmoid-bottleneck/bioasq/data/valid-5k.csv --no-sign-request  data/subsets-v-20000
aws s3 cp s3://sigmoid-bottleneck/bioasq/data/test-10k.csv --no-sign-request  data/subsets-v-20000
aws s3 cp s3://sigmoid-bottleneck/bioasq/data/vocab.txt --no-sign-request  data/subsets-v-20000
python construct_dataset.py --data allMeSH_2021.json
```

The script should take approximately 5 minutes to run and it will create 3 json files, so the directory structure should look like:

```
.
├── allMeSH_2021.json
├── construct_dataset.py
└── subsets-v-20000
    ├── test-10k.csv
    ├── train-100k-part-1.csv
    ├── valid-5k.csv
    ├── test-10k.json
    ├── train-100k-part-1.json
    ├── valid-5k.json
    └── vocab.txt
```

## Running the experiments

You can use the scripts `run-bsl.sh` and `run-dft.sh` to train the BSL and DFT models, correspondingly.
Set the `SEED` environment variable to change the random state.
```bash
export SEED=0
./run-bsl.sh
```

Each experiment is written to the experiment folder.

## Verifying results

```bash
# Verification via the LP is parallelisable (the larger you can afford to make NUM_PROC, the better)
export MLBL_NUM_PROC=10
./eval.sh
```
