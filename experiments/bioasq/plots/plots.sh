#!/usr/bin/env bash

mlbl.plot_dataset --data ../data/subsets-v-20000/train-100k-part-1.json --vocab ../data/subsets-v-20000/vocab.txt --title "BioASQ Training Set"
mlbl.plot_dataset --data ../data/subsets-v-20000/test-10k.json --vocab ../data/subsets-v-20000/vocab.txt --title "BioASQ Test Set"


python plot_feasibility.py --file ../experiments/
python plot_metrics.py --file ../experiments/ --metric f1@10
python plot_train_time.py --logs ../logs

python plot_metrics.py --file ../experiments/ --metric macrof1
python plot_metrics.py --file ../experiments/ --metric f1
python plot_metrics.py --file ../experiments/ --metric ndcg
python plot_metrics.py --file ../experiments/ --metric prec@10
python plot_metrics.py --file ../experiments/ --metric rec@10
