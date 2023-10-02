#!/usr/bin/env bash
python plot_metrics.py --file ../experiments/ --metric f1@10
python plot_feasibility.py --file ../experiments/
python plot_loss.py --results ../../bioasq/experiments/*100*/stats.tsv --attributes train.loss
python plot_train_time.py --logs ../logs

python plot_metrics.py --file ../experiments/ --metric macrof1
python plot_metrics.py --file ../experiments/ --metric f1
python plot_metrics.py --file ../experiments/ --metric ndcg
python plot_metrics.py --file ../experiments/ --metric prec@10
python plot_metrics.py --file ../experiments/ --metric rec@10
