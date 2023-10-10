#!/usr/bin/env bash

python plot_metrics.py --file ../experiments --metric f1_at_8_te
python plot_feasibility.py --file ../experiments/

python plot_metrics.py --file ../experiments/ --metric f1_macro_te
python plot_metrics.py --file ../experiments/ --metric f1_micro_te
python plot_metrics.py --file ../experiments/ --metric prec_at_8_te
python plot_metrics.py --file ../experiments/ --metric rec_at_8_te
