#!/usr/bin/env bash

python plot_metrics.py --file ../saved_models --metric f1_at_8_te
python plot_feasibility.py --file ../saved_models/

python plot_metrics.py --file ../saved_models/ --metric f1_macro_te
python plot_metrics.py --file ../saved_models/ --metric f1_micro_te
python plot_metrics.py --file ../saved_models/ --metric prec_at_8_te
python plot_metrics.py --file ../saved_models/ --metric rec_at_8_te
