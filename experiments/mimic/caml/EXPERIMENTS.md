# Reproduce plot 1 results:
```py
python plot_single_feasibility.py --file saved_models/cnn_vanilla_bottleneck_k0_s50_seed1/test-analysis.json --data /mnt/subverse/datasets/mimic/mimic3/train_full.csv
```

# Reproduce figure 6
python plot_metrics.py --file saved_models/ --metric f1_at_8_te

python plot_feasibility.py --file saved_models/
