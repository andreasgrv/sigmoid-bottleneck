#!/usr/bin/env bash

MAX_ITERS=50000
VOCAB_DIM=20000
PART="1"
BATCH_SIZE=32
LR=0.00005

for D in 25 50 100 200 400
do
	python train.py --blueprint blueprints/dft.yaml --paths.experiment_name DFT-L-P-$PART-D-$D-S-$SEED --data.train_path "data/subsets-v-20000/train-100k-part-[$PART].json" --data.valid_path data/subsets-v-20000/valid-5k.json --data.labels_file data/subsets-v-20000/vocab.txt --output_layer.out_dim $VOCAB_DIM --output_layer.slack_dims $D --output_layer.k 50 --max_iters $MAX_ITERS --lr $LR --batch_size $BATCH_SIZE  --num_proc 10 --seed $SEED --freeze_encoder false
done
