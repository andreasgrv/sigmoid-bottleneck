#!/usr/bin/env bash
MAX_ITERS=50000
BATCH_SIZE=64
LR=0.0001

for D in 25 50 100 200 400
do
	python train.py \
		--blueprint blueprints/dft.yaml \
	   	--paths.experiment_name DFT-D-$D-S-$SEED \
		--model model/Open_ImagesV6_TRresNet_L_448.pth \
		--batch_size $BATCH_SIZE \
		--output_layer.slack_dims $D \
		--output_layer.k 50 \
		--lr $LR \
		--seed $SEED > logs/DFT-D-$D-S-$SEED
done
