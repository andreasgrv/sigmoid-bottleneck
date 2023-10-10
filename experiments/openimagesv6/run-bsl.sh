#!/usr/bin/env bash
MAX_ITERS=50000
BATCH_SIZE=64
LR=0.0001

for D in 25 50 100 200 400
do
	python train.py \
		--blueprint blueprints/sigmoid-bottleneck.yaml \
		--paths.experiment_name BSL-D-$D-S-$SEED \
		--model model/Open_ImagesV6_TRresNet_L_448.pth \
		--max_iters $MAX_ITERS \
		--output_layer.feature_dim $D \
		--batch_size $BATCH_SIZE \
		--lr $LR \
		--seed $SEED > logs/BSL-D-$D-S-$SEED
done
