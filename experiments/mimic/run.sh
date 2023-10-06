#!/usr/bin/env bash
export PYTHONPATH="$PWD/caml"
FMAPS=500
DROPOUT=0.2
BS=16
LR=0.001

mkdir -p logs

for SEED in 0 1 2
do
	python caml/learn/training.py \
		/mnt/subverse/datasets/mimic/mimic3/train_full.csv \
		/mnt/subverse/datasets/mimic/mimic3/vocab.csv \
		full \
		cnn_vanilla \
		100 \
		--clf bottleneck \
		--k 0 \
		--slack-dims 25 \
		--filter-size 4 \
		--num-filter-maps $FMAPS \
		--dropout $DROPOUT \
		--lr $LR \
		--embed-file /mnt/subverse/datasets/mimic/mimic3/processed_full.embed \
		--batch-size $BS \
		--patience 10 \
		--criterion prec_at_8 \
		--seed $SEED \
		--gpu > logs/BSL-D-25-S-$SEED

	python caml/learn/training.py \
		/mnt/subverse/datasets/mimic/mimic3/train_full.csv \
		/mnt/subverse/datasets/mimic/mimic3/vocab.csv \
		full \
		cnn_vanilla \
		100 \
		--clf bottleneck \
		--k 0 \
		--slack-dims 50 \
		--filter-size 4 \
		--num-filter-maps $FMAPS \
		--dropout $DROPOUT \
		--lr $LR \
		--embed-file /mnt/subverse/datasets/mimic/mimic3/processed_full.embed \
		--batch-size $BS \
		--patience 10 \
		--criterion prec_at_8 \
		--seed $SEED \
		--gpu > logs/BSL-D-50-S-$SEED

	python caml/learn/training.py \
		/mnt/subverse/datasets/mimic/mimic3/train_full.csv \
		/mnt/subverse/datasets/mimic/mimic3/vocab.csv \
		full \
		cnn_vanilla \
		100 \
		--clf bottleneck \
		--k 0 \
		--slack-dims 100 \
		--filter-size 4 \
		--num-filter-maps $FMAPS \
		--dropout $DROPOUT \
		--lr $LR \
		--embed-file /mnt/subverse/datasets/mimic/mimic3/processed_full.embed \
		--batch-size $BS \
		--patience 10 \
		--criterion prec_at_8 \
		--seed $SEED \
		--gpu > logs/BSL-D-100-S-$SEED

	# D = 200
	python caml/learn/training.py \
		/mnt/subverse/datasets/mimic/mimic3/train_full.csv \
		/mnt/subverse/datasets/mimic/mimic3/vocab.csv \
		full \
		cnn_vanilla \
		100 \
		--clf bottleneck \
		--k 0 \
		--slack-dims 200 \
		--filter-size 4 \
		--num-filter-maps $FMAPS \
		--dropout $DROPOUT \
		--lr $LR \
		--embed-file /mnt/subverse/datasets/mimic/mimic3/processed_full.embed \
		--batch-size $BS \
		--patience 10 \
		--criterion prec_at_8 \
		--seed $SEED \
		--gpu > logs/BSL-D-200-S-$SEED

	# D = 400
	python caml/learn/training.py \
		/mnt/subverse/datasets/mimic/mimic3/train_full.csv \
		/mnt/subverse/datasets/mimic/mimic3/vocab.csv \
		full \
		cnn_vanilla \
		100 \
		--clf bottleneck \
		--k 0 \
		--slack-dims 400 \
		--filter-size 4 \
		--num-filter-maps $FMAPS \
		--dropout $DROPOUT \
		--lr $LR \
		--embed-file /mnt/subverse/datasets/mimic/mimic3/processed_full.embed \
		--batch-size $BS \
		--patience 10 \
		--criterion prec_at_8 \
		--seed $SEED \
		--gpu > logs/BSL-D-400-S-$SEED

	# VANDER FFT LAYERS
	python caml/learn/training.py \
		/mnt/subverse/datasets/mimic/mimic3/train_full.csv \
		/mnt/subverse/datasets/mimic/mimic3/vocab.csv \
		full \
		cnn_vanilla \
		100 \
		--clf fft \
		--k 80 \
		--slack-dims 25 \
		--filter-size 4 \
		--num-filter-maps $FMAPS \
		--dropout $DROPOUT \
		--lr $LR \
		--embed-file /mnt/subverse/datasets/mimic/mimic3/processed_full.embed \
		--batch-size $BS \
		--patience 10 \
		--criterion prec_at_8 \
		--seed $SEED \
		--gpu > logs/DFT-D-25-S-$SEED

	python caml/learn/training.py \
		/mnt/subverse/datasets/mimic/mimic3/train_full.csv \
		/mnt/subverse/datasets/mimic/mimic3/vocab.csv \
		full \
		cnn_vanilla \
		100 \
		--clf fft \
		--k 80 \
		--slack-dims 50 \
		--filter-size 4 \
		--num-filter-maps $FMAPS \
		--dropout $DROPOUT \
		--lr $LR \
		--embed-file /mnt/subverse/datasets/mimic/mimic3/processed_full.embed \
		--batch-size $BS \
		--patience 10 \
		--criterion prec_at_8 \
		--seed $SEED \
		--gpu > logs/DFT-D-50-S-$SEED

	python caml/learn/training.py \
		/mnt/subverse/datasets/mimic/mimic3/train_full.csv \
		/mnt/subverse/datasets/mimic/mimic3/vocab.csv \
		full \
		cnn_vanilla \
		100 \
		--clf fft \
		--k 80 \
		--slack-dims 100 \
		--filter-size 4 \
		--num-filter-maps $FMAPS \
		--dropout $DROPOUT \
		--lr $LR \
		--embed-file /mnt/subverse/datasets/mimic/mimic3/processed_full.embed \
		--batch-size $BS \
		--patience 10 \
		--criterion prec_at_8 \
		--seed $SEED \
		--gpu > logs/DFT-D-100-S-$SEED

	python caml/learn/training.py \
		/mnt/subverse/datasets/mimic/mimic3/train_full.csv \
		/mnt/subverse/datasets/mimic/mimic3/vocab.csv \
		full \
		cnn_vanilla \
		100 \
		--clf fft \
		--k 80 \
		--slack-dims 200 \
		--filter-size 4 \
		--num-filter-maps $FMAPS \
		--dropout $DROPOUT \
		--lr $LR \
		--embed-file /mnt/subverse/datasets/mimic/mimic3/processed_full.embed \
		--batch-size $BS \
		--patience 10 \
		--criterion prec_at_8 \
		--seed $SEED \
		--gpu > logs/DFT-D-200-S-$SEED

	python caml/learn/training.py \
		/mnt/subverse/datasets/mimic/mimic3/train_full.csv \
		/mnt/subverse/datasets/mimic/mimic3/vocab.csv \
		full \
		cnn_vanilla \
		100 \
		--clf fft \
		--k 80 \
		--slack-dims 400 \
		--filter-size 4 \
		--num-filter-maps $FMAPS \
		--dropout $DROPOUT \
		--lr $LR \
		--embed-file /mnt/subverse/datasets/mimic/mimic3/processed_full.embed \
		--batch-size $BS \
		--patience 10 \
		--criterion prec_at_8 \
		--seed $SEED \
		--gpu > logs/DFT-D-400-S-$SEED

done
