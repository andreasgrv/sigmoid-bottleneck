#!/usr/bin/env bash
export PYTHONPATH="$PWD"
FMAPS=500
DROPOUT=0.2
BS=16
LR=0.001

for SEED in 0 1 2
do
	python learn/training.py \
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
		--gpu

	python learn/training.py \
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
		--gpu

	python learn/training.py \
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
		--gpu

	# D = 200
	python learn/training.py \
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
		--gpu

	# D = 500
	python learn/training.py \
		/mnt/subverse/datasets/mimic/mimic3/train_full.csv \
		/mnt/subverse/datasets/mimic/mimic3/vocab.csv \
		full \
		cnn_vanilla \
		100 \
		--clf bottleneck \
		--k 0 \
		--slack-dims 500 \
		--filter-size 4 \
		--num-filter-maps $FMAPS \
		--dropout $DROPOUT \
		--lr $LR \
		--embed-file /mnt/subverse/datasets/mimic/mimic3/processed_full.embed \
		--batch-size $BS \
		--patience 10 \
		--criterion prec_at_8 \
		--seed $SEED \
		--gpu

	# VANDER FFT LAYERS
	python learn/training.py \
		/mnt/subverse/datasets/mimic/mimic3/train_full.csv \
		/mnt/subverse/datasets/mimic/mimic3/vocab.csv \
		full \
		cnn_vanilla \
		100 \
		--clf fft \
		--k 70 \
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
		--gpu

	python learn/training.py \
		/mnt/subverse/datasets/mimic/mimic3/train_full.csv \
		/mnt/subverse/datasets/mimic/mimic3/vocab.csv \
		full \
		cnn_vanilla \
		100 \
		--clf fft \
		--k 70 \
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
		--gpu

	python learn/training.py \
		/mnt/subverse/datasets/mimic/mimic3/train_full.csv \
		/mnt/subverse/datasets/mimic/mimic3/vocab.csv \
		full \
		cnn_vanilla \
		100 \
		--clf fft \
		--k 70 \
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
		--gpu

	python learn/training.py \
		/mnt/subverse/datasets/mimic/mimic3/train_full.csv \
		/mnt/subverse/datasets/mimic/mimic3/vocab.csv \
		full \
		cnn_vanilla \
		100 \
		--clf fft \
		--k 70 \
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
		--gpu

	python learn/training.py \
		/mnt/subverse/datasets/mimic/mimic3/train_full.csv \
		/mnt/subverse/datasets/mimic/mimic3/vocab.csv \
		full \
		cnn_vanilla \
		100 \
		--clf fft \
		--k 70 \
		--slack-dims 500 \
		--filter-size 4 \
		--num-filter-maps $FMAPS \
		--dropout $DROPOUT \
		--lr $LR \
		--embed-file /mnt/subverse/datasets/mimic/mimic3/processed_full.embed \
		--batch-size $BS \
		--patience 10 \
		--criterion prec_at_8 \
		--seed $SEED \
		--gpu
done
