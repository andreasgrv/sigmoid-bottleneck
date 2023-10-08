#!/usr/bin/env bash
for MODEL in experiments/*/*.pth
do
	python eval.py --test-file dev_full.csv  --model $MODEL
	python eval.py --test-file test_full.csv  --model $MODEL
done
