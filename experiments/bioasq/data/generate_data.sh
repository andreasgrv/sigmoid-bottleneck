#!/usr/bin/env bash
# spmlbl.create_contrained_docs --examples train-1m.json  --max-unique-labels 20000 --train-size 100000 --valid-size 10000 --out-folder subsets-v-20000

for jf in subsets-v-20000/*.json
do
	mlbl.extract_document_embeddings  --examples $jf --outfile $jf --blueprint ../blueprints/sigmoid-bottleneck.yaml
done
