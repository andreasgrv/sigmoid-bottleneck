#!/usr/bin/env bash

for blueprint in experiments/*/blueprint.yaml
do
	echo $blueprint
	python eval.py  --blueprint $blueprint
done
