#!/usr/bin/env bash

for blueprint in experiments/*/blueprint.yaml
do
	echo $blueprint
	python eval.py  --eval-type test --blueprint $blueprint
done
