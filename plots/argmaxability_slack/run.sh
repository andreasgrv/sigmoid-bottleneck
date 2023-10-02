#!/usr/bin/env bash

# N_SAMPLES=20000
#
# N_LINES=2
# for N in 50 100 200 500 1000 2000 5000 10000
# do
# 	for D in 2 4 6 8 10
# 	do
# 		OUT=$(python check_infeasible_cyclic.py --N $N --D $D --num-samples $N_SAMPLES)
# 		echo "$OUT" | tail -n $N_LINES
# 		N_LINES=1
# 	done
# done
#
# SLACK_SOURCE=random
# # SLACK_SOURCE=spread-freq
# # N_LINES=2
# for S in 2 4 8 16
# do
# 	for N in 50 100 200 500 1000 2000 5000 10000
# 	do
# 		for D in 2 4 6 8 10
# 		do
# 			OUT=$(python check_infeasible_cyclic.py --N $N --D $D --num-samples $N_SAMPLES --slack-dim $S --slack-source $SLACK_SOURCE)
# 			echo "$OUT" | tail -n $N_LINES
# 			N_LINES=1
# 		done
# 	done
# done

for C in 1 5 10 25 50 75 90 95 99
do
	python plot.py --data low-cardinality-slack.csv --prop rad_p_$C --save
done

SLACK_SOURCE=random

for S in 2 4 8 16 32
do
	for C in 1 5 10 25 50 75 90 95 99
	do
		python plot.py --data low-cardinality-slack.csv --prop rad_p_$C --slack-dims $S --slack-source $SLACK_SOURCE --save
	done
done

python table.py --data low-cardinality-slack.csv

for S in 2 4 8 16 32
do
	python table.py --data low-cardinality-slack.csv --slack-dims $S --slack-source $SLACK_SOURCE
done

# for each in images/*
# do
# 	readlink -f $each
# done
