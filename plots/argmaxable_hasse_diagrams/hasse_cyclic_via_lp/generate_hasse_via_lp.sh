#!/usr/bin/env bash

N=10
for C in 1 2 3 4
do
	python feasible_hasse_via_lp.py --N $N --C $C | grep -vwE "Academic" | grep -vwE "Set parameter Username" > hasse-n-$N-c-$C.tex
 
	xelatex hasse-n-$N-c-$C.tex
done

N=8
for C in 1 2 3
do
	python feasible_hasse_via_lp.py --N $N --C $C | grep -vwE "Academic" | grep -vwE "Set parameter Username" > hasse-n-$N-c-$C.tex
	xelatex hasse-n-$N-c-$C.tex
done
