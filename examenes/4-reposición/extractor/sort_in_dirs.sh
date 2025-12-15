#!/bin/bash

## Number of questions
E=`ls -1 ../examen_a/pregunta*_*.tex | sed -e "s/.*pregunta[^0-9]*\([0-9]*\)[^0-9]*.tex/\1/g" | wc -w`
## Quiz number
EN=`ls -1 ../examen_a | xargs basename | sed -e "s/examen_\([0-9]*\).tex/\1/" `
EP=E${EN}_P

for i in $(seq 1 $E); do
    DIR=${EP}$(printf "%02d" $i)
    echo $DIR

    mkdir -p $DIR
    mv ${DIR}_*.png $DIR
done
