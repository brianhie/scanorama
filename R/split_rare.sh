#!/bin/bash

for i in {0..9}
do
    printf "Genes\t" > simulate_rare_"$i".txt
    head -n1 simulate_rare_full.txt >> simulate_rare_"$i".txt
    tail -n+2 simulate_rare_full.txt | tail -n+$((i * 1000)) | head -n1000 >> simulate_rare_"$i".txt
    sed -i 's/ /\t/g' simulate_rare_"$i".txt
    sed -i 's/"//g' simulate_rare_"$i".txt
    cat simulate_rare_"$i".txt | datamash transpose > temp
    mv temp simulate_rare_"$i".txt
done
