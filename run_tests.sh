#!/bin/bash

powers=(60 65 70)

echo "" > results.txt
for p in "${powers[@]}"
do
    echo "\n** Power limit $p W ** " >> results.txt
    date >> results.txt
    sudo nvidia-smi -i 0 -pl "$p"
    python3 run_classifiers.py >> results.txt
done

