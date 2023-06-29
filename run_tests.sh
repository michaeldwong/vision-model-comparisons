#!/bin/bash

powers=(60 65 70)

touch results.txt
for p in "${powers[@]}"
do
    echo "\n** Power limit $p W ** " >> results.txt
    sudo nvidia-smi -i 0 -pl "$p"
    python3 run_classifiers >> results.txt
done

