#!/usr/bin/env bash

#declare -a datasets=("polblogs" "flylarva" "pyg_email_eu_core" "pyg_cora_ml" "enron_na" "wikivotes" "pubmed")
declare -a datasets=("pyg_cora_ml")

for d in "${datasets[@]}"
do
    echo "$(date) Starting dataset $d"
    python experiments/network_alignment.py --dataroot ./data --resultdir ./neb_gcp_results --tempdir /tmp --dataset $d --methods reachnes_ga_comparison --undirected 0 --noise-p 0.15 --num-reps 5 --node-attributed 0 --pp-mode whiten --seed 23423
done
