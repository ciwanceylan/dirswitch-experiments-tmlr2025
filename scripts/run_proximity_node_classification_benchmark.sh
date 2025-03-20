#!/usr/bin/env bash

declare -a algs=("reachnes_rsvd" "blade" "hope" "nerd" "app" "dggan")
#declare -a datasets=("pubmed" "cocite" "flylarva" "subelj_cora" "polblogs" "pyg_email_eu_core")
declare -a datasets=("flylarva")

for a in "${algs[@]}"
do
    for d in "${datasets[@]}"
    do
        echo "$(date) Starting $a for dataset $d"
        python experiments/node_classification.py --dataroot ./data --resultdir ./neb_gcp_results --tempdir /tmp --dataset $d --methods $a --undirected 0 --weighted 0 --node-attributed 0 --seed 56342234 --timeout 3600 --clf log_reg_sgd --num-reps 5 --pp-mode none --best-hps
    done
done
