#!/usr/bin/env bash

#declare -a datasets=("flylarva" "polblogs" "cocite" "subelj_cora" "pubmed" "pyg_email_eu_core")
declare -a datasets=("flylarva")

for d in "${datasets[@]}"
do
    echo "$(date) Starting dataset $d"
    python experiments/node_classification.py --dataroot ./data --resultdir ./neb_gcp_results --tempdir /tmp --dataset $d --methods reachnes_dirswitch_compare_rsvd --undirected 0 --weighted 0 --node-attributed 0 --seed 89932 --timeout 1000 --clf log_reg_sgd --num-reps 1 --pp-mode none
done