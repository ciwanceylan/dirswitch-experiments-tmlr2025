#!/usr/bin/env bash

#declare -a datasets=("roman_empire" "pyg_citeseer" "pyg_cora_ml" "pyg_pubmed" "pyg_cora" "ogb_arxiv" "ogb_arxiv_year" "pokec_gender" "snap_patents")
declare -a datasets=("pyg_cora_ml")

for d in "${datasets[@]}"
do
    echo "$(date) Starting dataset $d"
    python experiments/node_classification.py --dataroot ./data --resultdir ./neb_gcp_results --tempdir /tmp --dataset $d --methods reachnes_dirswitch_compare_message_passing --undirected 0 --weighted 0 --node-attributed 1 --seed 89932 --timeout 500 --clf log_reg_sgd --num-reps 1 --pp-mode standardize
done