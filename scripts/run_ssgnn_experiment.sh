#!/usr/bin/env bash

#declare -a datasets=("roman_empire" "pyg_cora_ml" "pyg_citeseer" "pyg_cora" "ogb_arxiv_year" "ogb_arxiv" "pokec_gender" "snap_patents")
declare -a datasets=("pyg_cora_ml")
declare -a algs=("ccassg_dir_comparison_graphsage" "graphmae2_dir_comparison_graphsage")

for a in "${algs[@]}"
do
    for d in "${datasets[@]}"
    do
        echo "$(date) Starting $a for dataset $d"
        python experiments/node_classification.py --dataroot ./data --resultdir ./neb_gcp_results --tempdir /tmp --dataset $d --methods $a --undirected 0 --weighted 0 --node-attributed 1 --seed 89932 --timeout 15000 --clf log_reg_sgd --num-reps 5 --pp-mode standardize --dims 512 --num-epochs 100
    done
done
