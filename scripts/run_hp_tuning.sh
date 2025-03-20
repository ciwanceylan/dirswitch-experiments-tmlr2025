#!/usr/bin/env bash
declare -a algs=("reachnes_rsvd_nc_hp" "nerd_hp_search" "app_hp_search" "hope_hp_search" "blade_hp_search" "dggan_hp_search")

#declare -a datasets=("pubmed" "cocite" "polblogs" "flylarva" "subelj_cora" "pyg_email_eu_core")
declare -a datasets=("flylarva")

mkdir -p "nc_hp_logs"
for a in "${algs[@]}"
do
    for d in "${datasets[@]}"
    do
        echo "$(date) Starting $a for dataset $d"
        python experiments/node_classification.py --dataroot ./data --resultdir ./neb_gcp_results --tempdir /tmp --dataset $d --methods $a --undirected 0 --weighted 0 --node-attributed 0 --seed 8453234 --timeout 3600 --clf log_reg_sgd --num-reps 1 --pp-mode standardize
    done
done
