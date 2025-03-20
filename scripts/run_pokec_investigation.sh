#!/usr/bin/env bash

python experiments/node_classification.py --dataroot ./data --resultdir ./neb_gcp_results --tempdir /tmp --dataset pokec_gender --methods reachnes_pokec_investigation --undirected 0 --weighted 0 --node-attributed 1 --seed 89923432 --timeout 15000 --clf log_reg_sgd --num-reps 3 --pp-mode standardize
