#!/bin/sh
# Fourth corpus (Amazon Reviews 2023, Home & Kitchen, 4 levels): main run with
# ablations, then Item-cert. under each fixed test on the same splits.
# levels=6 per depth keeps the per-depth grid at 7^4 candidates.
cd "$(dirname "$0")/../code" || exit 1
PY=../.venv/bin/python
A="--dataset amazon --tag amazon_deepseek_default_k5_t1.0 --R 100 --alphas 0.02,0.05,0.1,0.15 --levels 6"
$PY experiment2.py $A --out v9_amazon > ../results/v9_amazon.log 2>&1; echo "exit=$?" >> ../results/v9_amazon.log
$PY experiment2.py $A --item-only --out v9item_amazon > ../results/v9item_amazon.log 2>&1; echo "exit=$?" >> ../results/v9item_amazon.log
