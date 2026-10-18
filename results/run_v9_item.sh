#!/bin/sh
# Item-cert. under each fixed test, on the same 100 splits as v9 (merged by merge_item.py)
cd "$(dirname "$0")/../code" || exit 1
PY=../.venv/bin/python
run() { out=$1; shift; $PY experiment2.py "$@" --item-only --out $out > ../results/$out.log 2>&1; echo "exit=$?" >> ../results/$out.log; }
run v9item_dbpedia   --dataset dbpedia --tag dbpedia_deepseek_default_k5_t1.0 --R 100 --alphas 0.01,0.02,0.05,0.1 --levels 8
run v9item_wos       --dataset wos --tag wos_deepseek_default_k5_t1.0 --R 100 --alphas 0.1,0.15,0.2,0.3 --levels 8
run v9item_bgc       --dataset bgc --tag bgc_deepseek_default_k5_t1.0 --R 100 --alphas 0.02,0.05,0.1,0.15 --levels 8
run v9item_bgc_haiku --dataset bgc --tag bgc_alt_claude-haiku-4-5_k5_t1.0 --R 100 --alphas 0.02,0.05,0.1,0.15 --levels 8
