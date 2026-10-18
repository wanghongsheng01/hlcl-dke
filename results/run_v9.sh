#!/bin/sh
# v9 (major revision): Item-cert. gets the same test-selection rule; Conformal
# Labeling and the fixed rule use the learned score; per-field grid from the
# fit split; ordering/test choice on in-sample fit-split scores (an out-of-fold
# variant exists behind --oof and was worse in a pilot); fixed-test
# ablations; HSC baseline; calibration-size and delta sensitivity.
cd "$(dirname "$0")/../code" || exit 1
PY=../.venv/bin/python
run() { out=$1; shift; $PY experiment2.py "$@" --out $out > ../results/$out.log 2>&1; echo "exit=$?" >> ../results/$out.log; }
run v9_dbpedia   --dataset dbpedia --tag dbpedia_deepseek_default_k5_t1.0 --R 100 --alphas 0.01,0.02,0.05,0.1 --levels 8
run v9_wos       --dataset wos --tag wos_deepseek_default_k5_t1.0 --R 100 --alphas 0.1,0.15,0.2,0.3 --levels 8
run v9_bgc       --dataset bgc --tag bgc_deepseek_default_k5_t1.0 --R 100 --alphas 0.02,0.05,0.1,0.15 --levels 8
run v9_bgc_haiku --dataset bgc --tag bgc_alt_claude-haiku-4-5_k5_t1.0 --R 100 --alphas 0.02,0.05,0.1,0.15 --levels 8
for c0 in 0 1; do run v9_bgc_c0_$c0 --dataset bgc --tag bgc_deepseek_default_k5_t1.0 --R 100 --alphas 0.05,0.1 --levels 8 --c0 $c0 --light; done
for nc in 300 600; do run v9_bgc_ncal$nc --dataset bgc --tag bgc_deepseek_default_k5_t1.0 --R 100 --alphas 0.05,0.1 --levels 8 --ncal $nc --light; done
for d in 0.05 0.2; do run v9_bgc_delta$d --dataset bgc --tag bgc_deepseek_default_k5_t1.0 --R 100 --alphas 0.05,0.1 --levels 8 --delta $d --light; done
