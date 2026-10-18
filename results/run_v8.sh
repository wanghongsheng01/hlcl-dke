#!/bin/sh
# v8 final runs: IUT treats an empty depth as non-rejectable; c0 sensitivity.
cd "$(dirname "$0")/../code" || exit 1
PY=../.venv/bin/python
run() { $PY experiment2.py "$@" ; }
run --dataset dbpedia --tag dbpedia_deepseek_default_k5_t1.0 --R 100 --alphas 0.01,0.02,0.05,0.1 --levels 8 --out v8_dbpedia > ../results/v8_dbpedia.log 2>&1; echo "exit=$?" >> ../results/v8_dbpedia.log
run --dataset wos --tag wos_deepseek_default_k5_t1.0 --R 100 --alphas 0.1,0.15,0.2,0.3 --levels 8 --out v8_wos > ../results/v8_wos.log 2>&1; echo "exit=$?" >> ../results/v8_wos.log
run --dataset bgc --tag bgc_deepseek_default_k5_t1.0 --R 100 --alphas 0.02,0.05,0.1,0.15 --levels 8 --out v8_bgc > ../results/v8_bgc.log 2>&1; echo "exit=$?" >> ../results/v8_bgc.log
run --dataset bgc --tag bgc_alt_claude-haiku-4-5_k5_t1.0 --R 100 --alphas 0.02,0.05,0.1,0.15 --levels 8 --out v8_bgc_haiku > ../results/v8_bgc_haiku.log 2>&1; echo "exit=$?" >> ../results/v8_bgc_haiku.log
{ run --dataset bgc --tag bgc_deepseek_default_k5_t1.0 --R 100 --alphas 0.05,0.1 --levels 8 --c0 0 --out v8_bgc_c0_0; echo "exit=$?";
  run --dataset bgc --tag bgc_deepseek_default_k5_t1.0 --R 100 --alphas 0.05,0.1 --levels 8 --c0 1 --out v8_bgc_c0_1; echo "exit=$?"; } > ../results/v8_c0.log 2>&1
