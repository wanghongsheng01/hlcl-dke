# HLCL: label-level hierarchical selective acceptance of LLM annotations

Reproducibility package for the paper *Label-Level Hierarchical Selective
Acceptance of LLM Annotations with Certified Error Control*.

## Contents
| Path | What it is |
|---|---|
| `code/hlcl2.py` | acceptance rule, pooled betting test, depth-wise binomial IUT, learn-then-test calibration, effort model |
| `code/features.py` | consensus paths and node features from the cached LLM samples; logistic node score |
| `code/experiment2.py` | 100-split experiment loop, all baselines and ablations |
| `code/annotate.py` | the LLM annotation script (OpenAI-compatible API) that produced the cache |
| `code/datasets_local.py` | loaders for the three corpora with the fixed 3,000-item subsample (seed 2026) |
| `code/make_tables.py`, `code/make_figures.py` | regenerate every table, figure and number in the paper |
| `data/llm_cache/*.jsonl` | all 60,000 cached LLM responses (item index, sample index, raw JSON answer) |
| `results/v8_*.json` | per-split results for every method, dataset, annotator and alpha |

## Source corpora (not redistributed)
Place them under `data/` as follows.
* DBPedia classes: `DBPEDIA_val.csv` from https://huggingface.co/datasets/DeveloperOats/DBPedia_Classes
* WOS-46985: Mendeley Data `9rw3vkcfy4` (version 6); unzip to `data/wos/` (needs `Meta-data/Data.xlsx`)
* Blurb Genre Collection (EN), Universität Hamburg, CC BY-NC 4.0; unzip to `data/bgc/`
  (needs `BlurbGenreCollection_EN_test.txt` and `hierarchy.txt`)

## Reproduce
```bash
pip install -r requirements.txt
cd code
python experiment2.py --dataset bgc --tag bgc_deepseek_default_k5_t1.0 --R 100 --alphas 0.02,0.05,0.1,0.15 --levels 8 --out v8_bgc
python make_tables.py v8 && python make_figures.py v8
```
`results/run_v8.sh` runs all settings in the paper. No API key is needed: the
experiments read the cached responses. To re-annotate, set the API key in
`~/.config/rp/llm.env` and run `annotate.py`.

## Licence
Code: MIT. Cached responses: CC BY 4.0. The source corpora keep their own licences.
