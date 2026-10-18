# Amazon Reviews 2023 — Home & Kitchen (4-level subset)

Source: McAuley Lab, Amazon Reviews 2023, product metadata
`raw/meta_categories/meta_Home_and_Kitchen.jsonl`
(https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).

Construction (reproduced by `code/build_amazon.py`):
1. Read the first 400,000 records of the file.
2. Keep products whose `categories` list starts with "Home & Kitchen" and has at
   least 5 entries; use entries 2–5 as a 4-level path (the root is dropped).
3. Keep paths with at least 100 products (154,787 products; 8 / 31 / 112 / 296
   labels at levels 1–4).
4. Text = title + first 500 characters of features + first 400 characters of
   description, truncated to 1,200 characters.
5. 3,000 products are sampled with seed 2026 (as for the other corpora).
