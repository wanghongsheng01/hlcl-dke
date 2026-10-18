"""Rebuild data/amazon/amazon_hk_l4_min100.json from the Amazon Reviews 2023
Home & Kitchen metadata (see data/amazon/README.md)."""
import collections
import json
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "amazon"
src = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "meta_Home_and_Kitchen.jsonl"
rows = []
with open(src) as f:
    for n, line in enumerate(f):
        if n >= 400_000:
            break
        d = json.loads(line)
        c = d.get("categories") or []
        t = (d.get("title") or "").strip()
        if len(c) >= 5 and t and c[0] == "Home & Kitchen":
            feat = " ".join(d.get("features") or [])[:500]
            desc = " ".join(d.get("description") or [])[:400]
            rows.append({"text": (t + ". " + feat + " " + desc).strip()[:1200],
                         "path": c[1:5], "asin": d.get("parent_asin")})
pc = collections.Counter(tuple(r["path"]) for r in rows)
keep = {p for p, k in pc.items() if k >= 100}
out = [r for r in rows if tuple(r["path"]) in keep]
(DATA / "amazon_hk_l4_min100.json").write_text(json.dumps(out))
print(len(out), "products")
