"""Loaders for public hierarchical label datasets (fixed-seed subsamples)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"


def _dbpedia():
    # official validation split (11,152 items; 9 / 70 / 219 labels at l1 / l2 / l3)
    df = pd.read_csv(DATA / "DBPEDIA_val.csv").dropna().reset_index(drop=True)
    tree: dict = {}
    for a, b, c in df[["l1", "l2", "l3"]].drop_duplicates().itertuples(index=False):
        tree.setdefault(a, {}).setdefault(b, []).append(c)
    for a in tree:
        for b in tree[a]:
            tree[a][b] = sorted(tree[a][b])
    df = df.rename(columns={"l1": "y1", "l2": "y2", "l3": "y3"})
    return df, tree


def _wos():
    data = pd.read_excel(DATA / "wos" / "Meta-data" / "Data.xlsx")
    df = pd.DataFrame({"text": data["Abstract"].astype(str),
                       "y1": data["Domain"].str.strip(),
                       "y2": data["area"].str.strip()})
    tree: dict = {}
    for a, b in df[["y1", "y2"]].drop_duplicates().itertuples(index=False):
        tree.setdefault(a, []).append(b)
    for a in tree:
        tree[a] = sorted(set(tree[a]))
    return df, tree


def _bgc():
    """Blurb Genre Collection (EN, official test split). We keep blurbs with a
    single genre path of depth >= 3 and use its first three levels."""
    import re
    t = (DATA / "bgc" / "BlurbGenreCollection_EN_test.txt").read_text(encoding="utf-8")
    rows = []
    for book in re.findall(r"<book.*?</book>", t, re.S):
        tags = re.findall(r"<d(\d)>(.*?)</d\d>", re.search(r"<topics>(.*?)</topics>", book, re.S).group(1))
        lv = {}
        for d, g in tags:
            lv.setdefault(int(d), []).append(g.strip())
        if any(len(v) > 1 for v in lv.values()) or len(lv) < 3:
            continue
        body = re.search(r"<body>(.*?)</body>", book, re.S).group(1).strip()
        title = re.search(r"<title>(.*?)</title>", book, re.S).group(1).strip()
        rows.append((f"{title}. {body}", lv[0][0], lv[1][0], lv[2][0]))
    df = pd.DataFrame(rows, columns=["text", "y1", "y2", "y3"])
    tree: dict = {}
    for a, b, c in df[["y1", "y2", "y3"]].drop_duplicates().itertuples(index=False):
        tree.setdefault(a, {}).setdefault(b, []).append(c)
    # offer every child in the official hierarchy, not only those observed
    par: dict = {}
    for line in (DATA / "bgc" / "hierarchy.txt").read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            p, c = line.split("\t")
            par.setdefault(p.strip(), set()).add(c.strip())
    for a in tree:
        for b in list(par.get(a, [])):
            tree[a].setdefault(b, [])
        for b in tree[a]:
            tree[a][b] = sorted(set(tree[a][b]) | par.get(b, set())) or [b]
    return df, tree


LOADERS = {"dbpedia": _dbpedia, "wos": _wos, "bgc": _bgc}


def load_dataset_items(name: str, n: int, seed: int = 2026):
    df, tree = LOADERS[name]()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    sub = df.iloc[np.sort(idx)].reset_index(drop=True)
    ycols = [c for c in ("y1", "y2", "y3") if c in sub]
    items = [{"id": int(i), "text": t, "gold": [r[c] for c in ycols]}
             for i, (t, r) in enumerate(zip(sub["text"], sub.to_dict("records")))]
    return items, tree
