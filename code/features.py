"""Hierarchy-aware node features and a learned acceptance score.

For every item the k LLM samples define, at each depth, a distribution over
candidate nodes conditioned on the consensus prefix. The consensus node at
depth d is the mode of that conditional distribution. Raw vote scores are
non-increasing along a path, which makes ancestor closure a no-op; a learned
score that also looks at sibling competition and verbal confidence is not
monotone, so the hierarchy constraint becomes an active part of the rule.

Features of a consensus node v at depth d:
  vote        share of the k samples whose path agrees with the prefix up to v
  cond        vote(v) / vote(parent)   (conditional agreement at this level)
  margin      conditional share of v minus that of its strongest sibling
  ent         entropy of the conditional sibling distribution
  verb, verbsd  mean / std of verbalised confidence over supporting samples
  pvote       vote of the parent (1 at depth 0)
  fan         log fan-out of the parent in the taxonomy
  depth one-hot
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression

from annotate import CACHE, parse
from datasets_local import load_dataset_items
from hlcl import Annotated


def _fanout(tree, prefix):
    node = tree
    for p in prefix:
        if isinstance(node, dict):
            node = node.get(p, {})
        else:
            return 1
    return len(node) if node else 1


def build(dataset: str, tag: str, n: int, k: int):
    items, tree = load_dataset_items(dataset, n)
    depth = len(items[0]["gold"])
    samples: dict[int, list] = {}
    for line in (CACHE / f"{tag}.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["s"] < k:
            samples.setdefault(r["id"], []).append(parse(r["resp"], depth))

    rows, item, dep, correct, parent = [], [], [], [], []
    n_invalid = 0
    for it in items:
        ss = [s for s in samples.get(it["id"], []) if s is not None]
        n_invalid += k - len(ss)
        prefix: list[str] = []
        prev, pvote = -1, 1.0
        for d in range(depth):
            agree = [s for s in ss if s[0][:d] == prefix]
            if not agree:
                break
            cnt = Counter(s[0][d] for s in agree)
            (lab, c1), *rest = cnt.most_common()
            c2 = rest[0][1] if rest else 0
            pr = np.array(list(cnt.values()), float) / len(agree)
            sup = [s for s in agree if s[0][d] == lab]
            vote = len(sup) / k
            vb = np.array([s[1][d] for s in sup])
            oh = [0.0] * depth
            oh[d] = 1.0
            rows.append([vote, vote / max(pvote, 1e-9), (c1 - c2) / len(agree),
                         float(-(pr * np.log(pr)).sum()), vb.mean(), vb.std(),
                         pvote, np.log(_fanout(tree, prefix)), *oh])
            prefix = prefix + [lab]
            item.append(it["id"]); dep.append(d)
            correct.append(int(it["gold"][: d + 1] == prefix))
            parent.append(prev)
            prev, pvote = len(item) - 1, vote
    X = np.array(rows)
    a = Annotated(np.array(item), np.array(dep), X[:, 0] + 0.01 * X[:, 4],
                  np.array(correct), np.array(parent), len(items))
    meta = dict(n_items=len(items), depth=depth, invalid_samples=n_invalid,
                n_nodes=len(item),
                acc_by_depth=[float(np.mean(a.correct[a.depth == d])) for d in range(depth)])
    return a, X, meta


def fit_score(X_fit, y_fit, X_all):
    """Logistic score fitted on a disjoint split (no leakage into cal/test)."""
    mu, sd = X_fit.mean(0), X_fit.std(0) + 1e-9
    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit((X_fit - mu) / sd, y_fit)
    return clf.predict_proba((X_all - mu) / sd)[:, 1]
