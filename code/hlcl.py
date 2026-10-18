"""HLCL: hierarchy-aware, label-level conformal acceptance of model annotations.

Setting. Each item i carries a set of predicted labels (nodes of a label tree),
each with a confidence score s in [0, 1]. A predicted node is *correct* if it
belongs to the item's gold label set (gold sets are ancestor-closed).

Decision. For every predicted leaf-level label we may
  (a) accept it,
  (b) back off: accept only its deepest ancestor whose score clears that
      depth's threshold (the remaining finer labels go to a human), or
  (c) route the whole path to a human.
Acceptance of a node is a per-depth threshold rule: node v at depth d is
accepted iff s(v) >= lam[d].

Guarantee. Let FDP(lam) = (#accepted wrong nodes) / max(1, #accepted nodes).
We pick lam with Learn-then-Test (Angelopoulos et al., 2021): every candidate
threshold vector is a null hypothesis H_lam: E[FDP(lam)] > alpha (we use the
ratio-of-sums form, i.e. label-level FDR over the test population), tested on
calibration data with a Hoeffding-Bentkus style p-value on the per-item loss
L_i(lam) = wrong_i(lam) - alpha * accepted_i(lam), whose mean is <= 0 iff the
population-level ratio wrong/accepted is <= alpha. Candidates are ordered from
most to least conservative per depth and tested with fixed-sequence testing
(FWER control), so every selected lam satisfies, with probability >= 1 - delta,
  E[wrong(lam)] <= alpha * E[accepted(lam)].
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.stats import binom

# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------


@dataclass
class Annotated:
    """Flat representation of all predicted nodes over a set of items.

    item  : item index of each predicted node
    depth : depth of the node (0 = top level)
    score : confidence in [0, 1]
    correct: 1 if node is in the gold set
    parent: index (into these arrays) of the predicted parent node, -1 at depth 0
    """
    item: np.ndarray
    depth: np.ndarray
    score: np.ndarray
    correct: np.ndarray
    parent: np.ndarray
    n_items: int
    n_gold: np.ndarray | None = None   # gold label count per item (for recall)

    def subset(self, items: np.ndarray) -> "Annotated":
        keep_items = np.zeros(self.n_items, bool)
        keep_items[items] = True
        m = keep_items[self.item]
        new_idx = -np.ones(len(m), int)
        new_idx[m] = np.arange(m.sum())
        remap = -np.ones(self.n_items, int)
        remap[items] = np.arange(len(items))
        par = self.parent[m]
        par = np.where(par >= 0, new_idx[np.maximum(par, 0)], -1)
        return Annotated(remap[self.item[m]], self.depth[m], self.score[m],
                         self.correct[m], par, len(items),
                         None if self.n_gold is None else self.n_gold[items])


def accepted_mask(a: Annotated, lam: np.ndarray, hier: bool = True) -> np.ndarray:
    """Nodes accepted under per-depth thresholds lam.

    hier=True enforces ancestor closure: a node is accepted only if its own
    score clears lam[depth] AND its parent is accepted. This yields the
    back-off behaviour: a path is accepted down to the deepest node that
    clears its threshold, deeper labels go to a human.
    """
    own = a.score >= lam[np.minimum(a.depth, len(lam) - 1)]
    if not hier:
        return own
    acc = own.copy()
    order = np.argsort(a.depth, kind="stable")
    for j in order:
        p = a.parent[j]
        if p >= 0 and not acc[p]:
            acc[j] = False
    return acc


def per_item_counts(a: Annotated, acc: np.ndarray):
    n_acc = np.bincount(a.item, weights=acc, minlength=a.n_items)
    n_wrong = np.bincount(a.item, weights=acc & (a.correct == 0), minlength=a.n_items)
    return n_acc, n_wrong


# --------------------------------------------------------------------------
# Learn-then-Test calibration
# --------------------------------------------------------------------------


def hb_pvalue(losses: np.ndarray, bound: float) -> float:
    """Hoeffding-Bentkus p-value for H0: E[loss] > 0 with loss in [-a, b].

    losses are rescaled to [0, 1]; H0 becomes E[z] > t with t = a/(a+b).
    """
    n = len(losses)
    lo, hi = -bound[0], bound[1]
    z = (losses - lo) / (hi - lo)
    t = (0 - lo) / (hi - lo)
    zbar = z.mean()
    if zbar >= t:
        return 1.0
    # Hoeffding
    h1 = np.exp(-n * _kl(min(zbar, t), t)) if zbar < t else 1.0
    # Bentkus
    h2 = np.e * binom.cdf(np.ceil(n * zbar), n, t)
    return float(min(1.0, h1, h2))


def _kl(p, q):
    eps = 1e-12
    p = min(max(p, eps), 1 - eps)
    q = min(max(q, eps), 1 - eps)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def wsr_pvalue(losses: np.ndarray, bound) -> float:
    """Variance-adaptive betting p-value (Waudby-Smith & Ramdas, 2024).

    Tests H0: E[x] >= t for x in [0, 1] with the capital process
    K_n = prod_i (1 + lam_i (t - x_i)); lam_i is predictable and in
    [0, 0.5/(1-t)], so every factor is positive. By Ville's inequality
    p = 1 / max_n K_n is a valid p-value. Unlike Hoeffding-Bentkus it adapts to
    the (very small) variance of the per-item loss at conservative thresholds.
    """
    lo, hi = -bound[0], bound[1]
    x = np.clip((losses - lo) / (hi - lo), 0, 1)
    t = (0 - lo) / (hi - lo)
    n = len(x)
    x = x[np.random.default_rng(12345).permutation(n)]
    i = np.arange(1, n + 1)
    mu = (0.5 + np.cumsum(x)) / (i + 1)
    var = (0.25 + np.cumsum((x - mu) ** 2)) / (i + 1)
    var_prev = np.concatenate([[0.25], var[:-1]])
    lam = np.sqrt(2 * np.log(2 / 0.1) / (n * var_prev))
    lam = np.minimum(lam, 0.5 / (1 - t))
    logk = np.cumsum(np.log1p(lam * (t - x)))
    return float(min(1.0, np.exp(-logk.max())))


def pvalue(losses, bound):
    # both are valid; taking the min of two valid p-values is not, so we fix
    # one in advance: the betting p-value is much tighter here
    return wsr_pvalue(losses, bound)


def ltt_calibrate(cal: Annotated, grid: list[np.ndarray], alpha: float,
                  delta: float = 0.1, hier: bool = True,
                  max_items: float | None = None, max_labels_per_item: int = 50,
                  order_frac: float = 0.3, seed: int = 0):
    """Learn-then-Test with a data-driven fixed-sequence ordering.

    The calibration set is split: on the ordering part (order_frac) we compute
    a p-value for every candidate and sort candidates by it; on the remaining
    part we run fixed-sequence testing along that order (stop at the first
    non-rejection). Since the order is independent of the testing part, the
    set of rejected candidates controls FWER at delta, so every certified
    threshold vector has label-level FDR <= alpha with prob. >= 1 - delta.
    Among certified candidates (that also meet the optional human budget) we
    return the one that accepts the most labels on the testing part.
    """
    b = (alpha * max_labels_per_item, (1 - alpha) * max_labels_per_item)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(cal.n_items)
    k = int(order_frac * cal.n_items)
    ordp, test = cal.subset(np.sort(perm[:k])), cal.subset(np.sort(perm[k:]))

    def stats(a, lam):
        acc = accepted_mask(a, lam, hier)
        n_acc, n_wrong = per_item_counts(a, acc)
        return n_acc, n_wrong

    scores = []
    for lam in grid:
        n_acc, n_wrong = stats(ordp, lam)
        p = pvalue(np.clip(n_wrong - alpha * n_acc, -b[0], b[1]), b)
        scores.append((p, -n_acc.sum()))
    order = sorted(range(len(grid)), key=lambda j: scores[j])
    best, best_acc = None, -1
    n_pred = np.bincount(test.item, minlength=test.n_items)
    for j in order:
        n_acc, n_wrong = stats(test, grid[j])
        p = pvalue(np.clip(n_wrong - alpha * n_acc, -b[0], b[1]), b)
        if p > delta:
            break
        if max_items is not None and (n_acc < n_pred).mean() > max_items:
            continue
        if n_acc.sum() > best_acc:
            best, best_acc = grid[j], n_acc.sum()
    return best


def make_grid(depth: int, q: np.ndarray, tie: bool = False):
    """Fixed-sequence ordering of per-depth thresholds.

    q: decreasing threshold values (most conservative first).
    tie=True: all depths share one threshold (single-parameter family).
    Otherwise the grid follows a monotone path: we lower the threshold of the
    shallowest depth first (coarse labels are easier), then deeper ones, which
    keeps the sequence totally ordered as required by fixed-sequence testing.
    """
    if tie or depth == 1:
        return [np.full(depth, v) for v in q]
    path = []
    cur = np.full(depth, q[0])
    path.append(cur.copy())
    # interleaved descent: step depth 0 by two, then deeper by one, repeat
    idx = np.zeros(depth, int)
    while True:
        moved = False
        for d in range(depth):
            steps = 2 if d == 0 else 1
            for _ in range(steps):
                if idx[d] + 1 < len(q) and (d == 0 or idx[d] + 1 <= idx[d - 1]):
                    idx[d] += 1
                    moved = True
                    cur = q[idx]
                    path.append(cur.copy())
        if not moved:
            break
    return path


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------


def evaluate(test: Annotated, acc: np.ndarray):
    n_acc, n_wrong = per_item_counts(test, acc)
    tot = acc.sum()
    fdp = n_wrong.sum() / max(1, tot)
    n_pred = np.bincount(test.item, minlength=test.n_items)
    n_correct_pred = np.bincount(test.item, weights=test.correct, minlength=test.n_items)
    out = dict(
        fdp=float(fdp),
        acc_labels=int(tot),
        frac_labels=float(tot / max(1, len(acc))),
        # power: share of correct predicted labels that are auto-accepted
        power=float((acc & (test.correct == 1)).sum() / max(1, test.correct.sum())),
        items_fully_auto=float((n_acc == n_pred).mean()),
        items_touched_by_human=float((n_acc < n_pred).mean()),
    )
    return out
