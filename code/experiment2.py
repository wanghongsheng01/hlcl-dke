"""Experiment v2: learned hierarchy-aware score + effort-optimal certified thresholds.

Every random split partitions items into fit (20%) / calibration (40%) /
test (40%). The learned score is fitted on fit only; all calibrators see only
calibration; all numbers are on test.

Methods
  HLCL          learned score, ancestor-closed back-off, per-depth thresholds,
                certified family {thresholds} x {pooled, depth-wise test},
                effort-minimising selection                          (ours)
  HLCL-wsr      pooled betting test only            (ablation: test)
  HLCL-iut      exact per-depth binomial IUT only   (ablation: test)
  HLCL-vote     same, raw vote score                     (ablation: score)
  HLCL-flat     learned score, no ancestor closure       (ablation: hierarchy)
  HLCL-tied     learned score, one shared threshold      (ablation: per-depth)
  HLCL-maxlab   learned score, maximise #accepted labels (ablation: objective)
  HSC           hierarchical selective classification (Goren et al. 2024):
                ancestor-closed prefix with one shared threshold, certified for
                the item-level error rate over all items, most permissive
  ItemLTT       item-level certified selection, same certificate as HLCL
                (pooled / depth-wise test chosen by the same rule)
  ConfLabel     Conformal Labeling (Huang et al. 2025): conformal p-values on
                the learned item score + BH, item accepted or routed as a whole
  PerField-LTT  per-field selective risk control (2608.14639, field-iid
                Mondrian LTT tier with exact binomial tails): one certified
                threshold per depth, labels treated as independent fields
  DepthBH       conformal p-values + BH separately per depth (no hierarchy)
  Conf>=0.9     fixed confidence threshold (no guarantee)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binom

from features import build, fit_score
from hlcl2 import accept, calibrate, full_grid, metrics

ROOT = Path(__file__).resolve().parents[1]


def dense(a, X, s):
    D = int(a.depth.max()) + 1
    n = a.n_items
    S = np.full((n, D), np.nan)
    C = np.zeros((n, D), int)
    V = np.full((n, D), np.nan)
    S[a.item, a.depth] = s
    C[a.item, a.depth] = a.correct
    V[a.item, a.depth] = X[:, 0] + 0.01 * X[:, 4]
    M = ~np.isnan(S)
    return S, C, V, M


def levels(S, idx, m):
    out = []
    for d in range(S.shape[1]):
        v = S[idx, d]
        v = v[~np.isnan(v)]
        u = np.unique(v)
        if len(u) <= m:
            q = u
        else:
            q = np.unique(np.quantile(v, np.linspace(0, 1, m)))
        out.append(np.concatenate([q, [np.inf]]))
    return out


# ----------------------------------------------------------------- baselines
def bh(p, alpha):
    m = len(p)
    o = np.argsort(p)
    ok = p[o] <= alpha * np.arange(1, m + 1) / m
    k = np.max(np.nonzero(ok)[0]) + 1 if ok.any() else 0
    sel = np.zeros(m, bool)
    sel[o[:k]] = True
    return sel


def conf_p(cs, cbad, ts, rng):
    null = np.sort(cs[cbad])
    n = len(cs)
    ge = len(null) - np.searchsorted(null, ts, "left")
    eq = np.searchsorted(null, ts, "right") - np.searchsorted(null, ts, "left")
    return (ge - eq + rng.random(len(ts)) * (eq + 1)) / (n + 1)


def item_score(S, M):
    """Score of an item = score of its weakest predicted node (0 if the
    predicted path is shorter than the taxonomy)."""
    mn = np.where(M, np.nan_to_num(S, nan=np.inf), np.inf).min(1)
    return np.where(M.all(1), mn, 0.0)


def conf_label(S, C, M, ci, ti, alpha, rng):
    """Conformal Labeling on the learned item score. An item counts as
    correct only if every label is correct, so BH controls the *item-level*
    FDR at alpha, which upper-bounds the label-level FDR (each wrong item has
    at least one and at most D wrong labels out of D accepted ones)."""
    D = S.shape[1]
    good = C[ci].sum(1) == D
    sel = bh(conf_p(item_score(S[ci], M[ci]), ~good, item_score(S[ti], M[ti]), rng), alpha)
    return np.repeat(sel[:, None], D, 1) & M[ti]


def item_certified(S, C, M, fi, ci, alpha, delta, tests=("wsr", "iut"), So=None):
    """Item-level certified selection: an item's labels are accepted together or
    routed together. Implemented by giving every node of an item the item's
    *weakest* score and certifying a single shared threshold with the same
    learn-then-test machinery, so the comparison against HLCL isolates the
    granularity of the decision (item vs label), not the strength of the
    certificate. Returns (lam, how, ncert)."""
    So = S if So is None else So
    Si = np.where(M, item_score(S, M)[:, None], np.nan)
    Sio = np.where(M, item_score(So, M)[:, None], np.nan)
    qs = np.unique(np.concatenate([np.quantile(Sio[fi][:, 0], np.linspace(0, 1, 64)), [np.inf]]))
    grid = [np.full(S.shape[1], v) for v in qs]
    return calibrate(Si[ci], C[ci], M[ci], grid, alpha, delta, hier=False,
                     order=(Sio[fi], C[fi], M[fi]), objective="effort", tests=tests)


def hsc(S, C, M, fi, ci, ti, alpha, delta, So=None, m=64):
    """Hierarchical selective classification (Goren et al., NeurIPS 2024),
    adapted: each item returns the deepest prefix of its predicted path whose
    nodes all clear one shared threshold (root = abstain). The risk is the
    item-level error rate over ALL items, P(returned prefix contains a wrong
    node), certified with an exact binomial tail and fixed-sequence testing;
    the most permissive certified threshold is deployed."""
    So = S if So is None else So
    D = S.shape[1]
    qs = np.unique(np.concatenate([np.quantile(So[fi][M[fi]], np.linspace(0, 1, m)), [np.inf]]))
    grid = [np.full(D, v) for v in qs]

    def pval(SS, CC, lam):
        A = accept(SS, lam, True)
        e = (A & (CC == 0)).any(1)
        return float(binom.cdf(int(e.sum()), len(e), alpha)), A

    po = np.array([pval(So[fi], C[fi], g)[0] for g in grid])
    acc = np.array([accept(So[fi], g, True).sum() for g in grid])
    best, best_acc = None, -1
    for j in np.lexsort((acc, po)):
        p, A = pval(S[ci], C[ci], grid[j])
        if p > delta:
            break
        if A.sum() > best_acc:
            best, best_acc = grid[j], A.sum()
    if best is None:
        return np.zeros((len(ti), D), bool)
    return accept(S[ti], best, True) & M[ti]


def perfield_ltt(S, C, M, fi, ci, ti, alpha, delta, m=200):
    """Per-field selective risk control (2608.14639, field-iid tier): one
    threshold per depth (Mondrian group), exact binomial tail, Bonferroni
    delta/D over groups. Candidates are ordered by their binomial p-value on
    the fit split and fixed-sequence tested on calibration; the most
    permissive certified threshold is used."""
    D = S.shape[1]
    A = np.zeros((len(ti), D), bool)

    def pval(s, c, q):
        sel = s >= q
        nsel, nerr = sel.sum(), (sel & (c == 0)).sum()
        return 1.0 if nsel == 0 else binom.cdf(nerr, nsel, alpha)

    for d in range(D):
        sc, cc = S[ci, d], C[ci, d]
        k = ~np.isnan(sc)
        sc, cc = sc[k], cc[k]
        sf, cf = S[fi, d], C[fi, d]
        k = ~np.isnan(sf)
        sf, cf = sf[k], cf[k]
        qs = np.unique(np.quantile(sf, np.linspace(0, 1, m)))   # fit split only
        po = np.array([pval(sf, cf, q) for q in qs])
        lam = np.inf
        for j in np.lexsort((qs, po)):
            if pval(sc, cc, qs[j]) > delta / D:
                break
            lam = min(lam, qs[j])
        A[:, d] = np.nan_to_num(S[ti, d], nan=-np.inf) >= lam
    return A


def depth_bh(S, C, M, ci, ti, alpha, rng):
    D = S.shape[1]
    A = np.zeros((len(ti), D), bool)
    for d in range(D):
        cm, tm = M[ci, d], M[ti, d]
        p = conf_p(S[ci, d][cm], C[ci, d][cm] == 0, S[ti, d][tm], rng)
        A[np.nonzero(tm)[0], d] = bh(p, alpha)
    return A


# ----------------------------------------------------------------- main loop
def oof_scores(X, a, fi, s_full):
    """Out-of-fold scores for the fit split (2-fold cross-fitting), so that the
    ordering and the test-selection rule are not computed on the data the
    score was trained on. Scores of non-fit items are those of the model fitted
    on the whole fit split."""
    s = s_full.copy()
    rng = np.random.default_rng(7)
    halves = np.array_split(rng.permutation(fi), 2)
    for h in range(2):
        tr, te = halves[1 - h], halves[h]
        trn, ten = np.isin(a.item, tr), np.isin(a.item, te)
        s[ten] = fit_score(X[trn], a.correct[trn], X[ten])
    return s


OOF = False


def run(dataset, tag, n, k, R, alphas, delta, m_levels, out, ncal=None, light=False,
        item_only=False):
    a, X, meta = build(dataset, tag, n, k)
    print(meta, flush=True)
    D = meta["depth"]
    res = []
    for r in range(R):
        rng = np.random.default_rng(1000 + r)
        perm = rng.permutation(a.n_items)
        nf, nc = int(0.2 * a.n_items), int(0.4 * a.n_items)
        fi, ci, ti = perm[:nf], np.sort(perm[nf:nf + nc]), np.sort(perm[nf + nc:])
        if ncal is not None:          # sensitivity: smaller calibration split, same test split
            ci = np.sort(ci[:ncal])
        fit_nodes = np.isin(a.item, fi)
        s = fit_score(X[fit_nodes], a.correct[fit_nodes], X)
        S, C, V, M = dense(a, X, s)
        # fit-split decisions use either in-sample or out-of-fold fit scores;
        # both are functions of the fit split only, so both are valid
        So = dense(a, X, oof_scores(X, a, fi, s))[0] if OOF else S
        # candidate grids come from the fit split only, so the hypothesis set
        # is fixed before calibration data are seen (required by LTT)
        tied = lambda SS: [np.full(D, v) for v in np.unique(np.concatenate(
            [np.quantile(SS[fi][M[fi]], np.linspace(0, 1, 4 * m_levels)), [np.inf]]))]
        g_t = tied(So)
        g_l = full_grid(levels(So, fi, m_levels)) + g_t
        g_v = full_grid(levels(V, fi, m_levels)) + tied(V)
        for al in alphas:
            row = {"split": r, "alpha": al}
            if item_only:
                # Item-cert. under each fixed test (reviewer request); same splits
                for nm, tt in (("ItemLTT-wsr", ("wsr",)), ("ItemLTT-iut", ("iut",))):
                    lam_i, _, how_i = item_certified(S, C, M, fi, ci, al, delta, tests=tt, So=So)
                    Sti = np.where(M[ti], item_score(S[ti], M[ti])[:, None], np.nan)
                    Ai = (accept(Sti, lam_i, hier=False) if lam_i is not None
                          else np.zeros((len(ti), D), bool)) & M[ti]
                    row[nm] = metrics(Ai, C[ti], M[ti])
                res.append(row)
                continue
            variants = [("HLCL", S, So, g_l, True, "effort", ("wsr", "iut"))]
            if not light:
                for suf, tt in (("", ("wsr", "iut")), ("-wsr", ("wsr",)), ("-iut", ("iut",))):
                    if suf:
                        variants.append(("HLCL" + suf, S, So, g_l, True, "effort", tt))
                    variants += [("HLCL-vote" + suf, V, V, g_v, True, "effort", tt),
                                 ("HLCL-flat" + suf, S, So, g_l, False, "effort", tt),
                                 ("HLCL-tied" + suf, S, So, g_t, True, "effort", tt)]
                variants.append(("HLCL-maxlab", S, So, g_l, True, "labels", ("wsr", "iut")))
            for name, SS, SO, grid, hier, obj, tests in variants:
                lam, ncert, how = calibrate(SS[ci], C[ci], M[ci], grid, al, delta,
                                       hier=hier, order=(SO[fi], C[fi], M[fi]),
                                       objective=obj, tests=tests)
                A = (accept(SS[ti], lam, hier) if lam is not None
                     else np.zeros((len(ti), D), bool)) & M[ti]
                row[name] = metrics(A, C[ti], M[ti]) | {"ncert": ncert, "test": how}
            lam_i, _, how_i = item_certified(S, C, M, fi, ci, al, delta, So=So)
            Sti = np.where(M[ti], item_score(S[ti], M[ti])[:, None], np.nan)
            Ai = (accept(Sti, lam_i, hier=False) if lam_i is not None
                  else np.zeros((len(ti), D), bool)) & M[ti]
            row["ItemLTT"] = metrics(Ai, C[ti], M[ti]) | {"test": how_i}
            row["HSC"] = metrics(hsc(S, C, M, fi, ci, ti, al, delta, So=So), C[ti], M[ti])
            row["ConfLabel"] = metrics(conf_label(S, C, M, ci, ti, al, rng), C[ti], M[ti])
            row["PerField-LTT"] = metrics(perfield_ltt(S, C, M, fi, ci, ti, al, delta) & M[ti], C[ti], M[ti])
            row["DepthBH"] = metrics(depth_bh(S, C, M, ci, ti, al, rng), C[ti], M[ti])
            row["Conf>=0.9"] = metrics((np.nan_to_num(S[ti], nan=0) >= 0.9) & M[ti], C[ti], M[ti])
            res.append(row)
        if (r + 1) % 10 == 0:
            print(f"  split {r+1}/{R}", flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"{out}.json").write_text(json.dumps({"meta": meta, "delta": delta, "runs": res}))
    summarize(res, alphas)


def summarize(res, alphas):
    methods = [k for k in res[0] if k not in ("split", "alpha")]
    for al in alphas:
        rows = [r for r in res if r["alpha"] == al]
        print(f"\nalpha={al} (R={len(rows)})")
        print(f"{'method':13s} {'FDR':>7s} {'P(FDP>a)':>9s} {'power':>7s} {'effort':>7s} {'full-auto':>9s}")
        for m in methods:
            f = np.array([r[m]["fdp"] for r in rows])
            g = lambda key: np.mean([r[m][key] for r in rows])
            print(f"{m:13s} {f.mean():7.4f} {np.mean(f > al):9.3f} {g('power'):7.3f} "
                  f"{g('effort'):7.3f} {g('items_full_auto'):9.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--R", type=int, default=100)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--alphas", default="0.02,0.05,0.1")
    ap.add_argument("--levels", type=int, default=12)
    ap.add_argument("--out", default=None)
    ap.add_argument("--c0", type=float, default=0.5)
    ap.add_argument("--ncal", type=int, default=None)
    ap.add_argument("--light", action="store_true", help="skip ablations")
    ap.add_argument("--oof", action="store_true", help="out-of-fold fit scores for fit-split decisions")
    ap.add_argument("--item-only", action="store_true", help="only Item-cert. fixed-test variants")
    x = ap.parse_args()
    import hlcl2
    hlcl2.C0 = x.c0
    OOF = x.oof
    run(x.dataset, x.tag, x.n, x.k, x.R, [float(v) for v in x.alphas.split(",")],
        x.delta, x.levels, x.out or f"v2_{x.dataset}_{x.tag}", ncal=x.ncal, light=x.light,
        item_only=x.item_only)
