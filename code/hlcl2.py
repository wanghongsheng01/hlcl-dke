"""HLCL v2: dense, vectorised implementation of label-level hierarchical acceptance.

Each item has a predicted path of length <= D. We store
  S[n, D]  acceptance score per node (NaN where the path is shorter)
  C[n, D]  1 if the node is correct (gold prefix agrees), 0 otherwise
  M[n, D]  True where the path has a node at that depth

Acceptance with per-depth thresholds lam[0..D-1]:
  A[i, d] = prod_{e <= d} 1{S[i, e] >= lam[e]}
i.e. a node is accepted only if every ancestor is accepted (back-off): the item
is auto-labelled down to its deepest certified level and only the finer levels
are sent to a human.

Risk. Per-item loss L_i(lam) = W_i(lam) - alpha * A_i(lam) where A_i / W_i are
the numbers of accepted / accepted-and-wrong nodes of item i. Items are i.i.d.
(labels within an item need not be), so E[L(lam)] <= 0 is exactly the
label-level ratio-FDR condition E[W] <= alpha E[A]; no exchangeability across
labels of the same item is assumed.

Calibration (Learn-then-Test). Candidate thresholds and the test to use are
chosen on an ordering split that is disjoint from the calibration data; testing
is fixed-sequence on the calibration split, so with probability >= 1 - delta
every certified candidate satisfies E[W] <= alpha E[A].

Human effort. An item that is not fully auto-accepted costs a per-item overhead
c0 plus the share of every level the reviewer still has to decide (see
`effort`). Among certified candidates we return the one with minimum estimated
effort on the calibration data.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binom


def accept(S: np.ndarray, lam: np.ndarray, hier: bool = True) -> np.ndarray:
    ok = np.nan_to_num(S, nan=-np.inf) >= lam[None, :]
    return np.cumprod(ok, axis=1).astype(bool) if hier else ok


def counts(A, C):
    return A.sum(1), (A & (C == 0)).sum(1)


C0 = 0.5  # per-item review overhead; experiments may override (see --c0)


def effort(A, M, w=None, c0=None):
    """Expected human cost per item, in units of one full manual annotation.

    An item that is not fully auto-accepted costs a fixed overhead c0 (open the
    item, read the text) plus the share w_d of every level the reviewer still
    has to decide; a fully accepted item costs 0. Full manual annotation
    therefore costs c0 + 1, and we normalise by it. With c0 = 0 the objective
    reduces to maximising the number of accepted labels.
    """
    c0 = C0 if c0 is None else c0
    D = A.shape[1]
    w = np.full(D, 1.0 / D) if w is None else np.asarray(w, float) / np.sum(w)
    todo = ~(A & M)
    per = (todo * w[None, :]).sum(1) + c0 * todo.any(1)
    return float(per.mean() / (1 + c0))


def betting_p(loss: np.ndarray, lo: float, hi: float, seed: int = 12345) -> float:
    """Valid p-value for H0: E[loss] > 0 with loss in [lo, hi].

    Rescale to x in [0, 1], t = -lo / (hi - lo); under H0 E[x] > t. The capital
    K_n = prod_{i<=n} (1 + l_i (t - x_i)) with predictable bets l_i in
    [0, 0.5/(1-t)] is a nonnegative supermartingale under H0, so by Ville's
    inequality p = 1 / max_n K_n is valid (Waudby-Smith & Ramdas, 2024). Bets
    use the running variance, which adapts the test to the very small loss
    variance at conservative thresholds. The visiting order is a fixed
    pseudo-random permutation, independent of the data.
    """
    x = np.clip((loss - lo) / (hi - lo), 0, 1)
    t = -lo / (hi - lo)
    n = len(x)
    x = x[np.random.default_rng(seed).permutation(n)]
    i = np.arange(1, n + 1)
    mu = (0.5 + np.cumsum(x)) / (i + 1)
    var = (0.25 + np.cumsum((x - mu) ** 2)) / (i + 1)
    var_prev = np.concatenate([[0.25], var[:-1]])
    lam = np.minimum(np.sqrt(2 * np.log(2 / 0.1) / (n * var_prev)), 0.5 / (1 - t))
    logk = np.cumsum(np.log1p(lam * (t - x)))
    return float(min(1.0, np.exp(-logk.max())))


def iut_binom_pvalue(A: np.ndarray, C: np.ndarray, alpha: float, lam: np.ndarray) -> float:
    """Exact intersection-union p-value over depths, valid for the pooled FDR.

    For an item, A_{i,d}, W_{i,d} in {0, 1} are the accepted and the
    accepted-and-wrong indicators at depth d. Items are i.i.d., so conditioning
    on the acceptance vector leaves the accepted items' outcomes independent
    with the *same* conditional error probability
        q_d = P(W=1 | A=1) = E[W_d] / E[A_d].
    Hence, conditional on the observed n_d = sum_i A_{i,d} acceptances,
    W_d ~ Binomial(n_d, q_d) exactly, and the usual binomial tail
    p_d = P(Bin(n_d, alpha) <= w_d) is an exact p-value for H0^d: q_d > alpha.
    (A Poisson-binomial correction is *not* needed: the per-item probabilities
    conditioned on acceptance are constant, not varying.)

    The pooled FDR is E[W]/E[A] = sum_d E[W_d] / sum_d E[A_d], a weighted
    average of the q_d with weights E[A_d] / sum_e E[A_e]. Therefore pooled
    FDR > alpha implies q_d > alpha for at least one d, i.e. the bad event is
    contained in H_0 = union_d H0^d. The IUT p-value max_d p_d is valid for
    H_0 with no multiplicity correction, since for every d
        sup_{H_0} P(max_e p_e <= t) <= sup_{H_0^d} P(p_d <= t) <= t.
    Note that each p_d is exact and threshold-free: it depends on the score
    only through which nodes were accepted.
    """
    p = 0.0
    for d in range(A.shape[1]):
        n = int(A[:, d].sum())
        if n == 0:
            # no calibration evidence about depth d: the null H_d cannot be
            # rejected, unless the rule cannot accept at depth d at all (in
            # which case E[A_d] = 0 and depth d does not enter the ratio).
            if np.isfinite(lam[d]):
                return 1.0
            continue
        w = int((A[:, d] & (C[:, d] == 0)).sum())
        p = max(p, float(binom.cdf(w, n, alpha)))
    return p


def calibrate(S, C, M, grid, alpha, delta=0.1, hier=True, order=None,
              objective="effort", w=None, tests=("wsr", "iut")):
    """Learn-then-Test with the test statistic chosen by a pre-registered rule.

    Two statistics are valid for the same target {E[W] > alpha E[A]}: the pooled
    betting test ("wsr") and the depth-wise exact binomial intersection-union
    test ("iut"). They are not uniformly ordered: the pooled test is sharper
    whenever the per-depth error rates are heterogeneous (and it uses the pooled
    constraint, which is the actual target), while the depth-wise test only
    survives where the pooled test certifies nothing at all -- the regime of very
    small alpha and few calibration items.

    Rule, evaluated on held-out data disjoint from the calibration split:
    use "wsr" if it certifies at least one candidate there, otherwise "iut".
    The rule is a function of independent data and of alpha only, so fixed-
    sequence testing on the calibration split keeps the guarantee intact: with
    probability >= 1 - delta every certified threshold satisfies
    E[W] <= alpha E[A].

    `order` = (S_o, C_o, M_o) is that held-out data; we pass the split the score
    was fitted on, so no calibration item is spent on tuning. Candidates are
    ordered on it by p-value (conservative first among ties). On the calibration
    split we keep the certified candidate with the lowest estimated human
    effort. Returns (lam, n_certified, test used).
    """
    lo, hi = -alpha * S.shape[1], (1 - alpha) * S.shape[1]

    def pv(SS, CC, lam, how):
        A = accept(SS, lam, hier)
        if how == "wsr":
            a, wr = counts(A, CC)
            return betting_p(wr - alpha * a, lo, hi), A
        return iut_binom_pvalue(A, CC, alpha, lam), A

    def sequence(SS, CC, how):
        po = np.array([pv(SS, CC, lam, how)[0] for lam in grid])
        acc = np.array([accept(SS, lam, hier).sum() for lam in grid])
        return np.lexsort((acc, po))

    def scan(SS, CC, MM, seq, how):
        best, best_val, ncert = None, np.inf, 0
        for j in seq:
            p, A = pv(SS, CC, grid[j], how)
            if p > delta:
                break
            ncert += 1
            val = effort(A & MM, MM, w) if objective == "effort" else -A.sum()
            if val < best_val:
                best, best_val = grid[j], val
        return best, best_val, ncert

    So, Co, Mo = order
    if len(tests) == 1:
        best_how = tests[0]
    elif scan(So, Co, Mo, sequence(So, Co, "wsr"), "wsr")[2] > 0:
        best_how = "wsr"
    else:
        best_how = "iut"
    lam, _, ncert = scan(S, C, M, sequence(So, Co, best_how), best_how)
    return lam, ncert, best_how


def full_grid(levels: list[np.ndarray]) -> list[np.ndarray]:
    mesh = np.meshgrid(*levels, indexing="ij")
    return [np.array(v) for v in zip(*(m.ravel() for m in mesh))]


def metrics(A, C, M, w=None):
    a = A.sum()
    wrong = (A & (C == 0)).sum()
    return dict(
        fdp=float(wrong / max(1, a)),
        power=float((A & (C == 1)).sum() / max(1, (C == 1).sum())),
        frac_labels=float(a / max(1, M.sum())),
        items_full_auto=float(A.all(1).mean()),
        effort=effort(A, M, w),
    )
