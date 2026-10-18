"""Build every table and every number quoted in the paper from the released
per-split results, so that text, tables and figures cannot drift apart.

Usage: python make_tables.py [VER]      (default VER = v9)
Writes paper/tables/{main_results,ablations,sens,annot}.tex and
paper/tables/numbers.tex, which defines \\res{set}{alpha}{method}{key},
\\dif{set}{alpha}{method}{key} (paired mean difference HLCL - method with a
95% CI) and \\stat{name}.

Win/tie/loss against the strongest valid certified baseline is decided by a
paired test over the 100 splits: the 95% CI of the paired effort difference
must exclude zero for a win or a loss; otherwise the setting is a tie.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "paper" / "tables"
VER = sys.argv[1] if len(sys.argv) > 1 else "v9"
DELTA = 0.1

SETTINGS = [("dbpedia", "DBPedia (A)"), ("wos", "WOS-46985 (A)"),
            ("bgc", "Blurb (A)"), ("bgc_haiku", "Blurb (B)")]
CERTIFIED = ["ItemBest", "HSC", "PerField-LTT"]     # high-probability certificates
MAIN = ["HLCL", "ItemBest", "HSC", "PerField-LTT", "ConfLabel", "DepthBH", "Conf>=0.9"]
ITEM_VARIANTS = ["ItemLTT", "ItemLTT-wsr", "ItemLTT-iut"]
# Nadeau & Bengio (2003) corrected resampled-t variance factor for 100 random
# splits of the same items: 1/J + n_test/n_train (n_train = fit + calibration)
NB_FACTOR = 1 / 100 + 1200 / 1800
MAIN_HEAD = [r"\hlcl{}", r"Item-cert.$^\ast$", "HSC", "Per-field", "Item-conf.", "Depth-BH",
             r"$s\ge0.9$"]
ABL_GROUPS = [("automatic test", ""), ("pooled test only", "-wsr"), ("depth-wise test only", "-iut")]
ABL_VARS = [("HLCL", "full"), ("HLCL-vote", "vote"), ("HLCL-flat", "no cl."),
            ("HLCL-tied", "tied")]


def load(name):
    return json.loads((ROOT / "results" / f"{VER}_{name}.json").read_text())


def alphas(runs):
    return sorted({r["alpha"] for r in runs})


def at(runs, a):
    return [r for r in runs if r["alpha"] == a]


def vals(R, m, k):
    return np.array([r[m][k] for r in R], float)


def mean(R, m, k):
    return float(vals(R, m, k).mean())


def viol(R, m, a):
    return float(np.mean(vals(R, m, "fdp") > a + 1e-12))


def paired(R, m, k="effort"):
    """Paired mean difference HLCL - m and the half-width of a 95% interval
    with the Nadeau-Bengio correction for overlapping random splits."""
    d = vals(R, "HLCL", k) - vals(R, m, k)
    se = np.sqrt(NB_FACTOR * d.var(ddof=1)) if len(d) > 1 else 0.0
    return float(d.mean()), float(1.96 * se)


def add_item_best(data):
    """Item-cert. with the best of its three test choices in each setting,
    chosen on the test results (an oracle choice that can only favour the
    baseline)."""
    for key, _ in SETTINGS:
        runs = data[key]["runs"]
        for a in alphas(runs):
            R = at(runs, a)
            have = [m for m in ITEM_VARIANTS if m in R[0]]
            b = min(have, key=lambda m: mean(R, m, "effort"))
            for r in R:
                r["ItemBest"] = r[b]
                r["_itembest"] = b


def fmt(x, d=2):
    return f"{x:.{d}f}"


def abl_name(base, suf):
    return base + suf if base != "HLCL" or suf else "HLCL"


def main_table(data):
    L = [r"\begin{tabular}{ll" + "c" * len(MAIN) + "}", r"\toprule",
         r"Setting & $\alpha$ & " + " & ".join(MAIN_HEAD) + r" \\", r"\midrule"]
    for key, lab in SETTINGS:
        runs = data[key]["runs"]
        for i, a in enumerate(alphas(runs)):
            R = at(runs, a)
            valid = [m for m in ["HLCL"] + CERTIFIED if viol(R, m, a) <= DELTA]
            best = min(valid, key=lambda m: mean(R, m, "effort")) if valid else None
            cells = []
            for m in MAIN:
                s = f"{fmt(mean(R, m, 'effort'))}/{fmt(mean(R, m, 'power'))}"
                if m == best:
                    s = r"\textbf{" + s + "}"
                if viol(R, m, a) > DELTA:
                    s += r"$^\dagger$"
                cells.append(s)
            L.append(f"{lab if i == 0 else ''} & {a:g} & " + " & ".join(cells) + r" \\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    (TAB / "main_results.tex").write_text("\n".join(L) + "\n")


def ablation_table(data):
    head = " & ".join(f"\\multicolumn{{{len(ABL_VARS)}}}{{c}}{{{g}}}" for g, _ in ABL_GROUPS)
    sub = " & ".join(v for _ in ABL_GROUPS for _, v in ABL_VARS)
    ncol = len(ABL_GROUPS) * len(ABL_VARS)
    rules = " ".join(f"\\cmidrule(lr){{{3 + i * len(ABL_VARS)}-{2 + (i + 1) * len(ABL_VARS)}}}"
                     for i in range(len(ABL_GROUPS)))
    L = [r"\begin{tabular}{ll" + "c" * ncol + "}", r"\toprule",
         r" & & " + head + r" \\", rules, r"Setting & $\alpha$ & " + sub + r" \\", r"\midrule"]
    for key, lab in SETTINGS:
        runs = data[key]["runs"]
        for i, a in enumerate(alphas(runs)):
            R = at(runs, a)
            cells = []
            for _, suf in ABL_GROUPS:
                names = [abl_name(b, suf) for b, _ in ABL_VARS]
                v = [mean(R, n, "effort") for n in names]
                lo = min(v)
                cells += [(r"\textbf{" + fmt(x) + "}") if abs(x - lo) < 5e-3 else fmt(x) for x in v]
            L.append(f"{lab if i == 0 else ''} & {a:g} & " + " & ".join(cells) + r" \\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    (TAB / "ablations.tex").write_text("\n".join(L) + "\n")


def sens_rows():
    spec = [("$c_0=0$", "bgc_c0_0"), ("$c_0=0.5$ (default)", "bgc"), ("$c_0=1$", "bgc_c0_1"),
            ("$n_{\\mathrm{cal}}=300$", "bgc_ncal300"), ("$n_{\\mathrm{cal}}=600$", "bgc_ncal600"),
            ("$n_{\\mathrm{cal}}=1200$ (default)", "bgc"),
            ("$\\delta=0.05$", "bgc_delta0.05"), ("$\\delta=0.1$ (default)", "bgc"),
            ("$\\delta=0.2$", "bgc_delta0.2")]
    rows = []
    for lab, name in spec:
        p = ROOT / "results" / f"{VER}_{name}.json"
        if not p.exists():
            continue
        runs = json.loads(p.read_text())["runs"]
        for a in (0.05, 0.1):
            R = at(runs, a)
            if R:
                rows.append((lab, name, a, mean(R, "HLCL", "effort"), mean(R, "HLCL", "power"),
                             mean(R, "HLCL", "items_full_auto"), viol(R, "HLCL", a),
                             mean(R, "PerField-LTT", "effort"), mean(R, "ItemLTT", "effort"),
                             mean(R, "HSC", "effort")))
    return rows


def sens_table(rows):
    L = [r"\begin{tabular}{lcccccccc}", r"\toprule",
         r" & & \multicolumn{4}{c}{\hlcl{}} & \multicolumn{3}{c}{effort of baselines} \\",
         r"\cmidrule(lr){3-6} \cmidrule(lr){7-9}",
         r"Setting & $\alpha$ & effort & power & fully auto & viol. & "
         r"Item-cert. & HSC & Per-field \\", r"\midrule"]
    prev = None
    for lab, name, a, e, p, fa, v, ep, ei, eh in rows:
        if prev is not None and lab.split("=")[0] != prev.split("=")[0]:
            L.append(r"\midrule")
        L.append(f"{lab} & {a:g} & {fmt(e)} & {fmt(p)} & {fmt(fa)} & {100 * v:.0f}\\% & "
                 f"{fmt(ei)} & {fmt(eh)} & {fmt(ep)} \\\\")
        prev = lab
    L += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "sens.tex").write_text("\n".join(L) + "\n")


def annot_table(data):
    L = [r"\begin{tabular}{lccc}", r"\toprule",
         r"Dataset (annotator) & level 1 & level 2 & level 3 \\", r"\midrule"]
    for key, lab in SETTINGS:
        acc = data[key]["meta"]["acc_by_depth"]
        cells = [fmt(x, 3) for x in acc] + ["---"] * (3 - len(acc))
        L.append(f"{lab} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "annot.tex").write_text("\n".join(L) + "\n")


def numbers(data, srows):
    out = [r"% generated by code/make_tables.py -- do not edit",
           r"\makeatletter",
           r"\newcommand{\res}[4]{\@ifundefined{r@#1@#2@#3@#4}{\textbf{??}}{\csname r@#1@#2@#3@#4\endcsname}}",
           r"\newcommand{\dif}[4]{\@ifundefined{d@#1@#2@#3@#4}{\textbf{??}}{\csname d@#1@#2@#3@#4\endcsname}}",
           r"\newcommand{\stat}[1]{\@ifundefined{#1}{\textbf{??}}{\csname #1\endcsname}}",
           r"\makeatother"]

    def put(name, val):
        out.append(r"\expandafter\def\csname " + name + r"\endcsname{" + val + "}")

    every = sorted({m for k, _ in SETTINGS for m in data[k]["runs"][0]
                    if m not in ("split", "alpha", "_itembest")})
    maxv = {m: (0.0, None) for m in MAIN}
    tally = {"win": [], "tie": [], "loss": []}
    lower = 0
    tests = {}
    for key, _ in SETTINGS:
        runs = data[key]["runs"]
        for a in alphas(runs):
            R = at(runs, a)
            for m in every:
                for k in ("power", "effort", "fdp", "items_full_auto"):
                    put(f"r@{key}@{a:g}@{m}@{k}", fmt(mean(R, m, k)))
                put(f"r@{key}@{a:g}@{m}@viol", f"{100 * viol(R, m, a):.0f}\\%")
                if m != "HLCL":
                    for k in ("effort", "power"):
                        d, h = paired(R, m, k)
                        put(f"d@{key}@{a:g}@{m}@{k}", f"{d:+.3f}$\\pm${h:.3f}")
            for m in MAIN:
                v = viol(R, m, a)
                if v > maxv[m][0]:
                    maxv[m] = (v, (key, a))
            valid = [m for m in CERTIFIED if viol(R, m, a) <= DELTA]
            b = min(valid, key=lambda m: mean(R, m, "effort"))
            d, h = paired(R, b, "effort")
            lower += d < 0
            put(f"r@{key}@{a:g}@best@name", {"ItemBest": "Item-cert.", "HSC": "HSC",
                                              "PerField-LTT": "Per-field"}[b])
            put(f"r@{key}@{a:g}@best@dif", f"{d:+.3f}$\\pm${h:.3f}")
            put(f"r@{key}@{a:g}@best@winshare",
                f"{100 * np.mean(vals(R, 'HLCL', 'effort') < vals(R, b, 'effort') - 1e-9):.0f}\\%")
            cls = "win" if d + h < 0 else ("loss" if d - h > 0 else "tie")
            tally[cls].append((key, a, b, round(d, 3), round(h, 3)))
            iut = np.mean([r["HLCL"].get("test") == "iut" for r in R])
            tests[(key, a)] = iut
            put(f"r@{key}@{a:g}@HLCL@iutshare", f"{100 * iut:.0f}\\%")
    for m in MAIN:
        put(f"s@maxviol@{m}", f"{100 * maxv[m][0]:.0f}\\%")
        mf = max((mean(at(data[k]["runs"], a), m, "fdp") / a, k, a)
                 for k, _ in SETTINGS for a in alphas(data[k]["runs"]))
        put(f"s@maxfdpratio@{m}", fmt(mf[0]))
    for c in ("win", "tie", "loss"):
        put(f"s@{c}s" if c != "loss" else "s@losses", str(len(tally[c])))
    put("s@nsettings", str(sum(len(v) for v in tally.values())))
    put("s@lowermean", str(lower))
    nb = []
    put("s@nbfactor", fmt(NB_FACTOR, 3))
    put("s@wosfdpconf", fmt(mean(at(data["wos"]["runs"], 0.2), "Conf>=0.9", "fdp")))
    for lab, name, a, e, p, fa, v, ep, ei, eh in srows:
        put(f"s@sens@{name}@{a:g}@effort", fmt(e))
        put(f"s@sens@{name}@{a:g}@power", fmt(p))
        put(f"s@sens@{name}@{a:g}@fullauto", fmt(fa))
        put(f"s@sens@{name}@{a:g}@viol", f"{100 * v:.0f}\\%")
    (TAB / "numbers.tex").write_text("\n".join(out) + "\n")
    return maxv, tally, tests


if __name__ == "__main__":
    TAB.mkdir(parents=True, exist_ok=True)
    data = {k: load(k) for k, _ in SETTINGS}
    add_item_best(data)
    main_table(data)
    ablation_table(data)
    srows = sens_rows()
    sens_table(srows)
    annot_table(data)
    maxv, tally, tests = numbers(data, srows)
    print("max violation share:", {m: (round(v, 2), w) for m, (v, w) in maxv.items()})
    for c in ("win", "tie", "loss"):
        print(c, len(tally[c]))
        for x in tally[c]:
            print("   ", x)
    print("share of splits where HLCL used the depth-wise test:",
          {f"{k}@{a:g}": round(v, 2) for (k, a), v in tests.items()})
