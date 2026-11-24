"""Tables, figures and number macros of the revised letter and its supplement.

    python report.py   ->  <paper>/tables/*.tex, <paper>/figures/*.pdf, <paper>/tables/numbers.tex
Every number quoted in the manuscript is a macro written here from results_v2/.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results_v2"
PAPER = Path(os.environ.get("PAPER_DIR", HERE.parents[1] / "SPL_resubmission"))
TAB, FIG = PAPER / "tables", PAPER / "figures"

FAMS = ["sea20", "sea10", "sea5", "sea2", "sea1", "gauss5", "gauss1", "gmm5", "credit", "covtype", "insects"]
FHEAD = ["S$_{20}$", "S$_{10}$", "S$_{5}$", "S$_{2}$", "S$_{1}$", "G$_{5}$", "G$_{1}$", "M$_{3}$", "Cr.", "Co.", "In."]
MAIN = [("bank", "LR bank (ours)"), ("bank-theory", "LR bank, $h$ of Prop.~1"),
        ("ph-bank", "PH bank$^\\ast$"), ("ddm-bank", "DDM bank"), ("adwin-bank", "ADWIN bank"),
        ("pooled", "LR pooled$^\\ast$"), ("ph-pooled", "PH pooled$^\\ast$"), ("ra", "RA-CUSUM"),
        ("ddm-pooled", "DDM"), ("adwin-pooled", "ADWIN"),
        ("ddm-oci", "DDM-OCI"), ("lfr", "LFR"), ("rbmim", "RBM-IM$^\\dagger$")]
RANKED = [k for k, _ in MAIN if k != "bank-theory"]
N = {}


def macro(name, val):
    N[name] = val


def bench(f):
    return json.loads((RES / "bench" / f"{f}.json").read_text())


def delays(r, det):
    return np.array([z["delay"] if z["detected"] else z["horizon"] for z in r["det"][det]["runs"]], float)


def boot_ci(x, B=4000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(B, len(x))).mean(1)
    return np.percentile(m, [2.5, 97.5])


# --------------------------------------------------------------------------- #
def table_bench():
    R = {f: bench(f) for f in FAMS}
    mean = {d: [delays(R[f], d).mean() for f in FAMS] for d, _ in MAIN}
    nd = {d: [sum(z["detected"] for z in R[f]["det"][d]["runs"]) for f in FAMS] for d, _ in MAIN}
    M = np.array([mean[d] for d in RANKED])                      # detectors x families
    ranks = np.array([stats.rankdata(M[:, j]) for j in range(M.shape[1])]).T
    avg = dict(zip(RANKED, ranks.mean(1)))
    chi2, p = stats.friedmanchisquare(*M)
    # paired one-sided Wilcoxon over the 30 seeds: bank faster than each competitor, Holm within family
    sig = {}
    for j, f in enumerate(FAMS):
        b = delays(R[f], "bank")
        ps = {}
        for d in RANKED:
            if d == "bank":
                continue
            x = delays(R[f], d)
            ps[d] = stats.wilcoxon(b, x, alternative="less", zero_method="zsplit").pvalue if (b != x).any() else 1.0
        order = sorted(ps, key=ps.get)
        run = 0.0
        for i, d in enumerate(order):
            run = max(run, min(1.0, (len(order) - i) * ps[d]))
            sig[(d, f)] = run
    lines = ["\\begin{tabular}{@{}l*{11}{r}r@{}}", "\\toprule",
             "Detector & " + " & ".join(FHEAD) + " & Rank \\\\", "\\midrule"]
    disp = lambda v: f"{v / 1000:.1f}" if v < 1e5 else f"{v / 1000:.0f}"
    best = [min(float(disp(mean[d][j])) for d in RANKED) for j in range(len(FAMS))]   # at display precision
    for d, lab in MAIN:
        cells = []
        for j, f in enumerate(FAMS):
            s = disp(mean[d][j])
            if d != "bank-theory" and float(s) <= best[j]:
                s = f"\\textbf{{{s}}}"
            sup = []
            if nd[d][j] < 30:
                sup.append(str(nd[d][j]))
            ac = R[f]["det"][d]["cal_rate"]
            if ac and abs(1 / ac - 5e4) / 5e4 > 0.15:
                sup.append("\\ddagger")
            if sup:
                s += "$^{\\scriptscriptstyle " + ",".join(sup) + "}$"
            cells.append(s)
        rk = "--" if d == "bank-theory" else f"{avg[d]:.2f}"
        if d != "bank-theory" and avg[d] <= min(avg.values()) + 1e-12:
            rk = f"\\textbf{{{rk}}}"
        lines.append(f"{lab} & " + " & ".join(cells) + f" & {rk} \\\\")
        if d in ("bank-theory", "adwin-bank", "adwin-pooled"):
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "tab_bench.tex").write_text("\n".join(lines) + "\n")
    # numbers
    macro("FriedP", f"{p:.1e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}")
    macro("RankBank", f"{avg['bank']:.2f}")
    order = sorted(avg, key=avg.get)
    macro("RankSecondName", dict(MAIN)[order[1]].replace("$^\\ast$", "").replace("$^\\dagger$", ""))
    macro("RankSecond", f"{avg[order[1]]:.2f}")
    nbest = sum(float(disp(mean["bank"][j])) <= best[j] for j in range(len(FAMS)))
    macro("NBest", str(nbest))
    wins = {f: all(sig[(d, f)] < 0.05 for d in RANKED if d not in ("bank", "bank-oracle", "ph-bank"))
            for f in FAMS}
    macro("NSigAllButPH", str(sum(wins.values())))
    nsig_ph = sum(sig[("ph-bank", f)] < 0.05 for f in FAMS)
    macro("NSigPH", str(nsig_ph))
    # imbalance-aware competitors: ratio of mean delays
    for d, nm in (("ddm-oci", "Oci"), ("lfr", "Lfr"), ("rbmim", "Rbm"), ("pooled", "Pool"), ("ph-pooled", "PHPool"),
                  ("ra", "Ra")):
        ratio = np.array(mean[d]) / np.array(mean["bank"])
        macro(f"Ratio{nm}Med", f"{np.median(ratio):.1f}")
        macro(f"Ratio{nm}Min", f"{ratio.min():.1f}")
        macro(f"Ratio{nm}Max", f"{ratio.max():.0f}" if ratio.max() >= 10 else f"{ratio.max():.1f}")
        macro(f"NSig{nm}", str(sum(sig[(d, f)] < 0.05 for f in FAMS)))
    macro("NoDetTheory", str(sum(30 - x for x in nd["bank-theory"])))
    th_null = [sum(z["any"] for z in R[f]["det"]["bank-theory"]["null"]) for f in FAMS]
    macro("TheoryNullFA", str(sum(th_null)))
    macro("TheoryH", f"{np.mean([R[f]['det']['bank-theory']['knob'] for f in FAMS]):.0f}")
    return R, mean, nd, avg, sig


def table_supp_bench(R):
    """Full benchmark: mean delay with 95% bootstrap CI, detections, achieved ARL (calibration and evaluation)."""
    hdr = "Stream & Detector & Delay [95\\% CI] & Det. & ARL$_{\\mathrm{cal}}$ & ARL$_{\\mathrm{eval}}$ \\\\"
    lines = ["\\begin{longtable}{@{}ll*{4}{r}@{}}",
             "\\caption{Detection benchmark: mean delay ($\\times10^3$ samples) with 95\\% bootstrap confidence "
             "interval over 30 seeds, number of detections, and ARL achieved on the 30 calibration streams "
             "(ARL$_{\\mathrm{cal}}$) and on 30 independent drift-free evaluation streams (ARL$_{\\mathrm{eval}}$; "
             "$^{+}$: no false alarm, lower bound).}\\label{tab:s-bench}\\\\",
             "\\toprule", hdr, "\\midrule", "\\endfirsthead", "\\toprule", hdr, "\\midrule", "\\endhead"]
    for f, fh in zip(FAMS, FHEAD):
        r = R[f]
        for d, lab in MAIN + [("bank-oracle", "LR bank, oracle design$^\\ast$"), ("bank-multi", "LR bank, $r\\in\\{1.25,2,4\\}$")]:
            x = delays(r, d)
            lo, hi = boot_ci(x)
            det = sum(z["detected"] for z in r["det"][d]["runs"])
            ac = r["det"][d]["cal_rate"]
            arl_c = "--" if not ac else f"{1 / ac / 1000:.0f}k"
            fa = sum(z["fa"] for z in r["det"][d]["null"])
            arl_e = f"{r['n_mon'] * 30 / max(fa, 1) / 1000:.0f}k" + ("" if fa else "$^{+}$")
            lines.append(f"{fh} & {lab} & {x.mean() / 1000:.1f} [{lo / 1000:.1f}, {hi / 1000:.1f}] & {det} & "
                         f"{arl_c} & {arl_e} \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{longtable}")
    (TAB / "supp_bench.tex").write_text("\n".join(lines) + "\n")
    # streams table
    rows = ["\\begin{tabular}{@{}llrrrrr@{}}", "\\toprule",
            "Key & Stream & $K$ & $\\pi_c$ (\\%) & $\\mu_c$ & $\\Delta_c$ & $\\tau-t_0$ \\\\", "\\midrule"]
    desc = {"sea20": "SEA, 20\\% minority", "sea10": "SEA, 10\\%", "sea5": "SEA, 5\\%", "sea2": "SEA, 2\\%",
            "sea1": "SEA, 1\\%", "gauss5": "Gaussian, 5\\%", "gauss1": "Gaussian, 1\\%",
            "gmm5": "5-class Gaussian", "credit": "CreditCard + shift", "covtype": "Covertype + shift",
            "insects": "INSECTS + shift"}
    pre = {"sea20": 10, "sea10": 10, "sea5": 10, "sea2": 10, "sea1": 10, "gauss5": 10, "gauss1": 10,
           "gmm5": 15, "credit": 20, "covtype": 40, "insects": 30}
    K = {"credit": 2, "covtype": 7, "insects": 6, "gmm5": 5}
    for f, fh in zip(FAMS, FHEAD):
        r = R[f]
        rows.append(f"{fh} & {desc[f]} & {K.get(f, 2)} & {100 * r['share']:.2f} & {r['mu_pre_target']:.2f} & "
                    f"{r['oracle']['dc']:.2f} & {pre[f]}k \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "supp_streams.tex").write_text("\n".join(rows) + "\n")


def localisation(R):
    err = []
    for f in FAMS:
        err += [abs(z["loc_err"]) for z in R[f]["det"]["bank"]["runs"] if "loc_err" in z]
    d = np.concatenate([delays(R[f], "bank") for f in FAMS])
    macro("LocMedian", f"{np.median(err):.0f}")
    macro("LocRatio", f"{np.median(err) / np.median(d) * 100:.0f}")


# --------------------------------------------------------------------------- #
def fig_sim():
    S = json.loads((RES / "sim.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.75))
    plt.rcParams.update({"font.size": 7})
    sty = {"bank": ("LR bank", "k", "o", "-"), "bank-oracle": ("LR bank, oracle", "0.45", "s", "--"),
           "ph-bank": ("PH bank$^\\ast$", "tab:blue", "^", "-"),
           "pooled": ("LR pooled$^\\ast$", "tab:red", "v", "-"),
           "ph-pooled": ("PH pooled$^\\ast$", "tab:orange", "D", "-"), "ra": ("RA-CUSUM", "tab:purple", "x", "-")}
    rows = S["s1"]
    pis = [r["pi"] for r in rows]
    for m, (lab, c, mk, ls) in sty.items():
        ax[0].plot(pis, [r[m]["mean"] for r in rows], ls=ls, color=c, marker=mk, ms=2.8, lw=0.9, label=lab)
    ax[0].set(xscale="log", yscale="log", xlabel="share $\\pi_c$ of the changed class", ylabel="mean delay")
    ax[0].set_title("(a) delay at ARL $10^5$", fontsize=7)
    ax[0].legend(fontsize=4.6, frameon=False, loc="lower left", ncol=1)
    rows3 = S["s3"]
    h = [r["h"] for r in rows3]
    for key, lab, c, mk in [("bank_iid", "bank, i.i.d.", "k", "o"), ("bank_prior_shift", "bank, prior shift", "0.45", "s"),
                            ("bank_markov_labels", "bank, Markov labels", "tab:blue", "^"),
                            ("pooled_prior_shift", "pooled, prior shift", "tab:red", "v"),
                            ("bank_autocorr_res", "bank, corr. residuals", "tab:green", "D")]:
        y = np.maximum([r[key] for r in rows3], 1e-5)
        ax[1].plot(h, y, color=c, marker=mk, ms=2.8, lw=0.9, label=lab)
    ax[1].plot(h, [r["bound"] for r in rows3], "k:", lw=1.0, label="$n e^{-h}$ (Prop. 1)")
    ax[1].set(yscale="log", xlabel="threshold $h$", ylabel="$P(T\\leq 10^4)$", ylim=(1e-5, 1.5))
    ax[1].set_title("(b) no change", fontsize=7)
    ax[1].legend(fontsize=4.6, frameon=False, loc="lower left")
    for a in ax:
        a.tick_params(labelsize=6)
        a.grid(alpha=0.25, lw=0.4)
    fig.tight_layout(pad=0.2, w_pad=0.6)
    fig.savefig(FIG / "fig_sim.pdf")
    fig.savefig(FIG / "fig_sim.png", dpi=200)
    # numbers
    r5 = [r for r in rows if abs(r["pi"] - 0.05) < 1e-9][0]
    r1 = [r for r in rows if abs(r["pi"] - 0.01) < 1e-9][0]
    macro("SimRatioFive", f"{r5['pooled']['mean'] / r5['bank']['mean']:.1f}")
    macro("SimRatioOne", f"{r1['pooled']['mean'] / r1['bank']['mean']:.1f}")
    macro("SimInfoFive", f"{r5['I_R'] / r5['I_P']:.1f}")
    macro("SimInfoOne", f"{r1['I_R'] / r1['I_P']:.0f}")
    macro("SimOracleGap", f"{max(r['bank']['mean'] / r['bank-oracle']['mean'] for r in rows):.2f}")
    macro("SimBoundOk", "yes" if all(r["bank"]["mean"] <= r["bound_bank"] for r in rows) else "no")
    macro("SimRaFive", f"{r5['ra']['mean'] / r5['bank']['mean']:.1f}")
    s2 = S["s2"]
    macro("SimGainLo", f"{s2[0]['pooled']['mean'] / s2[0]['bank']['mean']:.1f}")
    macro("SimGainHi", f"{s2[-1]['pooled']['mean'] / s2[-1]['bank']['mean']:.1f}")
    r12 = [r for r in rows3 if r["h"] == 12][0]
    macro("FAiid", f"{r12['bank_iid']:.4f}")
    macro("FAbound", f"{r12['bound']:.3f}")
    macro("FApoolShift", f"{r12['pooled_prior_shift']:.2f}")
    macro("FAbankShift", f"{r12['bank_prior_shift']:.4f}")
    macro("FApoolMarkov", f"{r12['pooled_markov_labels']:.2f}")
    macro("FAbankMarkov", f"{r12['bank_markov_labels']:.4f}")
    macro("FAcorr", f"{r12['bank_autocorr_res']:.2f}")
    s4 = S["s4"]
    m100 = [r for r in s4 if r["m"] == 100][0]
    m25 = [r for r in s4 if r["m"] == 25][0]
    m400 = [r for r in s4 if r["m"] == 400][0]
    macro("EstPlugTwentyFive", f"{m25['plugin']['fa']:.2f}")
    macro("EstPlugHundred", f"{m100['plugin']['fa']:.3f}")
    macro("EstCPDelayRatio", f"{m400['cp']['delay'] / m400['plugin']['delay']:.1f}")
    # supplement: delay vs ARL table and estimation table
    lines = ["\\begin{tabular}{@{}r*{6}{r}r@{}}", "\\toprule",
             "ARL & LR bank & oracle & PH bank$^\\ast$ & LR pooled$^\\ast$ & PH pooled$^\\ast$ & RA & ratio \\\\",
             "\\midrule"]
    for r in s2:
        lines.append(f"$10^{{{math.log10(r['gamma']):.1f}}}$".replace(".0}", "}") + " & " +
                     " & ".join(f"{r[m]['mean']:.0f}" for m in ["bank", "bank-oracle", "ph-bank", "pooled", "ph-pooled", "ra"])
                     + f" & {r['pooled']['mean'] / r['bank']['mean']:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "supp_arl.tex").write_text("\n".join(lines) + "\n")
    lines = ["\\begin{tabular}{@{}r*{4}{r}@{}}", "\\toprule",
             "$m_c$ & FA, plug-in & FA, CP & delay, plug-in & delay, CP \\\\", "\\midrule"]
    for r in s4:
        lines.append(f"{r['m']} & {r['plugin']['fa']:.3f} & {r['cp']['fa']:.3f} & {r['plugin']['delay']:.0f} & "
                     f"{r['cp']['delay']:.0f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "supp_est.tex").write_text("\n".join(lines) + "\n")
    lines = ["\\begin{tabular}{@{}r*{5}{r}@{}}", "\\toprule",
             "$\\pi_c$ & $I_R/I_P$ & LR bank & bound (Prop. 2) & LR pooled$^\\ast$ & RA \\\\", "\\midrule"]
    for r in rows:
        lines.append(f"{r['pi']:.2f} & {r['I_R'] / r['I_P']:.1f} & {r['bank']['mean']:.0f} & {r['bound_bank']:.0f} & "
                     f"{r['pooled']['mean']:.0f} & {r['ra']['mean']:.0f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "supp_share.tex").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
def robust_tables():
    f = RES / "robust.json"
    if not f.exists():
        return
    Rb = json.loads(f.read_text())
    if "labels" in Rb:
        L = Rb["labels"]
        dets = [("bank", "LR bank"), ("pooled", "LR pooled$^\\ast$"), ("ph-bank", "PH bank$^\\ast$"),
                ("ddm-oci", "DDM-OCI"), ("lfr", "LFR")]
        fams = sorted({r["family"] for r in L}, key=["sea5", "gmm5", "credit"].index)
        lines = ["\\begin{tabular}{@{}ll*{" + str(len(dets)) + "}{r}@{}}", "\\toprule",
                 "Stream & $(\\rho, d)$ & " + " & ".join(l for _, l in dets) + " \\\\", "\\midrule"]
        for fam in fams:
            for rho, d in [(1.0, 0), (0.5, 0), (0.2, 0), (1.0, 1000), (1.0, 5000), (0.2, 5000)]:
                cells = []
                for dk, _ in dets:
                    x = [r for r in L if r["family"] == fam and r["rho"] == rho and r["d"] == d and r["det"] == dk]
                    cells.append(f"{x[0]['delay'] / 1000:.1f}" if x else "--")
                lines.append(f"{dict(zip(FAMS, FHEAD))[fam]} & ({rho:g}, {d}) & " + " & ".join(cells) + " \\\\")
            lines.append("\\midrule")
        lines[-1] = "\\bottomrule"
        lines.append("\\end{tabular}")
        (TAB / "supp_labels.tex").write_text("\n".join(lines) + "\n")
        g = lambda fam, rho, d, k: [r for r in L if r["family"] == fam and r["rho"] == rho and r["d"] == d
                                    and r["det"] == k][0]["delay"]
        b1, b05, b02 = (np.mean([g(f_, rho, 0, "bank") for f_ in fams]) for rho in (1.0, 0.5, 0.2))
        macro("LabRatioHalf", f"{b05 / b1:.1f}")
        macro("LabRatioFifth", f"{b02 / b1:.1f}")
        add = np.mean([g(f_, 1.0, 5000, "bank") - g(f_, 1.0, 0, "bank") for f_ in fams])
        macro("LabDelayAdd", f"{add / 1000:.1f}")
        rank_keep = all(g(f_, rho, d, "bank") <= min(g(f_, rho, d, k) for k, _ in dets if k not in ("bank", "ph-bank"))
                        for f_ in fams for rho in (1.0, 0.5, 0.2) for d in (0, 1000, 5000))
        macro("LabRankKept", "all" if rank_keep else "most")
    if "refmodel" in Rb:
        M = Rb["refmodel"]
        lines = ["\\begin{tabular}{@{}llrrrrrr@{}}", "\\toprule",
                 "Stream & Reference & BA & $\\mu_c$ & $p_c$ & bound & delay & 3 designs \\\\", "\\midrule"]
        for r in M:
            bd = "$\\infty$" if r["bound"] is None else f"{r['bound'] / 1000:.1f}k"
            lines.append(f"{dict(zip(FAMS, FHEAD))[r['family']]} & {r['ref']} & {r['ba_pre']:.2f} & {r['mu']:.2f} & "
                         f"{r['p']:.2f} & {bd} & {r['delay'] / 1000:.1f}k & {r['delay_multi'] / 1000:.1f}k \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        (TAB / "supp_ref.tex").write_text("\n".join(lines) + "\n")
        ok = [r for r in M if r["bound"] is not None]
        macro("RefWithin", f"{sum(r['delay'] <= r['bound'] for r in ok)}/{len(ok)}")
        macro("RefMuLo", f"{min(r['mu'] for r in M):.2f}")
        macro("RefMuHi", f"{max(r['mu'] for r in M):.2f}")
        worst = max(M, key=lambda r: r["delay"])
        macro("RefWorstDelay", f"{worst['delay'] / 1000:.0f}")
        macro("RefWorstName", worst["ref"])
        best = min(M, key=lambda r: r["delay"])
        macro("RefBestDelay", f"{best['delay'] / 1000:.1f}")
    if "prior" in Rb:
        P = {r["det"]: r["p_alarm_after"] for r in Rb["prior"]}
        C = {r["det"]: r["p_alarm_control"] for r in Rb["prior"]}
        for d, nm in (("bank", "Bank"), ("pooled", "Pool"), ("lfr", "Lfr"), ("ddm-oci", "Oci"),
                      ("ph-pooled", "PHPool"), ("rbmim", "Rbm"), ("ph-bank", "PHBank")):
            macro(f"Prior{nm}", f"{P[d]:.2f}")
            macro(f"PriorCtl{nm}", f"{C[d]:.2f}")
        lines = ["\\begin{tabular}{@{}lrr@{}}", "\\toprule", "Detector & no shift & prior shift \\\\", "\\midrule"]
        for d, lab in MAIN:
            if d in P:
                lines.append(f"{lab} & {C[d]:.2f} & {P[d]:.2f} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        (TAB / "supp_prior.tex").write_text("\n".join(lines) + "\n")
        return P


def retrain_table(prior):
    T = {f.stem: json.loads(f.read_text()) for f in (RES / "retrain").glob("*.json")}
    if not T:
        return
    pols_all = ["bank", "pooled", "ph-pooled", "ddm-pooled", "adwin-pooled", "ddm-oci", "lfr", "rbmim",
                "none", "periodic", "oracle"]
    fams = [x for x in ["sea5", "gauss5", "gmm5", "credit", "covtype"] if x in T and all(p in T[x] for p in pols_all)]
    print("retrain table uses", fams)
    for k in ("RetrainBestCount", "RetrainN", "RetrainGainNone", "RetrainGapOracle", "RetrainGainPool"):
        macro(k, "??")
    if not fams:
        return
    pols = [("bank", "LR bank"), ("pooled", "LR pooled$^\\ast$"), ("ph-pooled", "PH pooled$^\\ast$"),
            ("ddm-pooled", "DDM"), ("adwin-pooled", "ADWIN"), ("ddm-oci", "DDM-OCI"), ("lfr", "LFR"),
            ("rbmim", "RBM-IM$^\\dagger$"), ("none", "no retraining"), ("periodic", "periodic"),
            ("oracle", "oracle")]
    lines = ["\\begin{tabular}{@{}l*{" + str(len(fams)) + "}{r}" + ("r" if prior else "") + "@{}}", "\\toprule",
             "Policy & " + " & ".join(dict(zip(FAMS, FHEAD))[x] for x in fams)
             + (" & Prior \\\\" if prior else " \\\\"), "\\midrule"]
    ba = {p: [100 * np.mean([r["ba_post"] for r in T[x][p]]) for x in fams] for p, _ in pols if all(p in T[x] for x in fams)}
    nonor = [p for p in ba if p != "oracle"]
    for p, lab in pols:
        if p not in ba:
            continue
        cells = []
        for j in range(len(fams)):
            s = f"{ba[p][j]:.1f}"
            if p != "oracle" and round(ba[p][j], 1) >= max(round(ba[q][j], 1) for q in nonor):
                s = f"\\textbf{{{s}}}"
            cells.append(s)
        pr = ""
        if prior:
            pr = " & " + (f"{prior[p]:.2f}" if p in prior else "--")
        if p == "oracle":
            lines.append("\\midrule")
        lines.append(f"{lab} & " + " & ".join(cells) + pr + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "tab_retrain.tex").write_text("\n".join(lines) + "\n")
    best_disp = [max(round(ba[q][j], 1) for q in nonor) for j in range(len(fams))]
    macro("RetrainBestCount", str(sum(round(ba["bank"][j], 1) >= best_disp[j] for j in range(len(fams)))))
    macro("RetrainN", str(len(fams)))
    macro("RetrainGainNone", f"{np.mean(np.array(ba['bank']) - np.array(ba['none'])):.1f}")
    macro("RetrainGapOracle", f"{np.mean(np.array(ba['oracle']) - np.array(ba['bank'])):.1f}")
    macro("RetrainGainPool", f"{np.mean(np.array(ba['bank']) - np.array(ba['pooled'])):.1f}")


# --------------------------------------------------------------------------- #
# Additional studies (server runs): ARL sweep, scenarios, design, seeds, multi-change, Elec2, timing
# --------------------------------------------------------------------------- #
CORE = [("bank", "LR bank (ours)"), ("ph-bank", "PH bank$^\\ast$"), ("adwin-bank", "ADWIN bank"),
        ("pooled", "LR pooled$^\\ast$"), ("ph-pooled", "PH pooled$^\\ast$"), ("ra", "RA-CUSUM"),
        ("ddm-oci", "DDM-OCI"), ("lfr", "LFR"), ("rbmim", "RBM-IM$^\\dagger$")]


def _ranks(folder):
    """Average rank (over the 11 streams) of the RANKED detectors in one bench folder; None if incomplete."""
    fs = [RES / folder / f"{f}.json" for f in FAMS]
    if not all(f.exists() for f in fs):
        return None
    R = [json.loads(f.read_text()) for f in fs]
    if not all(all(d in r["det"] for d in RANKED) for r in R):
        return None
    M = np.array([[delays(r, d).mean() for r in R] for d in RANKED])
    ranks = np.array([stats.rankdata(M[:, j]) for j in range(M.shape[1])]).T
    return dict(zip(RANKED, ranks.mean(1))), M


def arl_sweep():
    rows = []
    for tag, folder in (("10^4", "bench_arl1e4"), ("5\\times10^4", "bench"), ("10^5", "bench_arl1e5"),
                        ("2\\times10^5", "bench_arl2e5")):
        out = _ranks(folder)
        if out is None:
            continue
        avg, M = out
        rows.append((tag, avg, M))
    if not rows:
        return
    lines = ["\\begin{tabular}{@{}l*{" + str(len(rows)) + "}{r}@{}}", "\\toprule",
             "Detector & " + " & ".join(f"ARL ${t}$" for t, _, _ in rows) + " \\\\", "\\midrule"]
    for d, lab in MAIN:
        if d == "bank-theory":
            continue
        cells = []
        for _, avg, _ in rows:
            v = f"{avg[d]:.2f}"
            cells.append(f"\\textbf{{{v}}}" if avg[d] <= min(avg.values()) + 1e-12 else v)
        lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "supp_arlsweep.tex").write_text("\n".join(lines) + "\n")
    best_all = all(min(avg, key=avg.get) == "bank" for _, avg, _ in rows)
    macro("SweepN", str(len(rows)))
    macro("SweepBankBest", "at every ARL" if best_all else "at " + ", ".join(
        f"${t}$" for t, avg, _ in rows if min(avg, key=avg.get) == "bank"))
    NONORACLE = [d for d in RANKED if not d.startswith("ph-") and d != "pooled"]
    nonor_first = all(min(NONORACLE, key=avg.get) == "bank" for _, avg, _ in rows)
    macro("SweepNonOracleFirst", "yes" if nonor_first else "no")
    second = [sorted(NONORACLE, key=avg.get)[1] for _, avg, _ in rows]
    macro("SweepNonOracleGapMin", f"{min(avg[s2] - avg['bank'] for (_, avg, _), s2 in zip(rows, second)):.2f}")
    hi = [t for t, avg, _ in rows if min(avg, key=avg.get) != "bank"]
    macro("SweepPHFirstAt", ", ".join(f"${t}$" for t in hi) if hi else "none")
    macro("SweepRankLo", f"{min(avg['bank'] for _, avg, _ in rows):.2f}")
    macro("SweepRankHi", f"{max(avg['bank'] for _, avg, _ in rows):.2f}")
    # delay versus ARL figure on four streams
    fams = [("sea5", "S$_5$"), ("gmm5", "M$_3$"), ("credit", "Cr."), ("insects", "In.")]
    arls = {"bench_arl1e4": 1e4, "bench": 5e4, "bench_arl1e5": 1e5, "bench_arl2e5": 2e5}
    sty = {"bank": ("k", "o", "-"), "ph-bank": ("tab:blue", "^", "-"), "pooled": ("tab:red", "v", "-"),
           "ph-pooled": ("tab:orange", "D", "-"), "ddm-oci": ("tab:brown", "s", "--"), "lfr": ("tab:green", "P", "--"),
           "rbmim": ("tab:purple", "x", "--")}
    fig, axs = plt.subplots(1, 4, figsize=(7.16, 1.6), sharey=False)
    for ax, (f, fl) in zip(axs, fams):
        for d, (c, mk, ls) in sty.items():
            xs, ys = [], []
            for folder, g in arls.items():
                p = RES / folder / f"{f}.json"
                if p.exists():
                    r = json.loads(p.read_text())
                    if d in r["det"]:
                        xs.append(g)
                        ys.append(delays(r, d).mean())
            if xs:
                ax.plot(xs, ys, color=c, marker=mk, ls=ls, ms=2.5, lw=0.8, label=dict(MAIN)[d])
        ax.set(xscale="log", yscale="log", title=fl, xlabel="ARL")
        ax.title.set_fontsize(7)
        ax.tick_params(labelsize=5.5)
        ax.xaxis.label.set_fontsize(6.5)
        ax.grid(alpha=0.25, lw=0.4)
    axs[0].set_ylabel("mean delay", fontsize=6.5)
    axs[-1].legend(fontsize=4.5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout(pad=0.2, w_pad=0.4)
    fig.savefig(FIG / "fig_arl.pdf")
    fig.savefig(FIG / "fig_arl.png", dpi=200)


def scenarios():
    fams = [("sea5grad", "S$_5$, gradual"), ("gauss5grad", "G$_5$, gradual"), ("gmm5two", "M$_3$, two classes"),
            ("sea5maj", "S$_5$, majority"), ("gauss5maj", "G$_5$, majority"), ("sea5wc1k", "S$_5$, $W_{\\mathrm{cal}}=10^3$"),
            ("sea5", "S$_5$, $W_{\\mathrm{cal}}=3\\times10^3$"), ("sea5wc10k", "S$_5$, $W_{\\mathrm{cal}}=10^4$")]
    fams = [(f, l) for f, l in fams if (RES / "bench" / f"{f}.json").exists()]
    if not fams:
        return
    R = {f: bench(f) for f, _ in fams}
    lines = ["\\begin{tabular}{@{}l*{" + str(len(CORE)) + "}{r}@{}}", "\\toprule",
             "Stream & " + " & ".join(l for _, l in CORE) + " \\\\", "\\midrule"]
    for f, lab in fams:
        m = {d: delays(R[f], d).mean() / 1000 for d, _ in CORE}
        best = min(round(v, 1) for v in m.values())
        cells = [(f"\\textbf{{{m[d]:.1f}}}" if round(m[d], 1) <= best else f"{m[d]:.1f}") for d, _ in CORE]
        lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "supp_scen.tex").write_text("\n".join(lines) + "\n")
    g = lambda f, d: delays(R[f], d).mean() / 1000
    macro("ScenGradBank", f"{g('sea5grad', 'bank'):.1f}")
    macro("ScenGradBestOther", f"{min(g('sea5grad', d) for d in ('pooled', 'ph-pooled', 'ra', 'ddm-oci', 'lfr', 'rbmim')):.1f}")
    macro("ScenMajTie", "yes" if abs(g("sea5maj", "bank") - g("sea5maj", "pooled")) < 0.15 else "no")
    macro("ScenMajBank", f"{g('sea5maj', 'bank'):.1f}")
    macro("ScenMajPool", f"{g('sea5maj', 'pooled'):.1f}")
    macro("ScenWcOneBank", f"{g('sea5wc1k', 'bank'):.1f}")
    macro("ScenWcOnePH", f"{g('sea5wc1k', 'ph-bank'):.1f}")
    macro("ScenWcTenBank", f"{g('sea5wc10k', 'bank'):.1f}")


def design_r():
    rows = []
    for f, fh in zip(FAMS, FHEAD):
        r = bench(f)
        if not all(d in r["det"] for d in ("bank-r1.5", "bank-r3", "bank-multi")):
            return
        rows.append((fh, [delays(r, d).mean() / 1000 for d in ("bank-r1.5", "bank", "bank-r3", "bank-multi")]))
    lines = ["\\begin{tabular}{@{}l*{4}{r}@{}}", "\\toprule",
             "Stream & $r=1.5$ & $r=2$ & $r=3$ & $r\\in\\{1.25,2,4\\}$ \\\\", "\\midrule"]
    for fh, v in rows:
        lines.append(f"{fh} & " + " & ".join(f"{x:.1f}" for x in v) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "supp_design.tex").write_text("\n".join(lines) + "\n")
    ratio = [v[1] / min(v) for _, v in rows]
    macro("DesignWorstRatio", f"{max(ratio):.2f}")
    macro("DesignMedRatio", f"{np.median(ratio):.2f}")


def seeds100():
    fs = [RES / "bench_100seeds" / f"{f}.json" for f in ("sea2", "sea1")]
    if not all(f.exists() for f in fs):
        return
    lines = ["\\begin{tabular}{@{}ll*{3}{r}@{}}", "\\toprule",
             "Stream & Detector & Delay [95\\% CI] & Det. & ARL$_{\\mathrm{eval}}$ \\\\", "\\midrule"]
    for f, fh in zip(fs, ("S$_2$", "S$_1$")):
        r = json.loads(f.read_text())
        for d, lab in CORE:
            x = delays(r, d)
            lo, hi = boot_ci(x)
            det = sum(z["detected"] for z in r["det"][d]["runs"])
            fa = sum(z["fa"] for z in r["det"][d]["null"])
            arl_e = f"{r['n_mon'] * len(r['det'][d]['null']) / max(fa, 1) / 1000:.0f}k"
            lines.append(f"{fh} & {lab} & {x.mean() / 1000:.1f} [{lo / 1000:.1f}, {hi / 1000:.1f}] & {det} & {arl_e} \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    (TAB / "supp_seeds100.tex").write_text("\n".join(lines) + "\n")


def multichange():
    names = [("seamulti", "S$_5$, 4 changes"), ("gaussmulti", "G$_5$, 4 changes"), ("insectsnat", "INSECTS, natural order")]
    names = [(s, l) for s, l in names if (RES / "multichange" / f"{s}.json").exists()]
    if not names:
        return
    lines = ["\\begin{tabular}{@{}ll*{3}{r}@{}}", "\\toprule",
             "Stream & Detector & Detected (\\%) & Delay ($\\times10^3$) & FA / $10^5$ \\\\", "\\midrule"]
    summ = {}
    for s, lab in names:
        M = json.loads((RES / "multichange" / f"{s}.json").read_text())
        for d, dl in CORE:
            if d not in M:
                continue
            runs = M[d]
            det = 100 * np.mean([np.mean(r["detected"]) for r in runs])
            de = np.mean([np.mean(r["delays"]) for r in runs]) / 1000
            fa = np.mean([r["false_alarms"] / r["n"] * 1e5 for r in runs])
            summ[(s, d)] = (det, de, fa)
            lines.append(f"{lab} & {dl} & {det:.0f} & {de:.1f} & {fa:.2f} \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    (TAB / "supp_multi.tex").write_text("\n".join(lines) + "\n")
    if ("insectsnat", "bank") in summ:
        b = summ[("insectsnat", "bank")]
        macro("InsNatBankDet", f"{b[0]:.0f}")
        macro("InsNatBankDelay", f"{b[1]:.1f}")
        others = [summ[("insectsnat", d)] for d, _ in CORE if d != "bank" and ("insectsnat", d) in summ]
        macro("InsNatRank", str(1 + sum(o[1] < b[1] for o in others)))
        macro("InsNatN", str(1 + len(others)))
        macro("InsNatPool", f"{summ[('insectsnat', 'pooled')][1]:.1f}")
        macro("InsNatRbm", f"{summ[('insectsnat', 'rbmim')][1]:.1f}")
    for s, nm in (("seamulti", "SeaMulti"), ("gaussmulti", "GaussMulti")):
        if (s, "bank") in summ:
            b = summ[(s, "bank")]
            others = [summ[(s, d)] for d, _ in CORE if d != "bank" and (s, d) in summ]
            macro(f"{nm}BankDet", f"{b[0]:.0f}")
            macro(f"{nm}Rank", str(1 + sum(o[1] < b[1] for o in others)))


def extras():
    f = RES / "extras.json"
    if not f.exists():
        return
    E = json.loads(f.read_text())
    if "elec2" in E:
        e = E["elec2"]
        lines = ["\\begin{tabular}{@{}lrrr@{}}", "\\toprule",
                 "Order & lag-1 corr. & LR bank & LR pooled \\\\", "\\midrule"]
        for k, lab in (("natural", "original"), ("block", "blocks of $10^3$ permuted"), ("shuffled", "fully permuted")):
            lines.append(f"{lab} & {e[k]['lag1']:.2f} & {e[k]['bank_alarms_per_1e4']:.1f} & {e[k]['pooled_alarms_per_1e4']:.1f} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        (TAB / "supp_elec2.tex").write_text("\n".join(lines) + "\n")
        macro("ElecLag", f"{e['natural']['lag1']:.2f}")
        macro("ElecLagShuf", f"{e['shuffled']['lag1']:.2f}")
        macro("ElecAlarmsNat", f"{e['natural']['bank_alarms_per_1e4']:.0f}")
        macro("ElecAlarmsBlock", f"{e['block']['bank_alarms_per_1e4']:.0f}")
        macro("ElecAlarmsShuf", f"{e['shuffled']['bank_alarms_per_1e4']:.0f}")
    if "timing" in E:
        t = E["timing"]
        lines = ["\\begin{tabular}{@{}lr@{}}", "\\toprule", "Detector & ns/sample \\\\", "\\midrule"]
        for d, lab in MAIN + [("bank-multi", "LR bank, 3 designs")]:
            if d in t:
                lines.append(f"{lab} & {1000 * t[d]:.0f} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        (TAB / "supp_timing.tex").write_text("\n".join(lines) + "\n")
        macro("TimeBank", f"{1000 * t['bank']:.0f}")
        macro("TimeRbm", f"{t['rbmim']:.1f}")


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    R, *_ = table_bench()
    table_supp_bench(R)
    localisation(R)
    fig_sim()
    prior = robust_tables()
    retrain_table(None)
    for fn in (arl_sweep, scenarios, design_r, seeds100, multichange, extras):
        fn()
    macro("ARL", "5\\times10^4")
    body = "% generated by Code_v2/v2/report.py -- do not edit\n" + "".join(
        f"\\newcommand{{\\n{k}}}{{{v}}}\n" for k, v in N.items())
    (TAB / "numbers.tex").write_text(body)
    for k, v in N.items():
        print(f"{k:22s} {v}")


if __name__ == "__main__":
    main()
