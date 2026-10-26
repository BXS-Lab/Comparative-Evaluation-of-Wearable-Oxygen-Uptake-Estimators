"""Cross-dataset outputs at the adopted budget K = 100 (manuscript Table 5 and Fig. 10).

Reads results_r1/K100/<DATASET>/analysis/ (primary) and results_r1/<DATASET>/analysis/ (25-trial sensitivity).
Writes results_r1/K100/{table5.csv, replication.csv, budget_sensitivity.csv, fig10_forest.png, fig10_forest.pdf}.
Usage: python scripts_r1/summarize_k100.py"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results_r1"
OUT = RES / "K100"
MAIN = ["BEL", "WEEE", "ING"]
ALL = MAIN + ["BEL_CORE", "WEEE_HRQ"]
LABEL = {"BEL": "BEL (n = 16)", "WEEE": "WEEE (n = 15)", "ING": "ING (n = 10)", "BEL_CORE": "BEL core set (n = 16)", "WEEE_HRQ": "WEEE HR-screened (n = 11)"}
CONTRASTS = ["Temporal - Tree", "Temporal - Linear", "Tree - Linear"]
EQUIV = {"O1", "O1L"}


def read(dataset, root=OUT):
    a = Path(root) / dataset / "analysis"
    return {"summary": json.loads((a / "summary.json").read_text()), "lb": pd.read_csv(a / "leaderboard.csv"),
            "fc": pd.read_csv(a / "family_contrasts.csv").set_index("contrast"),
            "fam": pd.read_csv(a / "family_per_participant_rmse.csv", index_col=0),
            "tr": pd.read_csv(a / "transition.csv").set_index("family"), "ba": pd.read_csv(a / "bland_altman.csv")}


def table5(R):
    rows = []
    for d in ALL:
        r, s = R[d], R[d]["summary"]
        row = {"dataset": d, "n": s["n_participants"], "sd_vo2": s["sd_y"], "m_strict": s["m_strict"],
               "m_lenient": s["m_lenient"], "omnibus_p": s["omnibus_p"], "n_pairs_holm_sig": s["n_pairs_holm_sig"]}
        for f, v in r["fam"].mean().items():
            row[f"rmse_{f}"] = v
        best = r["lb"].sort_values("rmse").iloc[0]
        row["best_model"], row["best_rmse"] = best["model"], best["rmse"]
        for c in CONTRASTS:
            x, k = r["fc"].loc[c], c.replace(" - ", "_minus_")
            row.update({f"{k}_diff": x.mean_diff, f"{k}_ci95_lo": x.ci95_lo, f"{k}_ci95_hi": x.ci95_hi,
                        f"{k}_delta_star": x.delta_star, f"{k}_p_holm": x.p_holm, f"{k}_class": x["class"]})
        for f in ["Linear", "Tree", "Temporal"]:
            t = r["tr"].loc[f]
            row.update({f"penalty_{f}": t.penalty_gmean, f"penalty_{f}_lo": t.penalty_ci_lo, f"penalty_{f}_hi": t.penalty_ci_hi,
                        f"R3_{f}": t.R3_class, f"claim_{f}": bool(t.claim_concentrates_in_transitions)})
        row["ppt_bias_min"], row["ppt_bias_max"] = r["ba"].ppt_bias_min.min(), r["ba"].ppt_bias_max.max()
        rows.append(row)
    return pd.DataFrame(rows)


def _side(x):
    """(kind, sign, detectable) of one dataset's contrast row; kind in {'eq', 'diff', 'unres'}."""
    c, sign = x["class"], ("first" if x.mean_diff < 0 else "second")
    if c in ("unresolved", "O3"):
        return "unres", None, False
    if c == "ML better":
        return "diff", "first", True
    if c == "linear better":
        return "diff", "second", True
    if c.startswith("O2-"):
        return "diff", ("first" if c == "O2-Temporal" else "second"), True
    sub = x.get("sublabel")
    detectable = (c == "O1" and sub == "statistically detectable") or (c == "O1L" and x.p_holm < 0.05)
    return "eq", sign, detectable


def r1_r2_status(bel, ext):
    """taxonomy for one contrast (rows of family_contrasts.csv)."""
    kb, sb, db = _side(bel)
    ke, se, de = _side(ext)
    if kb == "unres":
        return "BEL unresolved"
    if ke == "unres":
        return "not resolved externally"
    if kb == ke == "eq":
        return "replicated"
    if kb == ke == "diff":
        return "replicated" if sb == se else "contradicted"
    eq_detectable, eq_sign, diff_sign = (db, sb, se) if kb == "eq" else (de, se, sb)
    return "direction replicated, magnitude differs" if eq_detectable and eq_sign == diff_sign else "contradicted"


def replication(R):
    rows = []
    b = R["BEL"]
    for d in [x for x in R if x != "BEL"]:
        e = R[d]
        for c, crit in [("Temporal - Linear", "R1"), ("Tree - Linear", "R1"), ("Temporal - Tree", "R2")]:
            rows.append({"dataset": d, "criterion": crit, "contrast": c, "BEL": b["fc"].loc[c, "class"],
                         "external": e["fc"].loc[c, "class"], "status": r1_r2_status(b["fc"].loc[c], e["fc"].loc[c])})
        for f in ["Linear", "Tree", "Temporal"]:
            pb, pe = b["tr"].loc[f, "penalty_gmean"], e["tr"].loc[f, "penalty_gmean"]
            rows.append({"dataset": d, "criterion": "R3", "contrast": f, "BEL": f"{pb:.2f} {b['tr'].loc[f, 'R3_class']}",
                         "external": f"{pe:.2f} {e['tr'].loc[f, 'R3_class']}",
                         "status": "replicated" if (pb > 1) == (pe > 1) else "contradicted"})
    return pd.DataFrame(rows)


def budget_sensitivity(R):
    rows = []
    for d in ALL:
        a25 = RES / d / "analysis"
        if not (a25 / "family_contrasts.csv").exists():
            continue
        r25 = read(d, RES)
        lb = r25["lb"].set_index("model").rmse.rename("rmse_25").to_frame().join(R[d]["lb"].set_index("model").rmse.rename("rmse_100"))
        for m, x in lb.iterrows():
            rows.append({"dataset": d, "item": m, "trials_25": x.rmse_25, "trials_100": x.rmse_100})
        for c in CONTRASTS:
            for q in ["mean_diff", "delta_star", "p_holm", "class"]:
                rows.append({"dataset": d, "item": f"{c} {q}", "trials_25": r25["fc"].loc[c, q], "trials_100": R[d]["fc"].loc[c, q]})
    return pd.DataFrame(rows)


def forest(R):
    """Fig. 10 draft: family contrasts in SD units for the three datasets (90%/95% CI, strict/lenient margins)."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3), sharey=True)
    order = MAIN[::-1]
    for k, (ax, c) in enumerate(zip(axes, CONTRASTS)):
        for i, d in enumerate(order):
            x, sd = R[d]["fc"].loc[c], R[d]["summary"]["sd_y"]
            ax.plot([x.ci95_lo / sd, x.ci95_hi / sd], [i, i], color="k", lw=1)
            ax.plot([x.ci90_lo / sd, x.ci90_hi / sd], [i, i], color="k", lw=3.5, solid_capstyle="butt")
            ax.plot(x.mean_diff / sd, i, "o", color="white", mec="k", ms=5, zorder=3)
            lab = {"O1": "equivalent (strict)", "O1L": "equivalent (lenient)", "O3": "not resolved",
                   "unresolved": "not resolved", "ML better": "ML better", "O2-Tree": "Tree better",
                   "O2-Temporal": "Temporal better"}.get(x["class"], x["class"])
            ax.text(0.155, i, lab + f"\nΔ*/SD {x.delta_star / sd:.3f}", va="center", fontsize=7)
        ax.axvspan(-0.10, 0.10, color="0.92", zorder=0)
        ax.axvspan(-0.05, 0.05, color="0.82", zorder=0)
        ax.axvline(0, color="0.4", lw=0.6)
        ax.set_xlim(-0.16, 0.27)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_title(c.replace(" - ", " − "), fontsize=10)
        ax.set_xlabel("Paired difference / SD($\\dot{V}$O$_2$)", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([f"{LABEL[d]}\nSD {R[d]['summary']['sd_y']:.2f}" for d in order], fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig10_forest.{ext}", dpi=200, bbox_inches="tight")


def w4b(d):
    f = OUT / d / "analysis" / "w4b_summary.csv"
    return pd.read_csv(f) if f.exists() else None


def fig5_gain():
    """Fig. 5 draft: Ridge gain(d) with BCa 95% CI for the three datasets (BEL primary)."""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for d, mk in zip(MAIN, ["o", "s", "^"]):
        t = w4b(d)
        t = t[t.learner == "ridge"].sort_values("d")
        line, = ax.plot(t.d, t.gain_mean, marker=mk, ms=4, lw=1.2, label=LABEL[d])
        ax.fill_between(t.d, t.gain_bca_lo, t.gain_bca_hi, alpha=0.15, color=line.get_color(), lw=0)
    ax.axhline(0, color="0.4", lw=0.6)
    ax.set_xlabel("Staleness of the V̇O$_2$ history, d (s)", fontsize=8)
    ax.set_ylabel("gain(d) = 1 − RMSE$_{hist+sens}$/RMSE$_{hist}$", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig5_gain.{ext}", dpi=200, bbox_inches="tight")


def table5_protocol(R):
    """Table 5 organised by the four evaluation items (markdown)."""
    T = table5(R).set_index("dataset")
    rep = replication({d: R[d] for d in MAIN})
    def g(d, dd):
        t = w4b(d)
        if t is None:
            return "—"
        t = t[(t.learner == "ridge") & (t.d == dd)].iloc[0]
        return f"{t.gain_mean:.2f} [{t.gain_bca_lo:.2f}, {t.gain_bca_hi:.2f}]"
    def cls(d, c):
        k = c.replace(" - ", "_minus_")
        sub = R[d]["fc"].loc[c, "sublabel"]
        sub = f" ({sub})" if isinstance(sub, str) and sub else ""
        return (f"{T.loc[d, k + '_diff']:+.2f} ({T.loc[d, k + '_ci95_lo']:.2f}, {T.loc[d, k + '_ci95_hi']:.2f}); "
                f"Δ* {T.loc[d, k + '_delta_star']:.2f}; p_Holm {T.loc[d, k + '_p_holm']:.3f}; {T.loc[d, k + '_class']}{sub}")
    def rstat(d):
        if d == "BEL":
            return "reference"
        r = rep[rep.dataset == d].set_index("contrast")["status"]
        return f"R1 T−L: {r['Temporal - Linear']}; R1 Tr−L: {r['Tree - Linear']}; R2: {r['Temporal - Tree']}; R3: {', '.join(sorted(set(r[['Linear', 'Tree', 'Temporal']])))}"
    rows = [
        ("Dataset", "n; SD(V̇O₂)", lambda d: f"{int(T.loc[d, 'n'])}; {T.loc[d, 'sd_vo2']:.2f}"),
        ("Dataset", "Family mean RMSE (Linear / Tree / Temporal)", lambda d: f"{T.loc[d, 'rmse_Linear']:.2f} / {T.loc[d, 'rmse_Tree']:.2f} / {T.loc[d, 'rmse_Temporal']:.2f}"),
        ("Dataset", "Family mean RMSE / SD(V̇O₂)", lambda d: f"{T.loc[d, 'rmse_Linear'] / T.loc[d, 'sd_vo2']:.2f} / {T.loc[d, 'rmse_Tree'] / T.loc[d, 'sd_vo2']:.2f} / {T.loc[d, 'rmse_Temporal'] / T.loc[d, 'sd_vo2']:.2f}"),
        ("1 Baseline-relative", "Ridge gain(d), d = 10 s", lambda d: g(d, 10.0)),
        ("1 Baseline-relative", "Ridge gain(d), d = 30 s", lambda d: g(d, 30.0)),
        ("1 Baseline-relative", "Ridge gain(d), d = 60 s", lambda d: g(d, 60.0)),
        ("2 Transition", "Penalty Linear / Tree / Temporal (R3 class)", lambda d: f"{T.loc[d, 'penalty_Linear']:.2f} / {T.loc[d, 'penalty_Tree']:.2f} / {T.loc[d, 'penalty_Temporal']:.2f} ({T.loc[d, 'R3_Temporal']})"),
        ("3 Multiplicity / equivalence", "Margins strict / lenient (0.05 / 0.10 × SD)", lambda d: f"{T.loc[d, 'm_strict']:.2f} / {T.loc[d, 'm_lenient']:.2f}"),
        ("3 Multiplicity / equivalence", "Omnibus p (Iman–Davenport)", lambda d: f"{T.loc[d, 'omnibus_p']:.2g}"),
        ("3 Multiplicity / equivalence", "Temporal − Tree", lambda d: cls(d, "Temporal - Tree")),
        ("3 Multiplicity / equivalence", "Temporal − Linear", lambda d: cls(d, "Temporal - Linear")),
        ("3 Multiplicity / equivalence", "Tree − Linear", lambda d: cls(d, "Tree - Linear")),
        ("4 Agreement", "Per-participant bias range (all models)", lambda d: f"{T.loc[d, 'ppt_bias_min']:+.1f} to {T.loc[d, 'ppt_bias_max']:+.1f}"),
        ("Replication vs BEL", "replication class", rstat),
    ]
    lines = ["| Protocol item | Quantity | " + " | ".join(LABEL[d] for d in MAIN) + " |", "|---|---|" + "---|" * len(MAIN)]
    for item, q, fn in rows:
        lines.append(f"| {item} | {q} | " + " | ".join(fn(d) for d in MAIN) + " |")
    notes = [
        "", "Notes. Regime markers: BEL causal cadence detector (pooled terciles); WEEE and ING protocol timing (0–90 s after a step = transition; last 180 s of a condition ≥ 300 s = steady).",
        "Absolute RMSE is not comparable across datasets (different V̇O₂ range, devices, activities); compare RMSE/SD and classes. Family contrasts: Holm over 3 contrasts, exact Wilcoxon; CIs are t-based, so CI and p_Holm can disagree (BEL Tree − Linear; ING Temporal − Linear).",
        "Pairwise model tests only on BEL (WEEE omnibus not rejected; ING n < 12). Sensitivities (Supplement): WEEE HR-screened (n = 11) Temporal − Tree O3, same direction; BEL core set O2-Temporal.",
    ]
    (OUT / "table5_protocol.md").write_text("\n".join(lines + notes) + "\n", encoding="utf-8")


def main():
    R = {d: read(d) for d in ALL}
    table5(R).to_csv(OUT / "table5.csv", index=False)
    replication({d: R[d] for d in MAIN}).to_csv(OUT / "replication.csv", index=False)
    budget_sensitivity(R).to_csv(OUT / "budget_sensitivity.csv", index=False)
    forest(R)
    fig5_gain()
    table5_protocol(R)
    print(replication({d: R[d] for d in MAIN}).to_string(index=False))


if __name__ == "__main__":
    main()
