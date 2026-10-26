"""Statistical procedures of (participant = unit of analysis)."""
import itertools

import numpy as np
from scipy import optimize, stats


def rank_pratt(d):
    """Pratt ranking: rank |d| including zeros (midranks), then zeros contribute nothing."""
    d = np.asarray(d, float)
    r = stats.rankdata(np.abs(d))
    return r, d != 0


def wilcoxon_exact(d):
    """Two-sided exact signed-rank test by sign-flip enumeration (Pratt zeros, midrank ties).
    Returns (p, T_plus, T_minus)."""
    r, nz = rank_pratt(d)
    rn, dn = r[nz], np.asarray(d, float)[nz]
    m = len(rn)
    t_plus = rn[dn > 0].sum()
    t_minus = rn[dn < 0].sum()
    if m == 0:
        return 1.0, 0.0, 0.0
    signs = ((np.arange(2 ** m)[:, None] >> np.arange(m)) & 1).astype(float)
    tp = signs @ rn
    centre = rn.sum() / 2
    obs = abs(t_plus - centre)
    p = float(np.mean(np.abs(tp - centre) >= obs - 1e-12))
    return min(1.0, p), float(t_plus), float(t_minus)


def rank_biserial(d):
    _, tp, tm = wilcoxon_exact(d)
    return (tp - tm) / (tp + tm) if tp + tm > 0 else 0.0


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, (len(p) - i) * p[j])
        adj[j] = min(1.0, running)
    return adj


def iman_davenport(R):
    """R: (n participants, k models). Friedman chi2 and Iman–Davenport F with p."""
    n, k = R.shape
    chi2, _ = stats.friedmanchisquare(*[R[:, j] for j in range(k)])
    F = (n - 1) * chi2 / (n * (k - 1) - chi2) if n * (k - 1) - chi2 > 0 else np.inf
    p = float(stats.f.sf(F, k - 1, (k - 1) * (n - 1)))
    return float(chi2), float(F), p


def t_ci(d, level):
    d = np.asarray(d, float)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.5 + level / 2, n - 1) * se
    return float(d.mean() - h), float(d.mean() + h)


def delta_star(d):
    lo, hi = t_ci(d, 0.90)
    return max(abs(lo), abs(hi))


def tost_p(d, m):
    """Paired t-TOST p-value at symmetric margin m (equivalent to the 90% CI lying inside ±m)."""
    d = np.asarray(d, float)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    p_lo = stats.t.sf((d.mean() + m) / se, n - 1)
    p_hi = stats.t.cdf((d.mean() - m) / se, n - 1)
    return float(max(p_lo, p_hi))


def mde_difference(sd, n, alpha=0.05, power=0.80):
    """Smallest mean paired difference detectable by a two-sided paired t-test (noncentral t, exact).
    Solved on the noncentrality scale (depends only on df), then scaled by sd / sqrt(n)."""
    df = n - 1
    tc = stats.t.ppf(1 - alpha / 2, df)

    def pw(nc):
        return stats.nct.sf(tc, df, nc) + stats.nct.cdf(-tc, df, nc) - power

    nc = optimize.brentq(pw, 1e-6, 8.0)
    return float(nc * sd / np.sqrt(n))


def mde_equivalence(sd, n, alpha=0.05, power=0.80):
    """Smallest symmetric margin with the given paired-TOST power when the true difference is 0
    (normal approximation with the SD treated as known)."""
    return float((stats.t.ppf(1 - alpha, n - 1) + stats.norm.ppf(0.5 + power / 2)) * sd / np.sqrt(n))


def bca_mean(x, seed=20260601, n_resamples=5000, level=0.95):
    x = np.asarray(x, float)
    r = stats.bootstrap((x,), np.mean, n_resamples=n_resamples, method="BCa", confidence_level=level,
                        random_state=np.random.default_rng(seed))
    return float(r.confidence_interval.low), float(r.confidence_interval.high)


def classify_temporal_tree(d, p_holm, m_strict, m_lenient):
    """(d = per-participant Temporal - Tree differences)."""
    ds = delta_star(d)
    lo95, hi95 = t_ci(d, 0.95)
    sign = "Temporal" if np.mean(d) < 0 else "Tree"
    if ds < m_strict:
        return "O1", ("statistically detectable" if p_holm < 0.05 else "")
    if p_holm < 0.05 and (hi95 < 0 or lo95 > 0):
        return f"O2-{sign}", ("within the lenient margin" if ds < m_lenient else "")
    if ds < m_lenient:
        return "O1L", ("p<0.05 but 95% CI includes 0" if p_holm < 0.05 else "")
    return "O3", ("p<0.05 but 95% CI includes 0" if p_holm < 0.05 else "")


def r1_class(d, p_holm):
    """d = per-participant (ML family - Linear) differences in RMSE."""
    if p_holm < 0.05:
        return "ML better" if np.mean(d) < 0 else "linear better"
    return "unresolved"


def pairwise(R, names):
    """All model pairs: exact Wilcoxon p, Holm within the set, rank-biserial, mean difference."""
    rows = []
    for (i, a), (j, b) in itertools.combinations(enumerate(names), 2):
        d = R[:, i] - R[:, j]
        p, _, _ = wilcoxon_exact(d)
        rows.append({"a": a, "b": b, "mean_diff": float(d.mean()), "p": p, "rank_biserial": rank_biserial(d)})
    adj = holm([r["p"] for r in rows])
    for r, q in zip(rows, adj):
        r["p_holm"] = float(q)
    return rows
