"""
Sensitivity check: how much do the LINEAR leaderboard RMSEs move if their
regularization parameter is selected GLOBALLY (one alpha minimizing the 16-fold
LOPO mean RMSE, exactly like RF/XGB/temporal) instead of the current NESTED
inner-5-fold selection?

Preprocessing is identical to experiments/tuning/linear_models_perfold.py
(0.01 Hz Butterworth on the pooled series, per-fold standardization of X and y).
Only the alpha-selection protocol differs between the two columns, so the delta
isolates the nested-vs-global tuning effect (NOT the standardization effect the
paper's <0.02 note refers to).

Run from the repository root:
    python experiments/pmea_r1_matched_tuning/linear_global_vs_nested.py
"""
import warnings
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")

DATA = "data/TB-File-01.csv"


def butter_lowpass_filter(data, cutoff=0.01, fs=1.0, order=2):
    b, a = butter(order, cutoff / (0.5 * fs), btype="low", analog=False)
    return filtfilt(b, a, data, axis=0)


def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)
    pidx = df["ID"].values
    y = df["VO2"].values / df["Weight"].values
    N = len(df)
    X = np.zeros((N, 6))
    X[:, 0] = df["BF"].values
    X[:, 1] = df["VE"].values
    X[:, 2] = df["ACC"].values
    X[:, 3] = df["HR"].values
    X[:, 4] = df["CAD"].values
    for pid in np.unique(pidx):
        idx = pidx == pid
        hr = X[idx, 3]
        d = np.zeros(len(hr)); d[1:] = hr[1:] - hr[:-1]
        X[idx, 5] = d
    X = butter_lowpass_filter(X, cutoff=0.01, fs=1.0, order=2)
    return pidx, X, y, np.unique(pidx)


def fold_indices(pidx, parts):
    for test_pid in parts:
        tr = pidx != test_pid
        te = ~tr
        yield tr, te


def lopo_per_participant_rmse(make_model, pidx, X, y, parts):
    """Per-fold standardization (fit on train), fixed model, return 16 per-participant RMSEs."""
    rmses = []
    for tr, te in fold_indices(pidx, parts):
        scX, scY = StandardScaler(), StandardScaler()
        Xtr = scX.fit_transform(X[tr]); ytr = scY.fit_transform(y[tr].reshape(-1, 1)).ravel()
        Xte = scX.transform(X[te])
        m = make_model(); m.fit(Xtr, ytr)
        yp = scY.inverse_transform(m.predict(Xte).reshape(-1, 1)).ravel()
        rmses.append(np.sqrt(mean_squared_error(y[te], yp)))
    return np.array(rmses)


def nested_per_participant_rmse(factory, grid, pidx, X, y, parts):
    """Inner 5-fold GridSearchCV inside each LOPO fold (matches linear_models_perfold.py)."""
    rmses, chosen = [], []
    for tr, te in fold_indices(pidx, parts):
        scX, scY = StandardScaler(), StandardScaler()
        Xtr = scX.fit_transform(X[tr]); ytr = scY.fit_transform(y[tr].reshape(-1, 1)).ravel()
        Xte = scX.transform(X[te])
        gs = GridSearchCV(factory(), grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
        gs.fit(Xtr, ytr)
        m = gs.best_estimator_
        yp = scY.inverse_transform(m.predict(Xte).reshape(-1, 1)).ravel()
        rmses.append(np.sqrt(mean_squared_error(y[te], yp)))
        chosen.append(gs.best_params_)
    return np.array(rmses), chosen


def global_select(factory, param_combos, pidx, X, y, parts):
    """Pick ONE param combo minimizing the mean 16-fold LOPO RMSE (select-on-test)."""
    best = None
    for combo in param_combos:
        r = lopo_per_participant_rmse(lambda c=combo: factory().set_params(**c), pidx, X, y, parts)
        mean_r = r.mean()
        if best is None or mean_r < best[0]:
            best = (mean_r, combo, r)
    return best  # (mean_rmse, chosen_combo, per_participant_rmses)


def main():
    pidx, X, y, parts = load_and_prepare(DATA)
    print(f"[data] N={len(y)} participants={len(parts)} features={X.shape[1]}\n")

    ridge_grid = {"alpha": np.logspace(-3, 6, 30)}
    lasso_grid = {"alpha": np.logspace(-4, 3, 30)}
    en_grid = {"alpha": np.logspace(-4, 3, 20),
               "l1_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]}

    def combos(grid):
        keys = list(grid.keys())
        if len(keys) == 1:
            return [{keys[0]: v} for v in grid[keys[0]]]
        out = []
        for a in grid["alpha"]:
            for l in grid["l1_ratio"]:
                out.append({"alpha": a, "l1_ratio": l})
        return out

    models = {
        "OLS":        (lambda: LinearRegression(), None, None),
        "Ridge":      (lambda: Ridge(random_state=42), ridge_grid, combos(ridge_grid)),
        "Lasso":      (lambda: Lasso(max_iter=10000, tol=1e-6, random_state=42), lasso_grid, combos(lasso_grid)),
        "ElasticNet": (lambda: ElasticNet(max_iter=10000, tol=1e-6, random_state=42), en_grid, combos(en_grid)),
    }

    print(f"{'Model':<11}{'Nested RMSE':>13}{'Global RMSE':>13}{'Delta':>10}{'maxΔ/subj':>11}  notes")
    print("-" * 86)
    rows = []
    for name, (factory, grid, param_combos) in models.items():
        if grid is None:  # OLS has no hyperparameter
            r = lopo_per_participant_rmse(factory, pidx, X, y, parts)
            nested_mean = global_mean = r.mean()
            maxd = 0.0
            note = "no alpha (identical)"
            gsel = "-"
            nrange = "-"
        else:
            nr, chosen = nested_per_participant_rmse(factory, grid, pidx, X, y, parts)
            nested_mean = nr.mean()
            gmean, gcombo, grp = global_select(factory, param_combos, pidx, X, y, parts)
            global_mean = gmean
            maxd = np.max(np.abs(nr - grp))
            alphas = [c["alpha"] for c in chosen]
            nrange = f"alpha[{min(alphas):.2g},{max(alphas):.2g}]"
            if "l1_ratio" in gcombo:
                gsel = f"a={gcombo['alpha']:.2g},l1={gcombo['l1_ratio']}"
            else:
                gsel = f"a={gcombo['alpha']:.2g}"
            note = f"global:{gsel} | nested:{nrange}"
        delta = global_mean - nested_mean
        print(f"{name:<11}{nested_mean:>13.4f}{global_mean:>13.4f}{delta:>+10.4f}{maxd:>11.4f}  {note}")
        rows.append((name, nested_mean, global_mean, delta, maxd))

    print("\nInterpretation:")
    print("  Delta = Global - Nested (negative => global/leaky selection is optimistic, i.e. lower RMSE).")
    print("  maxΔ/subj = largest per-participant RMSE difference between the two protocols.")
    worst = max(rows[1:], key=lambda x: abs(x[3]))
    print(f"  Largest mean-RMSE shift among tuned linear models: {worst[0]} = {worst[3]:+.4f} mL/kg/min.")


if __name__ == "__main__":
    main()
