"""
Comprehensive Filter Cutoff Optimization for Linear and Tree-Based Models

Tests fine-grained low-pass cutoff frequencies for:
- Ridge (with per-fold standardization)
- OLS (with per-fold standardization)
- Lasso (with per-fold standardization)
- ElasticNet (with per-fold standardization)
- RF Beltrame (9 trees, max_features=2)
- RF 50 trees (max_features=auto)
- RF 200 trees (max_features=auto)

Saves per-model optimal cutoff, full results CSV, and comparison figures.
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def butter_lowpass(data, cutoff_hz, fs=1.0, order=2):
    nyq = 0.5 * fs
    nc = cutoff_hz / nyq
    if nc >= 1.0 or nc <= 0.0:
        return data.copy()
    b, a = butter(order, nc, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)


def load_raw(csv_path):
    df = pd.read_csv(csv_path)
    df = df[~(df == -1).any(axis=1)].reset_index(drop=True)
    pidx = df['ID'].to_numpy(int)
    weight = df['Weight'].to_numpy(float)
    vo2 = df['VO2'].to_numpy(float) / weight
    participants = np.unique(pidx)

    N = len(df)
    X = np.zeros((N, 6), dtype=float)
    X[:, 0] = df['BF'].values
    X[:, 1] = df['VE'].values
    X[:, 2] = df['ACC'].values
    X[:, 3] = df['HR'].values
    X[:, 4] = df['CAD'].values

    for pid in participants:
        idx = (pidx == pid)
        hr_pi = X[idx, 3]
        dhr = np.zeros_like(hr_pi)
        if len(hr_pi) > 1:
            dhr[1:] = np.diff(hr_pi)
        X[idx, 5] = dhr

    return pidx, X, vo2, participants


def apply_lowpass(X, pidx, participants, cutoff_hz):
    if cutoff_hz is None:
        return X.copy()
    X_out = X.copy()
    for pid in participants:
        idx = (pidx == pid)
        pdata = X_out[idx, :]
        mu = pdata.mean(axis=0, keepdims=True)
        pdata_filt = butter_lowpass(pdata - mu, cutoff_hz)
        X_out[idx, :] = pdata_filt + mu
    return X_out


def run_linear_lopo(pidx, X, y, participants, model_class, model_kwargs=None, tune_alpha=True):
    """Linear model with per-fold standardization and optional HP tuning."""
    if model_kwargs is None:
        model_kwargs = {}
    rmses, corrs, r2s = [], [], []

    for test_pid in participants:
        train_mask = (pidx != test_pid)
        test_mask = ~train_mask

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train = scaler_X.fit_transform(X[train_mask])
        y_train = scaler_y.fit_transform(y[train_mask].reshape(-1, 1)).ravel()
        X_test = scaler_X.transform(X[test_mask])

        if tune_alpha and model_class != LinearRegression:
            if model_class == Ridge:
                grid = {'alpha': np.logspace(-2, 5, 15)}
            elif model_class == Lasso:
                grid = {'alpha': np.logspace(-4, 2, 15)}
            elif model_class == ElasticNet:
                grid = {'alpha': np.logspace(-4, 2, 10), 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}

            gcv = GridSearchCV(model_class(**model_kwargs), param_grid=grid,
                               cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
            gcv.fit(X_train, y_train)
            model = gcv.best_estimator_
        else:
            model = model_class(**model_kwargs)
            model.fit(X_train, y_train)

        y_pred = scaler_y.inverse_transform(model.predict(X_test).reshape(-1, 1)).ravel()
        y_true = y[test_mask]

        rmses.append(np.sqrt(mean_squared_error(y_true, y_pred)))
        corrs.append(np.corrcoef(y_true, y_pred)[0, 1])
        r2s.append(r2_score(y_true, y_pred))

    return np.mean(rmses), np.std(rmses), np.mean(corrs), np.std(corrs), np.mean(r2s), np.std(r2s)


def run_rf_lopo(pidx, X, y, participants, n_estimators=50, max_features=None):
    rmses, corrs, r2s = [], [], []
    for test_pid in participants:
        train_mask = (pidx != test_pid)
        test_mask = ~train_mask

        rf = RandomForestRegressor(
            n_estimators=n_estimators, min_samples_leaf=1,
            max_features=max_features, random_state=42, n_jobs=-1
        )
        rf.fit(X[train_mask], y[train_mask])
        y_pred = rf.predict(X[test_mask])
        y_true = y[test_mask]

        rmses.append(np.sqrt(mean_squared_error(y_true, y_pred)))
        corrs.append(np.corrcoef(y_true, y_pred)[0, 1])
        r2s.append(r2_score(y_true, y_pred))

    return np.mean(rmses), np.std(rmses), np.mean(corrs), np.std(corrs), np.mean(r2s), np.std(r2s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='../../data/TB-File-01.csv')
    parser.add_argument('--output', default='../../results/filter_optimization')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Loading data...")
    pidx, X_raw, y, participants = load_raw(args.input)

    # Fine-grained low-pass cutoff sweep
    cutoffs = [None, 0.003, 0.005, 0.007, 0.008, 0.009, 0.01, 0.012, 0.015, 0.02,
               0.025, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]

    models = {
        'OLS':             lambda X: run_linear_lopo(pidx, X, y, participants, LinearRegression, tune_alpha=False),
        'Ridge':           lambda X: run_linear_lopo(pidx, X, y, participants, Ridge),
        'Lasso':           lambda X: run_linear_lopo(pidx, X, y, participants, Lasso, {'max_iter': 5000}),
        'ElasticNet':      lambda X: run_linear_lopo(pidx, X, y, participants, ElasticNet, {'max_iter': 5000}),
        'RF_9t_2f':        lambda X: run_rf_lopo(pidx, X, y, participants, 9, 2),
        'RF_50t':          lambda X: run_rf_lopo(pidx, X, y, participants, 50, None),
        'RF_200t':         lambda X: run_rf_lopo(pidx, X, y, participants, 200, None),
    }

    all_results = []
    total = len(cutoffs) * len(models)
    count = 0

    for cutoff in cutoffs:
        label = 'none' if cutoff is None else f'{cutoff}'
        X_filt = apply_lowpass(X_raw, pidx, participants, cutoff)

        for model_name, model_fn in models.items():
            count += 1
            print(f"[{count}/{total}] cutoff={label}, {model_name}...", end=' ', flush=True)
            mean_rmse, std_rmse, mean_corr, std_corr, mean_r2, std_r2 = model_fn(X_filt)
            print(f"RMSE={mean_rmse:.4f}±{std_rmse:.4f}  r={mean_corr:.4f}")
            all_results.append({
                'cutoff_hz': cutoff if cutoff else 999,
                'cutoff_label': label,
                'model': model_name,
                'mean_rmse': mean_rmse,
                'std_rmse': std_rmse,
                'mean_corr': mean_corr,
                'std_corr': std_corr,
                'mean_r2': mean_r2,
                'std_r2': std_r2,
            })

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(args.output, 'filter_optimization_full.csv'), index=False)

    # Find optimal cutoff per model
    print(f"\n{'='*80}")
    print("OPTIMAL CUTOFF PER MODEL")
    print(f"{'='*80}")
    print(f"{'Model':<15} {'Best Cutoff':>12} {'RMSE':>10} {'±std':>8} {'r':>8} {'R2':>8}")
    print('-' * 65)

    best_per_model = []
    for model_name in models.keys():
        sub = df[df['model'] == model_name]
        best = sub.loc[sub['mean_rmse'].idxmin()]
        c = best['cutoff_label']
        print(f"{model_name:<15} {c:>12} {best['mean_rmse']:>10.4f} {best['std_rmse']:>8.4f} "
              f"{best['mean_corr']:>8.4f} {best['mean_r2']:>8.4f}")
        best_per_model.append({
            'model': model_name,
            'best_cutoff_hz': best['cutoff_hz'] if best['cutoff_hz'] != 999 else None,
            'best_cutoff_label': c,
            'rmse': best['mean_rmse'],
            'std_rmse': best['std_rmse'],
            'corr': best['mean_corr'],
            'r2': best['mean_r2'],
        })

    pd.DataFrame(best_per_model).to_csv(
        os.path.join(args.output, 'optimal_cutoffs.csv'), index=False)

    # Full table sorted by cutoff for each model
    print(f"\n{'='*80}")
    print("FULL RESULTS BY MODEL")
    print(f"{'='*80}")
    for model_name in models.keys():
        sub = df[df['model'] == model_name].sort_values('cutoff_hz')
        print(f"\n  {model_name}:")
        print(f"  {'Cutoff':>10} {'RMSE':>10} {'±':>8} {'r':>8} {'R2':>8}")
        print(f"  {'-'*46}")
        for _, row in sub.iterrows():
            marker = ' <-- BEST' if row['mean_rmse'] == sub['mean_rmse'].min() else ''
            print(f"  {row['cutoff_label']:>10} {row['mean_rmse']:>10.4f} {row['std_rmse']:>8.4f} "
                  f"{row['mean_corr']:>8.4f} {row['mean_r2']:>8.4f}{marker}")

    # Plot: RMSE vs cutoff for each model
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    colors = {
        'OLS': '#1f77b4', 'Ridge': '#ff7f0e', 'Lasso': '#2ca02c', 'ElasticNet': '#d62728',
        'RF_9t_2f': '#9467bd', 'RF_50t': '#8c564b', 'RF_200t': '#e377c2',
    }

    for model_name in models.keys():
        sub = df[(df['model'] == model_name) & (df['cutoff_hz'] != 999)].sort_values('cutoff_hz')
        c = colors[model_name]
        for ax, metric in zip(axes, ['mean_rmse', 'mean_corr', 'mean_r2']):
            ax.plot(sub['cutoff_hz'], sub[metric], 'o-', color=c, label=model_name, markersize=4, linewidth=1.5)

        # Add no-filter baseline as horizontal line
        nf = df[(df['model'] == model_name) & (df['cutoff_hz'] == 999)]
        if len(nf) > 0:
            for ax, metric in zip(axes, ['mean_rmse', 'mean_corr', 'mean_r2']):
                ax.axhline(nf.iloc[0][metric], color=c, linestyle=':', alpha=0.4)

    for ax, title in zip(axes, ['Mean RMSE (lower=better)', 'Mean Pearson r (higher=better)', 'Mean R2 (higher=better)']):
        ax.set_xlabel('Low-pass Cutoff Frequency (Hz)')
        ax.set_ylabel(title.split('(')[0].strip())
        ax.set_title(title)
        ax.set_xscale('log')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Low-Pass Filter Cutoff Optimization (LOPO CV)', fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'filter_optimization.png'), dpi=200, bbox_inches='tight')
    print(f"\nFigure saved to {os.path.join(args.output, 'filter_optimization.png')}")

    # Zoomed plot around optimal region (0.005 - 0.05 Hz)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for model_name in models.keys():
        sub = df[(df['model'] == model_name) & (df['cutoff_hz'] != 999) &
                 (df['cutoff_hz'] >= 0.005) & (df['cutoff_hz'] <= 0.05)].sort_values('cutoff_hz')
        ax2.errorbar(sub['cutoff_hz'], sub['mean_rmse'], yerr=sub['std_rmse'],
                     fmt='o-', color=colors[model_name], label=model_name,
                     markersize=5, linewidth=1.5, capsize=3, alpha=0.8)

    ax2.set_xlabel('Low-pass Cutoff Frequency (Hz)', fontsize=12)
    ax2.set_ylabel('Mean RMSE (mL/kg/min)', fontsize=12)
    ax2.set_title('RMSE vs Low-Pass Cutoff (Zoomed, 0.005-0.05 Hz)', fontweight='bold')
    ax2.set_xscale('log')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'filter_optimization_zoomed.png'), dpi=200, bbox_inches='tight')
    print(f"Zoomed figure saved to {os.path.join(args.output, 'filter_optimization_zoomed.png')}")


if __name__ == '__main__':
    main()
