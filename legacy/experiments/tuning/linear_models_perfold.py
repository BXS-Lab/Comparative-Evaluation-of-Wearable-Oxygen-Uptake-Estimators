"""
Linear Models with Per-Fold Standardization (Fixed)

Fixes the data leakage in the original comprehensive_hyperparameter_search.py:
- Standardization is now fit on training data only within each LOPO fold
- Hyperparameter tuning (GridSearchCV) is done inside each LOPO fold
- No test participant data influences the scaler or hyperparameter selection

Produces results in the same format as the original for direct comparison.
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def butter_lowpass_filter(data, cutoff=0.01, fs=1.0, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)


def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)
    pidx = df['ID'].values
    weight = df['Weight'].values
    vo2_raw = df['VO2'].values
    y = vo2_raw / weight
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
        dhr_pi = np.zeros(len(hr_pi))
        dhr_pi[1:] = hr_pi[1:] - hr_pi[:-1]
        X[idx, 5] = dhr_pi

    X_filtered = butter_lowpass_filter(X, cutoff=0.01, fs=1.0, order=2)
    return pidx, X_filtered, y, participants


def run_lopo_perfold(pidx, X_filtered, y, participants, results_dir):
    """Run LOPO with per-fold standardization and per-fold hyperparameter tuning."""

    feature_cols = ['BF', 'VE', 'ACC', 'HR', 'CAD', 'dHR']

    # Hyperparameter grids (same search spaces as original)
    hp_grids = {
        'Ridge': {'alpha': np.logspace(-3, 6, 30)},
        'Lasso': {'alpha': np.logspace(-4, 3, 30)},
        'ElasticNet': {
            'alpha': np.logspace(-4, 3, 20),
            'l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        },
    }

    model_factories = {
        'Linear': lambda: LinearRegression(),
        'Ridge': lambda: Ridge(random_state=42),
        'Lasso': lambda: Lasso(max_iter=10000, tol=1e-6, random_state=42),
        'ElasticNet': lambda: ElasticNet(max_iter=10000, tol=1e-6, random_state=42),
    }

    all_model_results = {}

    for model_name in ['Linear', 'Ridge', 'Lasso', 'ElasticNet']:
        print(f"\n{'='*80}")
        print(f"  {model_name} — per-fold standardization + per-fold HP tuning")
        print(f"{'='*80}")

        per_participant = []
        all_y_true = []
        all_y_pred = []
        all_pids = []
        fold_alphas = []

        for fold_idx, test_pid in enumerate(participants):
            train_mask = (pidx != test_pid)
            test_mask = ~train_mask

            X_train_raw = X_filtered[train_mask]
            y_train_raw = y[train_mask]
            X_test_raw = X_filtered[test_mask]
            y_test_raw = y[test_mask]

            # Per-fold standardization: fit on TRAINING data only
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_train = scaler_X.fit_transform(X_train_raw)
            y_train = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).ravel()
            X_test = scaler_X.transform(X_test_raw)

            # Per-fold hyperparameter tuning (inside the fold)
            if model_name in hp_grids:
                grid = GridSearchCV(
                    model_factories[model_name](),
                    param_grid=hp_grids[model_name],
                    cv=5,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                )
                grid.fit(X_train, y_train)
                model = grid.best_estimator_
                if model_name == 'Ridge':
                    fold_alphas.append(grid.best_params_['alpha'])
                elif model_name == 'Lasso':
                    fold_alphas.append(grid.best_params_['alpha'])
                elif model_name == 'ElasticNet':
                    fold_alphas.append((grid.best_params_['alpha'], grid.best_params_['l1_ratio']))
            else:
                model = model_factories[model_name]()
                model.fit(X_train, y_train)

            # Predict and de-standardize
            y_pred_scaled = model.predict(X_test)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

            # Metrics in original VO2 units
            rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred))
            corr = np.corrcoef(y_test_raw, y_pred)[0, 1]
            r2 = r2_score(y_test_raw, y_pred)

            per_participant.append({
                'participant': test_pid,
                'rmse': rmse,
                'correlation': corr,
                'r2_score': r2
            })

            all_y_true.extend(y_test_raw)
            all_y_pred.extend(y_pred)
            all_pids.extend([test_pid] * len(y_test_raw))

            print(f"  Fold {fold_idx+1:2d}/16 (pid={test_pid:2d})  "
                  f"RMSE={rmse:.3f}  r={corr:.3f}  R2={r2:.3f}")

        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)

        overall_rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
        overall_corr = np.corrcoef(all_y_true, all_y_pred)[0, 1]
        overall_r2 = r2_score(all_y_true, all_y_pred)

        rmses = [m['rmse'] for m in per_participant]
        corrs = [m['correlation'] for m in per_participant]
        r2s = [m['r2_score'] for m in per_participant]

        print(f"\n  {model_name} Summary:")
        print(f"    Mean RMSE: {np.mean(rmses):.4f} +/- {np.std(rmses):.4f}")
        print(f"    Mean Corr: {np.mean(corrs):.4f} +/- {np.std(corrs):.4f}")
        print(f"    Mean R2:   {np.mean(r2s):.4f} +/- {np.std(r2s):.4f}")
        print(f"    Overall RMSE: {overall_rmse:.4f}  Corr: {overall_corr:.4f}  R2: {overall_r2:.4f}")

        if fold_alphas:
            if model_name == 'ElasticNet':
                alphas = [a for a, _ in fold_alphas]
                l1s = [l for _, l in fold_alphas]
                print(f"    Alpha across folds: median={np.median(alphas):.4f}, "
                      f"range=[{np.min(alphas):.4f}, {np.max(alphas):.4f}]")
                print(f"    L1_ratio across folds: median={np.median(l1s):.2f}, "
                      f"range=[{np.min(l1s):.2f}, {np.max(l1s):.2f}]")
            else:
                print(f"    Alpha across folds: median={np.median(fold_alphas):.4f}, "
                      f"range=[{np.min(fold_alphas):.4f}, {np.max(fold_alphas):.4f}]")

        # Save per-model results
        model_dir = os.path.join(results_dir, model_name.upper())
        os.makedirs(model_dir, exist_ok=True)

        metrics_df = pd.DataFrame(per_participant)
        metrics_df.to_csv(os.path.join(model_dir, 'metrics_per_participant.csv'), index=False)

        pred_df = pd.DataFrame({
            'participant': all_pids,
            'y_true': all_y_true,
            'y_pred': all_y_pred
        })
        pred_df.to_csv(os.path.join(model_dir, 'y_yhat.csv'), index=False)

        all_model_results[model_name] = {
            'overall_rmse': overall_rmse,
            'overall_corr': overall_corr,
            'overall_r2': overall_r2,
            'mean_rmse': np.mean(rmses),
            'std_rmse': np.std(rmses),
            'mean_corr': np.mean(corrs),
            'std_corr': np.std(corrs),
            'mean_r2': np.mean(r2s),
            'std_r2': np.std(r2s),
            'fold_alphas': fold_alphas,
        }

    return all_model_results


def load_old_results(results_dir):
    """Load original (leaky) results for comparison."""
    old = {}
    for model_name in ['LINEAR', 'RIDGE', 'LASSO', 'ELASTICNET']:
        csv_path = os.path.join(results_dir, model_name, 'metrics_per_participant.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            old[model_name] = {
                'mean_rmse': df['rmse'].mean(),
                'mean_corr': df['correlation'].mean(),
                'mean_r2': df['r2_score'].mean(),
            }
    return old


def print_comparison(new_results, old_results):
    """Print side-by-side comparison of old vs new results."""
    print(f"\n{'='*100}")
    print("COMPARISON: Old (global scaler) vs New (per-fold scaler)")
    print(f"{'='*100}")
    print(f"\n{'Model':<12} {'Old RMSE':>10} {'New RMSE':>10} {'Delta':>10} "
          f"{'Old r':>10} {'New r':>10} {'Delta':>10} "
          f"{'Old R2':>10} {'New R2':>10} {'Delta':>10}")
    print("-" * 102)

    for model_name in ['Linear', 'Ridge', 'Lasso', 'ElasticNet']:
        key = model_name.upper()
        if key not in old_results:
            continue
        old = old_results[key]
        new = new_results[model_name]

        d_rmse = new['mean_rmse'] - old['mean_rmse']
        d_corr = new['mean_corr'] - old['mean_corr']
        d_r2 = new['mean_r2'] - old['mean_r2']

        print(f"{model_name:<12} "
              f"{old['mean_rmse']:>10.4f} {new['mean_rmse']:>10.4f} {d_rmse:>+10.4f} "
              f"{old['mean_corr']:>10.4f} {new['mean_corr']:>10.4f} {d_corr:>+10.4f} "
              f"{old['mean_r2']:>10.4f} {new['mean_r2']:>10.4f} {d_r2:>+10.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear models with per-fold standardization')
    parser.add_argument('--input', type=str, default='TB-File-01.csv', help='Input CSV')
    parser.add_argument('--results', type=str, default='results_linear_fixed', help='Output directory')
    parser.add_argument('--old-results', type=str, default=None,
                        help='Path to old results for comparison (e.g. results/linear/)')
    args = parser.parse_args()

    pidx, X_filtered, y, participants = load_and_prepare(args.input)

    os.makedirs(args.results, exist_ok=True)
    new_results = run_lopo_perfold(pidx, X_filtered, y, participants, args.results)

    if args.old_results and os.path.isdir(args.old_results):
        old_results = load_old_results(args.old_results)
        if old_results:
            print_comparison(new_results, old_results)

    # Save summary
    summary_rows = []
    for model_name in ['Linear', 'Ridge', 'Lasso', 'ElasticNet']:
        r = new_results[model_name]
        summary_rows.append({
            'model': model_name,
            'mean_rmse': r['mean_rmse'],
            'std_rmse': r['std_rmse'],
            'mean_corr': r['mean_corr'],
            'std_corr': r['std_corr'],
            'mean_r2': r['mean_r2'],
            'std_r2': r['std_r2'],
            'overall_rmse': r['overall_rmse'],
            'overall_corr': r['overall_corr'],
            'overall_r2': r['overall_r2'],
        })
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(args.results, 'model_comparison_summary.csv'), index=False)

    print(f"\nResults saved to {args.results}/")
