"""Loaders for the two additional datasets: WEEE (Gashi et al. 2022) and Ingraham et al. (2019).

Both return an r1.data.Panel with columns pid, seg, t, y, y_hist, avail, features, regime (protocol-timing stratum)
and condition (activity label). Processed panels are cached as parquet files next to the raw data."""
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .data import Panel

EXT_ROOT = Path(os.environ.get("VO2_EXTERNAL_DATA", Path(__file__).resolve().parents[1] / "data" / "external"))
WEEE_ROOT = EXT_ROOT / "weee_raw" / "extracted" / "dataset"
ING_ROOT = EXT_ROOT / "ingraham_raw"
CACHE = EXT_ROOT / "cache"
GAP_1HZ_S = 5
GAP_BREATH_S = 15    # (breath-by-breath criterion)
MIN_SEG_S = 300
TRANSITION_S = 90
STEADY_S = 180

# Ingraham et al. dataset PDF, Table 4 (self-reported mass, kg)
ING_MASS = {1: 63.49, 2: 63.49, 3: 71.2, 4: 68.03, 5: 68.03, 6: 68.03, 7: 95.24, 8: 65.76, 9: 68.93, 10: 58.05}


def _segments_from_valid(valid, max_gap):
    """Split a boolean 1 Hz validity vector into segments: gaps (runs of invalid samples) longer than max_gap split;
    shorter gaps are bridged. Returns a list of (start, end_exclusive) index pairs over valid stretches."""
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return []
    br = np.flatnonzero(np.diff(idx) > max_gap + 1)
    starts = np.r_[idx[0], idx[br + 1]]
    ends = np.r_[idx[br], idx[-1]] + 1
    return [(s, e) for s, e in zip(starts, ends) if e - s >= MIN_SEG_S]


def _protocol_regime(cond_series):
    """protocol timing on a 1 Hz condition-label array within one segment: transition = 0-90 s after a
    condition change, steady = last 180 s of a condition lasting >= 300 s (within the segment), middle = otherwise."""
    c = np.asarray(cond_series, dtype=object)
    n = len(c)
    lab = np.array(["middle"] * n, dtype=object)
    change = np.r_[False, c[1:] != c[:-1]]
    run_starts = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
    run_ends = np.r_[run_starts[1:], n]
    for s, e in zip(run_starts, run_ends):
        if e - s >= 300:
            lab[e - STEADY_S:e] = "steady"
    for s in np.flatnonzero(change):
        lab[s:min(n, s + TRANSITION_S)] = "transition"
    lab[pd.isna(cond_series)] = None
    return lab


# ---------------------------------------------------------------------------------------------------- WEEE
def _weee_conditions(info_row, t_index, last_valid):
    names = ["Sit", "Stand", "Cycle1", "Cycle2", "Run1", "Run2"]
    starts = [pd.Timestamp(info_row[f"Start_{n}"]) for n in names]
    cond = pd.Series([None] * len(t_index), index=t_index, dtype=object)
    for i, n in enumerate(names):
        end = starts[i + 1] if i + 1 < len(names) else last_valid + pd.Timedelta(seconds=1)
        cond[(t_index >= starts[i]) & (t_index < end)] = n
    return cond


def load_weee(include="protocol", offsets_csv=EXT_ROOT / "weee_clock_offsets.csv", use_cache=True):
    """include='protocol' -> inclusion list (n = 15); include='all' -> all participants with >= 300 s
    valid data (sensitivity). Target = VO2[mL/min] / body mass from Demographics.csv. HR_zephyr is kept for the HR plausibility
    screen only (not a model feature)."""
    cache = CACHE / f"weee_{include}_v2.parquet"
    if use_cache and cache.exists():
        df = pd.read_parquet(cache)
        return Panel("WEEE", df, ["HR", "dHR", "BR", "BRAmplitude", "Activity"])
    inc = pd.read_csv(EXT_ROOT / "weee_inclusion_v04.csv")
    pids = inc.loc[inc["included"], "participant"].tolist() if include == "protocol" else inc["participant"].tolist()
    info = pd.read_csv(WEEE_ROOT / "Study_Information.csv").set_index("Participant")
    off = pd.read_csv(offsets_csv).set_index("participant")["offset_s"].to_dict()
    mass = pd.read_csv(WEEE_ROOT / "Demographics.csv").set_index("Participant")["Weight"].astype(float).to_dict()
    parts, seg_id = [], 0
    for p in pids:
        a = pd.read_csv(WEEE_ROOT / p / "VO2" / "DataAverage.csv", usecols=["Time", "VO2[mL/min]", "HR[bpm]"])
        a["ts"] = pd.to_datetime(a["Time"])
        a = a.drop_duplicates("ts").set_index("ts").sort_index()
        grid = pd.date_range(a.index.min(), a.index.max(), freq="1s")
        a = a.reindex(grid)
        y = (a["VO2[mL/min]"] / mass[p]).where(a["VO2[mL/min]"] > 0)
        hr = a["HR[bpm]"].where(a["HR[bpm]"] > 0)
        zf = sorted((WEEE_ROOT / p / "ZEPHYR").glob("*_Summary.csv"))[0]
        z = pd.read_csv(zf, usecols=["Time", "HR", "BR", "BRAmplitude", "Activity"]).rename(columns={"HR": "HR_zephyr"})
        z["ts"] = pd.to_datetime(z["Time"], format="%d/%m/%Y %H:%M:%S.%f") - pd.Timedelta(seconds=off.get(p, 0.0))
        z = z.drop(columns="Time").sort_values("ts")
        zm = pd.merge_asof(pd.DataFrame({"ts": grid}), z, on="ts", direction="backward",
                           tolerance=pd.Timedelta(seconds=2)).set_index("ts")
        valid = (y.notna() & hr.notna() & zm[["BR", "BRAmplitude", "Activity"]].notna().all(axis=1)).to_numpy()
        cond = _weee_conditions(info.loc[p], grid, y.dropna().index.max())
        for s, e in _segments_from_valid(valid, GAP_1HZ_S):
            sl = slice(s, e)
            d = pd.DataFrame({"pid": int(p[1:]), "seg": seg_id, "t": np.arange(e - s)})
            d["y"] = y.iloc[sl].interpolate(limit=GAP_1HZ_S, limit_area="inside").to_numpy()
            d["y_hist"] = y.iloc[sl].ffill().to_numpy()   # gap-interpolated targets never used as history
            d["HR"] = hr.iloc[sl].ffill().to_numpy()
            for c in ["BR", "BRAmplitude", "Activity"]:
                d[c] = zm[c].iloc[sl].ffill().to_numpy()
            d["HR_zephyr"] = zm["HR_zephyr"].iloc[sl].to_numpy()
            d["dHR"] = np.r_[0.0, np.diff(d["HR"].to_numpy())]
            d["condition"] = cond.iloc[sl].to_numpy()
            d["regime"] = _protocol_regime(d["condition"].to_numpy())
            parts.append(d)
            seg_id += 1
    df = pd.concat(parts, ignore_index=True)
    df["avail"] = np.arange(len(df))  # zero-order-held breath values: available at t
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return Panel("WEEE", df, ["HR", "dHR", "BR", "BRAmplitude", "Activity"])


# ---------------------------------------------------------------------------------------------------- Ingraham
def _ing_blocks(subject):
    with h5py.File(ING_ROOT / f"Subject{subject:02d}.mat", "r") as f:
        s = f[f"Subject{subject:02d}"]
        for blk in s.keys():
            m = s[blk]["Metabolics_System"]["Data"][()]           # 9 x n_breaths
            apdm = s[blk]["APDM_Accel"]["Data"][0:5, :]             # time, code, waist acc x/y/z
            yield blk, m, apdm


def load_ing(use_cache=True):
    cache = CACHE / "ing.parquet"
    feats = ["HR", "dHR", "ACC", "BF"]
    if use_cache and cache.exists():
        return Panel("ING", pd.read_parquet(cache), feats)
    parts, seg_id = [], 0
    for subj in range(1, 11):
        mass = ING_MASS[subj]
        for blk, m, apdm in _ing_blocks(subj):
            tb, code_b = m[0], m[1]
            vo2_b = m[2] * 60.0 / mass                               # mL/s -> mL kg^-1 min^-1
            bf_b, hr_b = m[5], m[8]
            order = np.argsort(tb)
            tb, code_b, vo2_b, bf_b, hr_b = tb[order], code_b[order], vo2_b[order], bf_b[order], hr_b[order]
            t_end = int(np.floor(tb.max()))
            grid = np.arange(int(np.ceil(tb.min())), t_end + 1)
            # target for scoring: linear interpolation between breaths; history: zero-order hold of breaths
            y = np.interp(grid, tb, vo2_b)
            last_breath = np.searchsorted(tb, grid, side="right") - 1
            y_hist = vo2_b[last_breath]
            hr = hr_b[last_breath]
            bf = bf_b[last_breath]
            cond = code_b[last_breath]
            # breath gaps > 15 s invalidate the target between those breaths
            gap_after = np.r_[np.diff(tb), 0.0]
            bad = gap_after[last_breath] > GAP_BREATH_S
            # waist dynamic acceleration: SD of |a| over the trailing window (t-1, t]
            ta = apdm[0]
            mag = np.sqrt((apdm[2:5] ** 2).sum(axis=0))
            ok = np.isfinite(ta) & np.isfinite(mag)
            ta, mag = ta[ok], mag[ok]
            lo = np.searchsorted(ta, grid - 1, side="right")
            hi = np.searchsorted(ta, grid, side="right")
            c1 = np.r_[0, np.cumsum(mag)]
            c2 = np.r_[0, np.cumsum(mag ** 2)]
            cnt = hi - lo
            with np.errstate(invalid="ignore", divide="ignore"):
                mu = (c1[hi] - c1[lo]) / cnt
                acc = np.sqrt(np.maximum((c2[hi] - c2[lo]) / cnt - mu ** 2, 0) * cnt / np.maximum(cnt - 1, 1))
            acc[cnt < 64] = np.nan
            valid = ~bad & np.isfinite(acc) & np.isfinite(hr) & np.isfinite(bf) & np.isfinite(y)
            for s, e in _segments_from_valid(valid, GAP_1HZ_S):
                d = pd.DataFrame({"pid": subj, "seg": seg_id, "t": np.arange(e - s)})
                d["y"] = y[s:e]
                d["y_hist"] = y_hist[s:e]
                hr_s = pd.Series(hr[s:e]).ffill()
                d["HR"] = hr_s.to_numpy()
                d["dHR"] = np.r_[0.0, np.diff(hr_s.rolling(10, min_periods=1).mean().to_numpy())]
                d["ACC"] = pd.Series(acc[s:e]).ffill().to_numpy()
                d["BF"] = pd.Series(bf[s:e]).ffill().to_numpy()
                d["condition"] = [f"{blk}:{int(c)}" for c in cond[s:e]]
                d["regime"] = _protocol_regime(d["condition"].to_numpy())
                d["block"] = blk
                parts.append(d)
                seg_id += 1
    df = pd.concat(parts, ignore_index=True)
    df["avail"] = np.arange(len(df))  # y_hist is already a zero-order hold of breaths (available at breath time)
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return Panel("ING", df, feats)
