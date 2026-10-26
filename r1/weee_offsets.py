"""WEEE clock-offset estimation between the VO2 Master clock and the Zephyr clock.

Sign convention: positive offset = Zephyr clock ahead of the V̇O₂ clock; corrected Zephyr time = Zephyr time - offset.
Primary: R-R tachograms (Wahoo RR on the V̇O₂ clock vs Zephyr R-R from R-peaks detected in the timestamped ECG),
resampled to 2 Hz, high-passed by removing a 60 s moving mean, cross-correlated over ±90 s in non-overlapping 300 s
windows; accepted if >= 3 windows have peak r >= 0.4 and agree within ±2 s (median lag). Fallback: first differences
of 10 s-smoothed HR (Wahoo vs Zephyr), ±90 s, accepted if r >= 0.3. Otherwise 0 and flagged. Uses no target data."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

FS_GRID = 2.0
MAX_LAG_S = 90


def _workout_start(vo2_dir):
    a = pd.read_csv(Path(vo2_dir) / "DataAverage.csv", usecols=["Time[s]", "Time"])
    t0 = pd.to_datetime(a["Time"].iloc[0]) - pd.to_timedelta(a["Time[s]"].iloc[0], unit="s")
    return t0


def vo2_parts(p_dir):
    """VO2 Master recording folders: VO2/Part1, VO2/Part2 for restarted sessions, else VO2."""
    vo2 = Path(p_dir) / "VO2"
    parts = sorted(d for d in vo2.glob("Part*") if d.is_dir())
    return parts or [vo2]


def wahoo_rr(p_dir):
    out = []
    for vo2 in vo2_parts(p_dir):
        h = pd.read_csv(vo2 / "HeartRateMonitor-Data.csv")
        h = h.dropna(subset=["RR[ms]"]).drop_duplicates(subset=["Time[s]", "RR[ms]"])
        t = _workout_start(vo2) + pd.to_timedelta(h["Time[s]"].to_numpy(), unit="s")
        out.append(pd.Series(h["RR[ms]"].to_numpy(float), index=t))
    return pd.concat(out).sort_index()


def wahoo_hr(p_dir):
    vo2 = Path(p_dir) / "VO2"
    a = pd.read_csv(vo2 / "DataAverage.csv", usecols=["Time", "HR[bpm]"])
    return pd.Series(a["HR[bpm]"].to_numpy(float), index=pd.to_datetime(a["Time"])).replace(0, np.nan)


def zephyr_rr_from_ecg(p_dir):
    f = sorted((Path(p_dir) / "ZEPHYR").glob("*_ECG.csv"))[0]
    e = pd.read_csv(f)
    t = pd.to_datetime(e["Time"], format="%d/%m/%Y %H:%M:%S.%f")
    x = e["EcgWaveform"].to_numpy(float)
    fs = 250.0
    b, a = butter(2, [5 / (fs / 2), 20 / (fs / 2)], btype="band")
    y = filtfilt(b, a, x) ** 2  # offline peak detection on the ECG; does not enter any model
    thr = pd.Series(y).rolling(int(2 * fs), min_periods=1, center=True).quantile(0.98).to_numpy() * 0.3
    pk, _ = find_peaks(y, distance=int(0.3 * fs))
    pk = pk[y[pk] > thr[pk]]
    tp = t.iloc[pk].to_numpy()
    rr = np.diff(tp).astype("timedelta64[us]").astype(float) / 1000.0
    ok = (rr > 300) & (rr < 2000)
    return pd.Series(rr[ok], index=pd.DatetimeIndex(tp[1:][ok]))


def zephyr_hr(p_dir):
    f = sorted((Path(p_dir) / "ZEPHYR").glob("*_Summary.csv"))[0]
    s = pd.read_csv(f, usecols=["Time", "HR"])
    return pd.Series(s["HR"].to_numpy(float), index=pd.to_datetime(s["Time"], format="%d/%m/%Y %H:%M:%S.%f")).replace(0, np.nan)


def _grid(series, t0, t1, fs=FS_GRID):
    g = pd.date_range(t0, t1, freq=pd.Timedelta(seconds=1 / fs))
    s = series[~series.index.duplicated()].sort_index()
    v = np.interp(g.asi8, s.index.asi8, s.to_numpy(), left=np.nan, right=np.nan)
    return g, v


def _highpass(v, fs=FS_GRID, win_s=60):
    return v - pd.Series(v).rolling(int(win_s * fs), min_periods=1, center=True).mean().to_numpy()


def _xcorr(a, b, max_lag):
    """r(lag) where lag > 0 means b lags a (b's clock ahead)."""
    out = {}
    for L in range(-max_lag, max_lag + 1):
        if L >= 0:
            x, y = a[:len(a) - L] if L else a, b[L:]
        else:
            x, y = a[-L:], b[:len(b) + L]
        m = np.isfinite(x) & np.isfinite(y)
        out[L] = np.corrcoef(x[m], y[m])[0, 1] if m.sum() > 100 and x[m].std() > 0 and y[m].std() > 0 else np.nan
    return pd.Series(out)


def estimate(p_dir):
    rr_w, rr_z = wahoo_rr(p_dir), zephyr_rr_from_ecg(p_dir)
    t0, t1 = max(rr_w.index.min(), rr_z.index.min() - pd.Timedelta(seconds=MAX_LAG_S)), rr_w.index.max()
    g, vw = _grid(rr_w, t0, t1)
    _, vz = _grid(rr_z, t0, t1)
    vw, vz = _highpass(vw), _highpass(vz)
    win = int(300 * FS_GRID)
    lags, rs = [], []
    for s in range(0, len(g) - win, win):
        c = _xcorr(vw[s:s + win], vz[s:s + win], int(MAX_LAG_S * FS_GRID))
        if c.notna().any():
            L = int(c.idxmax())
            lags.append(L / FS_GRID)
            rs.append(float(c.max()))
    good = [l for l, r in zip(lags, rs) if r >= 0.4]
    if len(good) >= 3 and (max(good) - min(good)) <= 4 and abs(np.max(np.abs(np.array(good) - np.median(good)))) <= 2:
        return {"method": "rr_tachogram", "offset_s": float(np.median(good)), "n_windows_ok": len(good),
                "peak_r": float(np.median([r for r in rs if r >= 0.4])), "flag": False}
    # Fallback: first differences of 10 s-smoothed HR at 1 Hz.
    hw, hz = wahoo_hr(p_dir), zephyr_hr(p_dir)
    t0, t1 = hw.index.min(), hw.index.max()
    g, a = _grid(hw, t0, t1, fs=1.0)
    _, b = _grid(hz, t0, t1, fs=1.0)
    a = np.diff(pd.Series(a).rolling(10, min_periods=5).mean().to_numpy())
    b = np.diff(pd.Series(b).rolling(10, min_periods=5).mean().to_numpy())
    c = _xcorr(a, b, MAX_LAG_S)
    if c.notna().any() and c.max() >= 0.3:
        return {"method": "hr_diff", "offset_s": float(c.idxmax()), "n_windows_ok": len(good), "peak_r": float(c.max()), "flag": False}
    return {"method": "none", "offset_s": 0.0, "n_windows_ok": len(good), "peak_r": float(c.max()) if c.notna().any() else np.nan,
            "flag": True}
