"""Causal per-segment filtering and fold-local standardisation."""
import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi

CUTOFF_GRID = [0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, None]  # Hz; None = unfiltered


def causal_filter(X, seg, cutoff, fs=1.0, order=2):
    """2nd-order Butterworth low-pass, lfilter per feature and per segment, initial state zi * first sample.
    X: (n, p) raw features in panel order; seg: (n,) segment ids. Returns a new array."""
    X = np.asarray(X, dtype=float)
    if cutoff is None:
        return X.copy()
    b, a = butter(order, cutoff / (0.5 * fs), btype="low")
    zi = lfilter_zi(b, a)
    out = np.empty_like(X)
    seg = np.asarray(seg)
    bounds = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1], True])
    for s, e in zip(bounds[:-1], bounds[1:]):
        block = X[s:e]
        for j in range(X.shape[1]):
            out[s:e, j], _ = lfilter(b, a, block[:, j], zi=zi * block[0, j])
    return out


class FoldScaler:
    """Mean/SD standardisation fitted on training rows only."""

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mu) / self.sd

    def inverse(self, Z):
        return np.asarray(Z, dtype=float) * self.sd + self.mu
