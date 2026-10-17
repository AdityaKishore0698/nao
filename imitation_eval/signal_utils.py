"""
signal_utils.py — resampling, gap filling and filtering for angle time series.
"""

import numpy as np
from scipy.signal import butter, filtfilt


def resample_uniform(t, x, fs, t_start=None, t_end=None):
    """
    Linearly resample samples x (T,) or (T, D) taken at times t onto a uniform
    grid at fs Hz. NaN samples are dropped before interpolation per column.
    Returns (t_uniform, x_uniform).
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    t0 = t[0] if t_start is None else t_start
    t1 = t[-1] if t_end is None else t_end
    tu = np.arange(t0, t1 + 1e-9, 1.0 / fs)
    cols = x.reshape(len(t), -1)
    out = np.full((len(tu), cols.shape[1]), np.nan)
    for c in range(cols.shape[1]):
        ok = np.isfinite(cols[:, c])
        if ok.sum() >= 2:
            out[:, c] = np.interp(tu, t[ok], cols[ok, c], left=np.nan, right=np.nan)
    return tu, out.reshape((len(tu),) + x.shape[1:])


def fill_nans(x):
    """
    Linear interpolation across NaNs (nearest value at the edges).
    Returns (filled, valid_fraction). All-NaN input is returned unchanged.
    """
    x = np.asarray(x, dtype=float).copy()
    ok = np.isfinite(x)
    frac = float(ok.mean()) if len(x) else 0.0
    if ok.sum() == 0:
        return x, 0.0
    if ok.sum() < len(x):
        idx = np.arange(len(x))
        x[~ok] = np.interp(idx[~ok], idx[ok], x[ok])
    return x, frac


def lowpass(x, fs, cutoff=6.0, order=4):
    """Zero-lag Butterworth low-pass (standard 6 Hz for upper-limb kinematics)."""
    x = np.asarray(x, dtype=float)
    if cutoff >= fs / 2.0:
        return x
    b, a = butter(order // 2, cutoff / (fs / 2.0))   # filtfilt doubles the order
    padlen = 3 * max(len(a), len(b))
    if len(x) <= padlen:
        return x
    return filtfilt(b, a, x)
