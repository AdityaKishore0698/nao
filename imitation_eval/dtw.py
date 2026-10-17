"""
dtw.py — z-normalisation + classic dynamic time warping
=======================================================
Follows Fears et al. (2023), section 2.4: each joint-angle trajectory is
z-normalised (mean removed, divided by SD) to compensate for range-of-motion
differences between robot and participant, then the participant's curve is
aligned to the robot's with DTW. The DTW cost (sum of |a_i - b_j| along the
optimal warping path) is the outcome: lower = more accurate imitation.

Exact DTW (not the FastDTW approximation) so costs are comparable to the paper.
"""

import numpy as np


def znorm(x, eps=1e-8):
    x = np.asarray(x, dtype=float)
    sd = x.std()
    if sd < eps:
        return np.zeros_like(x)
    return (x - x.mean()) / sd


def dtw(a, b, window=None):
    """
    Exact DTW between 1-D sequences a (len n) and b (len m), |a_i - b_j| local cost.

    window: optional Sakoe-Chiba band half-width in samples (None = unconstrained).
            It is widened to |n - m| if needed so a path always exists.

    Returns (cost, path_length).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        raise ValueError("DTW needs two non-empty sequences")
    w = max(n, m) if window is None else max(int(window), abs(n - m))

    inf = float("inf")
    # cost and path length matrices, (n+1) x (m+1); plain Python floats for speed
    D = [[inf] * (m + 1) for _ in range(n + 1)]
    L = [[0] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    bl = b.tolist()
    for i in range(1, n + 1):
        ai = float(a[i - 1])
        Di, Dp = D[i], D[i - 1]
        Li, Lp = L[i], L[i - 1]
        j_lo = max(1, i - w)
        j_hi = min(m, i + w)
        for j in range(j_lo, j_hi + 1):
            diag, up, left = Dp[j - 1], Dp[j], Di[j - 1]
            if diag <= up and diag <= left:
                best, lb = diag, Lp[j - 1]
            elif up <= left:
                best, lb = up, Lp[j]
            else:
                best, lb = left, Li[j - 1]
            Di[j] = abs(ai - bl[j - 1]) + best
            Li[j] = lb + 1
    return D[n][m], L[n][m]


def dtw_znorm(robot_series, child_series, window=None):
    """z-normalise both series, then DTW. Returns dict with raw and per-step cost."""
    cost, path_len = dtw(znorm(robot_series), znorm(child_series), window=window)
    return {"dtw_cost": cost, "dtw_cost_per_step": cost / path_len, "path_length": path_len}
