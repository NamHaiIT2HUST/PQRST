from __future__ import annotations
import numpy as np

def rr_from_beat_annotations(beat_times: np.ndarray, exclude_ectopic: bool = True) -> tuple[np.ndarray, np.ndarray]:
    if len(beat_times) < 2: return np.array([]), np.array([])
    rr = np.diff(beat_times)
    t_rr = beat_times[1:]
    
    if exclude_ectopic:
        # local median filter for 20% deviation
        med = np.median(rr)
        valid = (rr > 0.8 * med) & (rr < 1.2 * med)
        rr = rr[valid]
        t_rr = t_rr[valid]
    return t_rr, rr

def interpolate_rr(t_rr: np.ndarray, rr: np.ndarray, grid_fs: float = 4.0, t_start: float | None = None, t_end: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    if len(t_rr) < 2: return np.array([]), np.array([])
    if t_start is None: t_start = t_rr[0]
    if t_end is None: t_end = t_rr[-1]
    
    t_grid = np.arange(t_start, t_end, 1.0 / grid_fs)
    # Use step interpolation (previous value) as recommended for RR, or linear
    rr_grid = np.interp(t_grid, t_rr, rr)
    return t_grid, rr_grid
