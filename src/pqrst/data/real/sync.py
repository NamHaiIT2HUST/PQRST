from __future__ import annotations
import numpy as np
from pqrst.data.synthetic.corpus import Window

def sync_and_window(source: np.ndarray, target: np.ndarray, grid_fs: float, window_seconds: float = 30.0, config_name: str = "", record_id: str = "", step_seconds: float | None = None) -> list[Window]:
    if step_seconds is None:
        step_seconds = window_seconds
        
    assert len(source) == len(target), "Source and target must have the same length"
    
    n_samples = int(window_seconds * grid_fs)
    step_samples = int(step_seconds * grid_fs)
    
    windows = []
    for i in range(0, len(source) - n_samples + 1, step_samples):
        s_seg = source[i:i+n_samples]
        t_seg = target[i:i+n_samples]
        
        if np.isnan(s_seg).any() or np.isnan(t_seg).any():
            continue
            
        y_t = t_seg[1:]
        x_lag = s_seg[:-1]
        y_lag = t_seg[:-1]
        
        windows.append(Window(
            y_t=y_t,
            x_lag=x_lag,
            y_lag=y_lag,
            n_samples=len(y_t),
            te_ground_truth=float('nan'),
            mi_full_ground_truth=float('nan'),
            mi_reduced_ground_truth=float('nan'),
            config_name=config_name,
            params={'grid_fs': grid_fs, 'window_seconds': window_seconds, 'record_id': record_id, 'config_name': config_name},
            seed=42 + i
        ))
    return windows

def verify_synchronization(source: np.ndarray, target: np.ndarray, grid_fs: float, estimator, shift_seconds: float = 10.0) -> dict:
    wins_align = sync_and_window(source, target, grid_fs, 30.0)
    te_align = []
    for w in wins_align[:50]:
        x = np.zeros(w.n_samples + 1)
        y = np.zeros(w.n_samples + 1)
        x[:-1] = w.x_lag
        y[:-1] = w.y_lag
        y[1:] = w.y_t
        te_align.append(estimator.estimate(x, y))
        
    shift_samples = int(shift_seconds * grid_fs)
    s_shift = source[shift_samples:]
    t_shift = target[:-shift_samples]
    wins_shift = sync_and_window(s_shift, t_shift, grid_fs, 30.0)
    te_shift = []
    for w in wins_shift[:50]:
        x = np.zeros(w.n_samples + 1)
        y = np.zeros(w.n_samples + 1)
        x[:-1] = w.x_lag
        y[:-1] = w.y_lag
        y[1:] = w.y_t
        te_shift.append(estimator.estimate(x, y))
        
    m_align = np.mean(te_align)
    m_shift = np.mean(te_shift)
    ratio = m_shift / m_align if m_align > 0 else float('inf')
    
    return {
        "te_aligned": m_align,
        "te_shifted": m_shift,
        "ratio": ratio,
        "passed": ratio < 0.7
    }
