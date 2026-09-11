import pytest
import numpy as np
from pqrst.data.real.cardiac import rr_from_beat_annotations, interpolate_rr
from pqrst.data.real.eeg import compute_band_power, detect_cardiac_artifact
from pqrst.data.real.sync import sync_and_window, verify_synchronization
from pqrst.evaluation.sanity_check import bidirectional_te, permutation_test_te, bootstrap_ci, run_sanity_check

class DummyEstimator:
    def estimate(self, x, y, **kwargs):
        # returns sum of difference, just a deterministic fake TE
        return float(np.mean(x - y))

def test_rr_from_beat_annotations_basic():
    t = np.array([1.0, 1.8, 2.6, 3.4])
    t_rr, rr = rr_from_beat_annotations(t, exclude_ectopic=False)
    assert np.allclose(rr, 0.8)
    assert len(t_rr) == 3

def test_rr_from_beat_annotations_ectopic():
    # 0.8, 0.8, 0.2 (ectopic), 1.4 (compensatory), 0.8
    t = np.cumsum([0.0, 0.8, 0.8, 0.2, 1.4, 0.8])
    t_rr, rr = rr_from_beat_annotations(t, exclude_ectopic=True)
    # median is 0.8. Valid range: 0.64 to 0.96.
    assert len(rr) == 3
    assert np.allclose(rr, 0.8)

def test_interpolate_rr():
    t_rr = np.array([1.0, 2.0, 3.0])
    rr = np.array([0.8, 0.9, 1.0])
    t_grid, rr_grid = interpolate_rr(t_rr, rr, grid_fs=2.0)
    assert len(t_grid) > 0
    assert len(t_grid) == len(rr_grid)

def test_compute_band_power():
    fs = 100.0
    t = np.arange(0, 5, 1/fs)
    # 10 Hz signal
    eeg = np.sin(2 * np.pi * 10 * t)
    t_bp, bp = compute_band_power(eeg, fs, band=(8, 12), window_seconds=1.0, step_seconds=0.5)
    assert len(bp) > 0
    assert np.all(bp > np.log(1e-10))

def test_detect_cardiac_artifact_clean():
    eeg = np.random.randn(1000)
    beats = np.array([1.0, 2.0, 3.0, 4.0])
    res = detect_cardiac_artifact(eeg, beats, fs=100.0, window_seconds=0.5)
    assert 'has_artifact' in res

def test_detect_cardiac_artifact_noisy():
    eeg = np.random.randn(1000) * 0.1
    # add sharp spikes at beat times
    beats = np.array([1.0, 2.0, 3.0, 4.0])
    for b in beats:
        eeg[int(b*100)] += 10.0
    res = detect_cardiac_artifact(eeg, beats, fs=100.0, window_seconds=0.5)
    assert 'has_artifact' in res

def test_sync_and_window():
    src = np.arange(100)
    tgt = np.arange(100) * 2
    windows = sync_and_window(src, tgt, grid_fs=1.0, window_seconds=10.0)
    assert len(windows) == 10
    w = windows[0]
    assert w.n_samples == 9
    assert len(w.y_t) == 9
    assert len(w.x_lag) == 9

def test_verify_synchronization():
    src = np.random.randn(200)
    tgt = np.random.randn(200)
    est = DummyEstimator()
    res = verify_synchronization(src, tgt, grid_fs=1.0, estimator=est, shift_seconds=2.0)
    assert 'passed' in res

def test_bidirectional_te():
    src = np.arange(50)
    tgt = np.arange(50)
    w_f = sync_and_window(src, tgt, 1.0, 10.0)[:2]
    w_b = sync_and_window(tgt, src, 1.0, 10.0)[:2]
    est = DummyEstimator()
    res = bidirectional_te(w_f, w_b, est)
    assert res['n_windows'] == 2
    assert 'difference' in res

def test_permutation_test_te():
    src = np.arange(50)
    tgt = np.arange(50)
    w = sync_and_window(src, tgt, 1.0, 10.0)[:5]
    est = DummyEstimator()
    res = permutation_test_te(w, est, n_permutations=10)
    assert 'p_value' in res
    assert 0.0 <= res['p_value'] <= 1.0

def test_bootstrap_ci():
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    low, high = bootstrap_ci(vals, n_bootstrap=50)
    assert low <= high

def test_run_sanity_check():
    src = np.arange(50)
    tgt = np.arange(50)
    w_f = sync_and_window(src, tgt, 1.0, 10.0)[:5]
    w_b = sync_and_window(tgt, src, 1.0, 10.0)[:5]
    est = DummyEstimator()
    res = run_sanity_check(w_f, w_b, est, n_bootstrap=10)
    assert 'passed' in res
