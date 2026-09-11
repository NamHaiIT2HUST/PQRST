from __future__ import annotations
import numpy as np

EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

def compute_band_power(eeg: np.ndarray, fs: float, band: tuple[float, float], window_seconds: float = 2.0, step_seconds: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    from scipy.signal import welch
    n_samples = len(eeg)
    window_samples = int(window_seconds * fs)
    step_samples = int(step_seconds * fs)
    
    times = []
    powers = []
    
    for i in range(0, n_samples - window_samples + 1, step_samples):
        segment = eeg[i:i+window_samples]
        f, Pxx = welch(segment, fs, nperseg=window_samples)
        idx = np.logical_and(f >= band[0], f <= band[1])
        power = np.trapz(Pxx[idx], f[idx])
        powers.append(np.log(power + 1e-12))  # log power
        times.append((i + window_samples/2) / fs)
        
    return np.array(times), np.array(powers)

def detect_cardiac_artifact(eeg: np.ndarray, beat_times: np.ndarray, fs: float, window_seconds: float = 1.0) -> dict:
    half_win = int((window_seconds / 2) * fs)
    segments = []
    for t in beat_times:
        i = int(t * fs)
        if i - half_win >= 0 and i + half_win < len(eeg):
            segments.append(eeg[i - half_win : i + half_win])
            
    if not segments:
        return {"has_artifact": False, "qrs_amplitude": 0.0, "snr": 0.0}
        
    avg = np.mean(segments, axis=0)
    qrs_amp = np.max(avg) - np.min(avg)
    bg_amp = np.std(avg)
    snr = qrs_amp / (bg_amp + 1e-12)
    
    return {
        "has_artifact": snr > 3.0,
        "qrs_amplitude": float(qrs_amp),
        "snr": float(snr)
    }
