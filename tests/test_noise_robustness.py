from __future__ import annotations

import numpy as np
import pytest

from pqrst.utils.noise import add_gaussian_noise


def test_add_gaussian_noise_identity():
    x = np.linspace(-1, 1, 100)
    x_clean = add_gaussian_noise(x, snr_db=None)
    np.testing.assert_array_equal(x, x_clean)

    x_inf = add_gaussian_noise(x, snr_db=float("inf"))
    np.testing.assert_array_equal(x, x_inf)


def test_add_gaussian_noise_reproducibility():
    x = np.sin(np.linspace(0, 10, 500))
    noisy_1 = add_gaussian_noise(x, snr_db=10.0, seed=123)
    noisy_2 = add_gaussian_noise(x, snr_db=10.0, seed=123)
    noisy_diff = add_gaussian_noise(x, snr_db=10.0, seed=456)

    np.testing.assert_allclose(noisy_1, noisy_2)
    assert not np.allclose(noisy_1, noisy_diff)


def test_add_gaussian_noise_snr_level():
    rng = np.random.default_rng(42)
    n = 200_000
    x = rng.normal(0, 2.0, n)  # var ~ 4.0

    target_snrs = [20.0, 10.0, 5.0, 0.0]
    for snr in target_snrs:
        x_noisy = add_gaussian_noise(x, snr_db=snr, seed=42)
        noise = x_noisy - x
        p_signal = np.var(x)
        p_noise = np.var(noise)
        measured_snr = 10.0 * np.log10(p_signal / p_noise)
        # Check within 0.1 dB tolerance
        assert abs(measured_snr - snr) < 0.1, f"SNR expected {snr}, got {measured_snr}"


def test_constant_signal_no_crash():
    x = np.ones(50) * 3.14
    x_out = add_gaussian_noise(x, snr_db=10.0, seed=42)
    np.testing.assert_array_equal(x, x_out)
