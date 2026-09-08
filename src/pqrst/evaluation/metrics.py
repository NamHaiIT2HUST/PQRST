"""Metric so sanh estimator TE voi ground-truth. Dung tu Pha P (validate baseline)
den Pha R (so sanh T_phi vs baseline) va Nhip 2 (so sanh T_theta).
"""

from __future__ import annotations

import numpy as np


def bias(estimates: np.ndarray, ground_truth: float) -> float:
    """Bias trung binh = mean(estimates) - ground_truth.

    TODO(ban tu code): 1 dong, dung np.mean.
    """
    raise NotImplementedError


def mse(estimates: np.ndarray, ground_truth: float) -> float:
    """Mean squared error giua estimates va ground_truth.

    TODO(ban tu code): 1 dong, dung np.mean((estimates - ground_truth) ** 2).
    """
    raise NotImplementedError
