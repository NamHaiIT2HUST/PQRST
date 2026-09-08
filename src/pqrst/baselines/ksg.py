"""Baseline TE dua tren KSG (Kraskov-Stogbauer-Grassberger k-NN estimator), qua IDTxl/JIDT.

TODO(ban tu code, xac minh API truoc):
    - Doc README/doc cua IDTxl hien tai (API co the da doi) - xem docs/PHASE_P_GUIDE.md
      muc 1.3 va docs/references/.
    - IDTxl thuong yeu cau du lieu dang `idtxl.data.Data` va estimator TE qua workflow
      rieng (vd: MultivariateTE / BivariateTE analysis class) - kiem tra ten class/ham
      chinh xac trong phien ban dang cai, dung hard-code theo tri nho khong xac minh.
    - Tham so k (so lang gieng) va cac tham so IDTxl khac doc tu configs/baselines/ksg.yaml.
"""

from __future__ import annotations

import numpy as np

from pqrst.baselines.base import BaseTEEstimator


class KSGTEEstimator(BaseTEEstimator):
    """TE(X->Y) uoc luong bang KSG k-NN estimator (qua IDTxl)."""

    def __init__(self, k: int = 4, **idtxl_kwargs):
        self.k = k
        self.idtxl_kwargs = idtxl_kwargs

    def estimate(self, x: np.ndarray, y: np.ndarray, **kwargs) -> float:
        raise NotImplementedError
