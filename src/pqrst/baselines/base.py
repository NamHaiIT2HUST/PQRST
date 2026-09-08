"""Interface chung cho moi bo uoc luong transfer entropy (baseline lan sau nay la MINE/quantum).

Day la quyet dinh kien truc quan trong nhat cua Pha P: toan bo Pha Q->T' se goi
estimator qua interface nay, khong quan tam ben trong la KSG, MLP (T_phi, Pha Q),
hay mach luong tu (T_theta, Nhip 2 Pha P'). Nhip 2 chi them 1 class moi ke thua
BaseTEEstimator - khong sua code goi o Pha R/S/T. Xem docs/PHASE_P_GUIDE.md, muc 3.2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseTEEstimator(ABC):
    """Moi estimator (baseline cha) phai implement estimate(x, y) -> TE(X->Y)."""

    @abstractmethod
    def estimate(self, x: np.ndarray, y: np.ndarray, **kwargs) -> float:
        """Uoc luong TE(X -> Y) tu 2 chuoi thoi gian 1D cung do dai.

        Args:
            x: chuoi nguon, mang 1D.
            y: chuoi dich, mang 1D, cung do dai voi x.
            **kwargs: tham so rieng cua tung estimator (vd: k cho KSG, so bin cho binning).

        Returns:
            Gia tri TE(X->Y) uoc luong duoc, don vi nat (hoac bit - quyet dinh 1 don vi
            thong nhat cho toan bo du an va ghi ro trong README).
        """
        raise NotImplementedError
