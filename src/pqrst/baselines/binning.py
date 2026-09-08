"""Baseline TE dua tren roi rac hoa histogram/binning (uoc luong TE tu bang xac suat roi rac).

TODO(ban tu code):
    - Co the dung estimator discrete/binning co san trong IDTxl, hoac tu code:
      roi rac hoa x, y (vd: equal-width hoac equal-frequency binning), uoc luong xac suat
      dong tu histogram, tinh TE = sum p(y_t, y_t-1, x_t-1) * log( p(y_t | y_t-1, x_t-1) /
      p(y_t | y_t-1) ).
    - So bin la sieu tham so quan trong - can 1 quy tac chon so bin hop ly theo N (vd:
      Sturges' rule, hoac Freedman-Diaconis) thay vi co dinh, hoac doc tu config.
    - Tham so doc tu configs/baselines/binning.yaml.
"""

from __future__ import annotations

import numpy as np

from pqrst.baselines.base import BaseTEEstimator


class BinningTEEstimator(BaseTEEstimator):
    """TE(X->Y) uoc luong bang roi rac hoa histogram (binning)."""

    def __init__(self, n_bins: int = 8, **kwargs):
        self.n_bins = n_bins
        self.extra_kwargs = kwargs

    def estimate(self, x: np.ndarray, y: np.ndarray, **kwargs) -> float:
        raise NotImplementedError
