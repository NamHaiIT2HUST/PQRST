"""Baseline TE dua tren symbolic transfer entropy (Staniek & Lehnertz, 2008).

Y tuong: ma hoa moi cua so m diem lien tiep cua chuoi thanh 1 "symbol" theo thu tu
tuong doi (ordinal pattern) cua cac gia tri, roi tinh TE tren chuoi symbol roi rac
(co the tai dung logic uoc luong TE roi rac tu binning.py cho buoc cuoi).

TODO(ban tu code, xac minh truoc):
    - Kiem tra IDTxl co san estimator "symbolic" hay khong (chua chac chan - can doc
      doc/API hien tai). Neu khong co, list lua chon: cai them thu vien co san
      (vd pyinform hoac tuong duong), hoac tu code ordinal-pattern encoding + TE roi rac.
    - Ghi lai quyet dinh cuoi cung (thu vien nao / tu code) vao
      docs/references/symbolic_te_decision.md - day la 1 muc trong checklist thoat Pha P.
    - Tham so m (do dai pattern) va delay doc tu configs/baselines/symbolic_te.yaml.
"""

from __future__ import annotations

import numpy as np

from pqrst.baselines.base import BaseTEEstimator


class SymbolicTEEstimator(BaseTEEstimator):
    """TE(X->Y) uoc luong bang symbolic transfer entropy (ordinal pattern encoding)."""

    def __init__(self, pattern_length: int = 3, delay: int = 1, **kwargs):
        self.pattern_length = pattern_length
        self.delay = delay
        self.extra_kwargs = kwargs

    def estimate(self, x: np.ndarray, y: np.ndarray, **kwargs) -> float:
        from idtxl.estimators_jidt import JidtDiscreteTE
        
        def encode_ordinal(series: np.ndarray, m: int, delay: int) -> np.ndarray:
            n = len(series)
            patterns = []
            for i in range(n - (m - 1) * delay):
                window = series[i : i + m * delay : delay]
                patterns.append(tuple(np.argsort(window)))
            # Map tuples to consecutive integers
            unique_patterns, indices = np.unique(patterns, axis=0, return_inverse=True)
            return indices
            
        x_sym = encode_ordinal(x, self.pattern_length, self.delay)
        y_sym = encode_ordinal(y, self.pattern_length, self.delay)
        
        import math
        settings = {
            'history_target': 1,
            'discretise_method': 'none', # already discrete
            'alph1': math.factorial(self.pattern_length),
            'alph2': math.factorial(self.pattern_length),
        }
        settings.update(self.extra_kwargs)
        settings.update(kwargs)
        
        estimator = JidtDiscreteTE(settings)
        te_bits = estimator.estimate(x_sym, y_sym)
        return float(te_bits * np.log(2))
