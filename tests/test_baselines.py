"""Smoke test cho 3 baseline (KSG, symbolic TE, binning). Xem checklist Pha P.

Muc tieu toi thieu: moi estimator chay khong loi tren du lieu tong hop don gian,
tra ve so thuc huu han (khong NaN/Inf), va TE(coupled) > TE(uncoupled) - dau hieu
estimator "cam nhan" duoc ghep noi, du chua can chinh xac tuyet doi o Pha P.
"""

from __future__ import annotations

import numpy as np
import pytest

from pqrst.baselines.ksg import KSGTEEstimator
from pqrst.baselines.binning import BinningTEEstimator
from pqrst.baselines.symbolic_te import SymbolicTEEstimator


class TestKSGTEEstimator:
    def test_smoke_finite_output(self):
        from pqrst.data.synthetic.var_linear_gaussian import generate_var_linear_gaussian
        x, y, _ = generate_var_linear_gaussian(200, 0.5, 0.5, 0.2, 0.1, 42)
        est = KSGTEEstimator()
        te = est.estimate(x, y)
        assert np.isfinite(te)

    def test_coupled_greater_than_uncoupled(self):
        from pqrst.data.synthetic.var_linear_gaussian import generate_var_linear_gaussian
        x_coup, y_coup, _ = generate_var_linear_gaussian(500, 0.5, 0.5, 0.8, 0.1, 42)
        est = KSGTEEstimator()
        te_coup = est.estimate(x_coup, y_coup)
        
        x_unc, y_unc, _ = generate_var_linear_gaussian(500, 0.5, 0.5, 0.0, 0.1, 42)
        te_unc = est.estimate(x_unc, y_unc)
        
        assert te_coup > te_unc


class TestBinningTEEstimator:
    def test_smoke_finite_output(self):
        from pqrst.data.synthetic.var_linear_gaussian import generate_var_linear_gaussian
        x, y, _ = generate_var_linear_gaussian(200, 0.5, 0.5, 0.2, 0.1, 42)
        est = BinningTEEstimator()
        te = est.estimate(x, y)
        assert np.isfinite(te)


class TestSymbolicTEEstimator:
    def test_smoke_finite_output(self):
        from pqrst.data.synthetic.var_linear_gaussian import generate_var_linear_gaussian
        x, y, _ = generate_var_linear_gaussian(200, 0.5, 0.5, 0.2, 0.1, 42)
        est = SymbolicTEEstimator()
        te = est.estimate(x, y)
        assert np.isfinite(te)
