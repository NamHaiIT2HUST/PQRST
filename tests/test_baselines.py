"""Smoke test cho 3 baseline (KSG, symbolic TE, binning). Xem checklist Pha P.

Muc tieu toi thieu: moi estimator chay khong loi tren du lieu tong hop don gian,
tra ve so thuc huu han (khong NaN/Inf), va TE(coupled) > TE(uncoupled) - dau hieu
estimator "cam nhan" duoc ghep noi, du chua can chinh xac tuyet doi o Pha P.
"""

from __future__ import annotations

import pytest

from pqrst.baselines.ksg import KSGTEEstimator
from pqrst.baselines.binning import BinningTEEstimator
from pqrst.baselines.symbolic_te import SymbolicTEEstimator


class TestKSGTEEstimator:
    def test_smoke_finite_output(self):
        pytest.skip("TODO: implement sau khi KSGTEEstimator.estimate xong")

    def test_coupled_greater_than_uncoupled(self):
        pytest.skip("TODO: implement")


class TestBinningTEEstimator:
    def test_smoke_finite_output(self):
        pytest.skip("TODO: implement sau khi BinningTEEstimator.estimate xong")


class TestSymbolicTEEstimator:
    def test_smoke_finite_output(self):
        pytest.skip("TODO: implement sau khi SymbolicTEEstimator.estimate xong")
