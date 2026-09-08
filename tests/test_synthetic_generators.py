"""Test cho 3 bo sinh du lieu tong hop. Xem checklist Pha P trong docs/PHASE_P_GUIDE.md.

Cac test duoi day la KHUNG (skeleton) - than ham con thieu, ban tu dien logic assert.
Muc tieu toi thieu cho moi bo sinh (nhac lai tu docs/PHASE_P_GUIDE.md muc 2.4):
    - Shape output dung n_samples.
    - Seed co dinh -> ket qua tai lap duoc (chay 2 lan cung seed ra y het).
    - (rieng var_linear_gaussian) TE ground-truth (cong thuc dong) khop voi TE uoc luong
      bang KSG o N lon - sanity check cheo.
    - Truong hop khong ghep noi (c=0 hoac coupling_k=0) -> TE (ground-truth hoac pseudo)
      phai gan 0.
"""

from __future__ import annotations

import numpy as np
import pytest

from pqrst.data.synthetic.var_linear_gaussian import generate_var_linear_gaussian
from pqrst.data.synthetic.var_nonlinear import generate_var_nonlinear
from pqrst.data.synthetic.periodic_coupling import generate_periodic_coupling


class TestVarLinearGaussian:
    def test_output_shape(self):
        x, y, te = generate_var_linear_gaussian(100, 0.5, 0.5, 0.1, 0.1, 42)
        assert x.shape == (100,)
        assert y.shape == (100,)
        assert isinstance(te, float)

    def test_reproducible_with_same_seed(self):
        x1, y1, te1 = generate_var_linear_gaussian(100, 0.5, 0.5, 0.1, 0.1, 42)
        x2, y2, te2 = generate_var_linear_gaussian(100, 0.5, 0.5, 0.1, 0.1, 42)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)
        assert te1 == te2

    def test_uncoupled_ground_truth_near_zero(self):
        x, y, te = generate_var_linear_gaussian(100, 0.5, 0.5, 0.0, 0.1, 42)
        assert np.isclose(te, 0.0)

    def test_ground_truth_matches_ksg_at_large_n(self):
        """Sanity check cheo bat buoc (PHASE_P_GUIDE.md muc 2.4): cong thuc dong
        te_ground_truth phai khop voi TE uoc luong boi KSG (estimator lien tuc, khong
        gia dinh gi ve dang tuyen tinh) o N lon. Day chinh la test le ra phai bat duoc
        loi cong thuc ground-truth cu (gia dinh sai X[t-1] doc lap Y[t-1] khi a != 0).
        """
        from pqrst.baselines.ksg import KSGTEEstimator

        x, y, te_gt = generate_var_linear_gaussian(200_000, 0.5, 0.5, 0.6, 0.5, seed=42)
        te_ksg = KSGTEEstimator().estimate(x, y)

        assert abs(te_ksg - te_gt) / te_gt < 0.10


class TestVarNonlinear:
    def test_output_shape(self):
        x, y = generate_var_nonlinear(100, 0.5, 0.5, 0.1, 0.1, 42)
        assert x.shape == (100,)
        assert y.shape == (100,)

    def test_reproducible_with_same_seed(self):
        x1, y1 = generate_var_nonlinear(100, 0.5, 0.5, 0.1, 0.1, 42)
        x2, y2 = generate_var_nonlinear(100, 0.5, 0.5, 0.1, 0.1, 42)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)


class TestPeriodicCoupling:
    def test_output_shape(self):
        x, y = generate_periodic_coupling(100, 0.1, 0.2, 0.1, 0.01, 42)
        assert x.shape == (100,)
        assert y.shape == (100,)

    def test_reproducible_with_same_seed(self):
        x1, y1 = generate_periodic_coupling(100, 0.1, 0.2, 0.1, 0.01, 42)
        x2, y2 = generate_periodic_coupling(100, 0.1, 0.2, 0.1, 0.01, 42)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)

    def test_output_range_bounded(self):
        """x, y phai trong [-1, 1] vi la sin() cua pha."""
        x, y = generate_periodic_coupling(100, 0.1, 0.2, 0.1, 0.01, 42)
        assert np.all(x >= -1.0) and np.all(x <= 1.0)
        assert np.all(y >= -1.0) and np.all(y <= 1.0)
