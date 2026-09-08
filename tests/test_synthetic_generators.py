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
        pytest.skip("TODO: implement sau khi generate_var_linear_gaussian xong")

    def test_reproducible_with_same_seed(self):
        pytest.skip("TODO: implement")

    def test_uncoupled_ground_truth_near_zero(self):
        pytest.skip("TODO: implement - c=0 thi te_ground_truth phai gan 0")


class TestVarNonlinear:
    def test_output_shape(self):
        pytest.skip("TODO: implement sau khi generate_var_nonlinear xong")

    def test_reproducible_with_same_seed(self):
        pytest.skip("TODO: implement")


class TestPeriodicCoupling:
    def test_output_shape(self):
        pytest.skip("TODO: implement sau khi generate_periodic_coupling xong")

    def test_reproducible_with_same_seed(self):
        pytest.skip("TODO: implement")

    def test_output_range_bounded(self):
        """x, y phai trong [-1, 1] vi la sin() cua pha."""
        pytest.skip("TODO: implement")
