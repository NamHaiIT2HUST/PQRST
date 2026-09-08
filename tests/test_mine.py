"""Test cho MINE (StatisticsNetwork, loss, shuffle_batch, train_mine). Xem checklist
Pha Q trong docs/PHASE_Q_GUIDE.md muc 7.

Cac test duoi day la KHUNG (skeleton) - than ham con thieu, ban tu dien logic assert.
"""

from __future__ import annotations

import pytest


class TestStatisticsNetwork:
    def test_output_shape(self):
        """forward(x, y) voi x,y shape (batch, 1) phai tra ve tensor shape (batch,)."""
        pytest.skip("TODO: implement sau khi StatisticsNetwork xong")

    def test_reproducible_with_same_seed(self):
        """torch.manual_seed(seed) truoc khi khoi tao 2 network giong kien truc phai
        cho weight khoi tao giong het nhau."""
        pytest.skip("TODO: implement")


class TestLosses:
    def test_shuffle_batch_permutes_rows(self):
        """shuffle_batch(y) phai la 1 hoan vi cua y (cung tap gia tri, thu tu khac -
        voi batch_size > 1, xac suat rat cao thu tu se khac ban goc)."""
        pytest.skip("TODO: implement")

    def test_donsker_varadhan_loss_finite(self):
        """Loss tren du lieu random phai la so thuc huu han (khong NaN/Inf)."""
        pytest.skip("TODO: implement")

    def test_donsker_varadhan_loss_higher_for_independent_marginal(self):
        """Neu t_marginal >> t_joint (mang "nham lam" marginal giong joint), DV bound
        phai NHO hon so voi truong hop t_marginal thap - kiem tra dau/huong cua loss
        dung voi dinh nghia DV bound."""
        pytest.skip("TODO: implement")


class TestTrainMine:
    def test_loss_decreases_on_correlated_gaussian(self):
        """Train vai chuc epoch tren du lieu Gaussian tuong quan don gian (KHONG can
        dung generate_var_linear_gaussian - co the tu sinh nhanh bang
        np.random.default_rng trong test), kiem tra val_loss_history giam dan (vd
        loss epoch cuoi < loss epoch dau)."""
        pytest.skip("TODO: implement")

    def test_mi_estimate_matches_ground_truth_on_var_linear_gaussian(self):
        """Test cham (slow) - tuong duong quy mo scripts/train_mine_smoke_test.py
        nhung kiem tra bang assert thay vi doc bang mat. Co the danh dau
        @pytest.mark.slow hoac giam quy mo (it epoch/it sample hon) neu chay lau qua
        trong CI, nhung phai giu du lon de MI estimate co y nghia thong ke."""
        pytest.skip("TODO: implement")
