"""Test cho MINE (StatisticsNetwork, loss, shuffle_batch, train_mine). Xem checklist
Pha Q trong docs/PHASE_Q_GUIDE.md muc 7.

Cac test duoi day la KHUNG (skeleton) - than ham con thieu, ban tu dien logic assert.
"""

from __future__ import annotations

import pytest
import torch
import numpy as np
from pqrst.estimators.mine.network import StatisticsNetwork
from pqrst.estimators.mine.losses import shuffle_batch, donsker_varadhan_loss
from pqrst.estimators.mine.train import train_mine, TrainConfig
from pqrst.data.synthetic.var_linear_gaussian import generate_var_linear_gaussian, compute_lag1_mi_ground_truth

class TestStatisticsNetwork:
    def test_output_shape(self):
        net = StatisticsNetwork(input_dim=2, hidden_dims=[64, 64])
        x = torch.randn(32, 1)
        y = torch.randn(32, 1)
        out = net(x, y)
        assert out.shape == (32,)

    def test_reproducible_with_same_seed(self):
        torch.manual_seed(42)
        net1 = StatisticsNetwork(input_dim=2, hidden_dims=[64, 64])
        
        torch.manual_seed(42)
        net2 = StatisticsNetwork(input_dim=2, hidden_dims=[64, 64])
        
        for p1, p2 in zip(net1.parameters(), net2.parameters()):
            assert torch.equal(p1, p2)


class TestLosses:
    def test_shuffle_batch_permutes_rows(self):
        y = torch.arange(100).unsqueeze(-1)
        torch.manual_seed(42)
        y_shuffled = shuffle_batch(y)
        
        assert y_shuffled.shape == y.shape
        assert not torch.equal(y, y_shuffled)
        assert torch.equal(torch.sort(y.squeeze())[0], torch.sort(y_shuffled.squeeze())[0])

    def test_donsker_varadhan_loss_finite(self):
        t_joint = torch.randn(32)
        t_marginal = torch.randn(32)
        loss = donsker_varadhan_loss(t_joint, t_marginal)
        assert torch.isfinite(loss)

    def test_donsker_varadhan_loss_higher_for_independent_marginal(self):
        t_joint = torch.ones(32) * 5
        t_marginal_high = torch.ones(32) * 10
        t_marginal_low = torch.ones(32) * -10
        
        loss_high = donsker_varadhan_loss(t_joint, t_marginal_high)
        loss_low = donsker_varadhan_loss(t_joint, t_marginal_low)
        
        # Loss = -DV. High DV means lower loss.
        # DV = mean(T_joint) - logsumexp(T_marginal).
        # t_marginal_low will give high DV -> lower loss.
        assert loss_low < loss_high


class TestTrainMine:
    def test_loss_decreases_on_correlated_gaussian(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 5000)
        y = x * 0.8 + rng.normal(0, 0.5, 5000)
        
        config = TrainConfig(hidden_dims=[32], max_epochs=20, patience=20, batch_size=256)
        result = train_mine(x, y, config)
        
        assert result.train_loss_history[-1] < result.train_loss_history[0]

    def test_mi_estimate_matches_ground_truth_on_var_linear_gaussian(self):
        x_pool = []
        y_pool = []
        a, b, c, noise_std = 0.5, 0.5, 0.6, 0.5
        for i in range(10):  # scaled down for test speed
            x, y, _ = generate_var_linear_gaussian(1000, a, b, c, noise_std, seed=42+i)
            x_pool.append(x[:-1])
            y_pool.append(y[1:])
        x_pool = np.concatenate(x_pool)
        y_pool = np.concatenate(y_pool)
        
        gt_mi = compute_lag1_mi_ground_truth(a, b, c, noise_std)
        config = TrainConfig(hidden_dims=[32, 32], max_epochs=30, batch_size=256)
        result = train_mine(x_pool, y_pool, config)
        
        # At least check it converges to positive MI not too far off
        assert result.final_val_mi_estimate > 0.0
        assert abs(result.final_val_mi_estimate - gt_mi) / gt_mi < 0.5
