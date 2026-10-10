from __future__ import annotations

import torch
import pytest

from pqrst.estimators.quantum.hybrid_vqc import HybridClassicalQuantumStatisticsNetwork
from pqrst.estimators.mine.temporal import Conv1DStatisticsNetwork


def test_hybrid_vqc_forward_and_shapes():
    batch_size = 8
    model = HybridClassicalQuantumStatisticsNetwork(
        n_qubits=4, n_layers=2, encoder_hidden_dim=16, readout_hidden_dim=8
    )

    y_t = torch.randn(batch_size, 1)
    x_lag = torch.randn(batch_size, 1)
    y_lag = torch.randn(batch_size, 1)
    mask = torch.ones(batch_size, 1)

    out = model(y_t, x_lag, y_lag, mask)
    assert out.shape == (batch_size,)

    blocks = model.forward_blocks(y_t, x_lag, y_lag, mask)
    assert "block0_input" in blocks
    assert "block1_classical_latent" in blocks
    assert "block2_quantum_latent" in blocks
    assert "block3_readout_latent" in blocks
    assert blocks["block1_classical_latent"].shape == (batch_size, 4)
    assert blocks["block2_quantum_latent"].shape == (batch_size, 4)


def test_hybrid_vqc_backward_gradients():
    model = HybridClassicalQuantumStatisticsNetwork(
        n_qubits=4, n_layers=2, encoder_hidden_dim=16, readout_hidden_dim=8
    )
    y_t = torch.randn(4, 1)
    x_lag = torch.randn(4, 1)
    y_lag = torch.randn(4, 1)
    mask = torch.ones(4, 1)

    out = model(y_t, x_lag, y_lag, mask)
    loss = out.mean()
    loss.backward()

    # Check classical encoder params have gradients
    for p in model.encoder.parameters():
        assert p.grad is not None
        assert not torch.all(p.grad == 0)

    # Check quantum theta has gradients
    assert model.theta.grad is not None

    # Check readout params have gradients
    for p in model.readout.parameters():
        assert p.grad is not None


def test_hybrid_vqc_mask_zeroes_xlag():
    model = HybridClassicalQuantumStatisticsNetwork(
        n_qubits=4, n_layers=2, encoder_hidden_dim=16, readout_hidden_dim=8
    )
    model.eval()

    y_t = torch.randn(4, 1)
    y_lag = torch.randn(4, 1)
    mask_zero = torch.zeros(4, 1)

    x_lag_1 = torch.randn(4, 1)
    x_lag_2 = torch.randn(4, 1) + 10.0  # Very different

    with torch.no_grad():
        out1 = model(y_t, x_lag_1, y_lag, mask_zero)
        out2 = model(y_t, x_lag_2, y_lag, mask_zero)

    torch.testing.assert_close(out1, out2)


def test_conv1d_forward_and_shapes():
    batch_size = 8
    model = Conv1DStatisticsNetwork(hidden_channels=[16, 32])

    y_t = torch.randn(batch_size, 1)
    x_lag = torch.randn(batch_size, 1)
    y_lag = torch.randn(batch_size, 1)
    mask = torch.ones(batch_size, 1)

    out = model(y_t, x_lag, y_lag, mask)
    assert out.shape == (batch_size,)

    blocks = model.forward_blocks(y_t, x_lag, y_lag, mask)
    assert "block0_input" in blocks
    assert "block1_conv" in blocks
    assert "block2_dense" in blocks


def test_conv1d_mask_zeroes_xlag():
    model = Conv1DStatisticsNetwork(hidden_channels=[16, 32])
    model.eval()

    y_t = torch.randn(4, 1)
    y_lag = torch.randn(4, 1)
    mask_zero = torch.zeros(4, 1)

    x_lag_1 = torch.randn(4, 1)
    x_lag_2 = torch.randn(4, 1) + 10.0

    with torch.no_grad():
        out1 = model(y_t, x_lag_1, y_lag, mask_zero)
        out2 = model(y_t, x_lag_2, y_lag, mask_zero)

    torch.testing.assert_close(out1, out2)
