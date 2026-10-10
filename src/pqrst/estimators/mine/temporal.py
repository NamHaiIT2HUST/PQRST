"""Kien truc mang tich chap 1 chieu (1D-CNN / Temporal Statistics Network).

Dung lam baseline mo hinh hoc sau chuoi thoi gian (Temporal model) de so sanh
doi chung voi MLP co ban, Fourier Features, Pure VQC va Hybrid MLP+VQC.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Conv1DStatisticsNetwork(nn.Module):
    """Mô hình thống kê dựa trên mạng tích chập 1D (Conv1D) cho ước lượng TE."""

    def __init__(self, in_features: int = 4, hidden_channels: list[int] | None = None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [32, 64]

        # Convolutional feature extractor
        # Coi vector [y_t, x_lag_masked, y_lag, mask] nhu 1 chuoi 1D voi in_channels=1, length=4
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=hidden_channels[0], kernel_size=2, padding=1)
        self.act1 = nn.ELU()
        self.conv2 = nn.Conv1d(in_channels=hidden_channels[0], out_channels=hidden_channels[1], kernel_size=2)
        self.act2 = nn.ELU()
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Dense readout head
        self.fc1 = nn.Linear(hidden_channels[1], 32)
        self.act3 = nn.ELU()
        self.fc_out = nn.Linear(32, 1)

    def forward(
        self,
        y_t: torch.Tensor,
        x_lag: torch.Tensor,
        y_lag: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        x_lag_masked = x_lag * mask
        xy = torch.cat([y_t, x_lag_masked, y_lag, mask], dim=-1)  # (batch, 4)
        xy_seq = xy.unsqueeze(1)  # (batch, 1, 4)

        h1 = self.act1(self.conv1(xy_seq))
        h2 = self.act2(self.conv2(h1))
        pooled = self.pool(h2).squeeze(-1)  # (batch, 64)

        dense = self.act3(self.fc1(pooled))
        out = self.fc_out(dense)
        return out.squeeze(-1)

    def forward_blocks(
        self,
        y_t: torch.Tensor,
        x_lag: torch.Tensor,
        y_lag: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x_lag_masked = x_lag * mask
        xy = torch.cat([y_t, x_lag_masked, y_lag, mask], dim=-1)
        xy_seq = xy.unsqueeze(1)

        h1 = self.act1(self.conv1(xy_seq))
        h2 = self.act2(self.conv2(h1))
        pooled = self.pool(h2).squeeze(-1)
        dense = self.act3(self.fc1(pooled))

        return {
            "block0_input": xy,
            "block1_conv": pooled,
            "block2_dense": dense,
        }
