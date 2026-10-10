"""Mo hinh lai Co dien - Luong tu (Hybrid Classical-Quantum / MLP + VQC).

Kien truc HQNN (Hybrid Quantum Neural Network):
1. Classical Encoder (MLP): Nen dac trung dau vao (y_t, x_lag_masked, y_lag, mask)
   thanh vector dac trung an h in R^{n_qubits}, chuan hoa goc quay vao [-pi, pi].
2. Quantum VQC Core (PennyLane): Mach data re-uploading n_qubits x n_layers voi
   cac phep quay tham so hoa RY, RZ va cong vuong mac CNOT, do gia tri ky vong <Z>.
3. Classical Readout (MLP): Bien doi khong gian do luong tu thanh gia tri logit T_phi(x,y).

Giai quyet 2 nhuoc diem lon:
- So voi Pure VQC: Khong bi ep buoc gan truc tiep dac trung tho vao goc quay cung nhac.
- So voi Classical MLP: Khai thac khong gian Hilbert luong tu de mo hinh hoa tuong tac
  phi tuyen phi cuc bo (entanglement).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pqrst.estimators.quantum.circuit import make_circuit


class HybridClassicalQuantumStatisticsNetwork(nn.Module):
    """Mô hình thống kê lai MLP + VQC cho bài toán MINE / Transfer Entropy."""

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 3,
        encoder_hidden_dim: int = 32,
        readout_hidden_dim: int = 16,
        device_name: str = "lightning.qubit",
        mask_conditioning: bool = False,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.mask_conditioning = mask_conditioning

        # 1. Classical Encoder (MLP): 4 -> encoder_hidden_dim -> n_qubits
        self.encoder = nn.Sequential(
            nn.Linear(4, encoder_hidden_dim),
            nn.ELU(),
            nn.Linear(encoder_hidden_dim, n_qubits),
            nn.Tanh(),  # Bounded to [-1, 1], multiplied by pi to form rotation angles in [-pi, pi]
        )

        # 2. Quantum Core: Variational Quantum Circuit
        self.circuit = make_circuit(n_qubits, n_layers, device_name, mask_conditioning)
        self.theta = nn.Parameter(torch.randn(n_layers, n_qubits, 2) * 0.1)

        if mask_conditioning:
            self.mask_theta = nn.Parameter(torch.zeros(n_layers, n_qubits))

        # 3. Classical Readout: n_qubits -> readout_hidden_dim -> 1
        self.readout = nn.Sequential(
            nn.Linear(n_qubits, readout_hidden_dim),
            nn.ELU(),
            nn.Linear(readout_hidden_dim, 1),
        )

    def _encode(
        self, y_t: torch.Tensor, x_lag: torch.Tensor, y_lag: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        x_lag_masked = x_lag * mask
        xy = torch.cat([y_t, x_lag_masked, y_lag, mask], dim=-1)  # (batch, 4)
        latent = self.encoder(xy)  # (batch, n_qubits) in [-1, 1]
        return latent * torch.pi

    def _run_circuit(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.mask_conditioning:
            z_list = self.circuit(inputs, self.theta, mask[:, 0], self.mask_theta)
        else:
            z_list = self.circuit(inputs, self.theta)
        return torch.stack(z_list, dim=-1)  # (batch, n_qubits)

    def forward(
        self,
        y_t: torch.Tensor,
        x_lag: torch.Tensor,
        y_lag: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        inputs = self._encode(y_t, x_lag, y_lag, mask)
        z_stack = self._run_circuit(inputs, mask)
        out = self.readout(z_stack)
        return out.squeeze(-1)

    def forward_blocks(
        self,
        y_t: torch.Tensor,
        x_lag: torch.Tensor,
        y_lag: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Cung cap cac bieu dien dac trung theo tung tang de phan tich PCA."""
        x_lag_masked = x_lag * mask
        xy = torch.cat([y_t, x_lag_masked, y_lag, mask], dim=-1)
        latent_classical = self.encoder(xy)
        inputs = latent_classical * torch.pi
        z_stack = self._run_circuit(inputs, mask)

        # Layer 1 cua readout
        h_readout = self.readout[1](self.readout[0](z_stack))

        return {
            "block0_input": xy,
            "block1_classical_latent": latent_classical,
            "block2_quantum_latent": z_stack,
            "block3_readout_latent": h_readout,
        }
