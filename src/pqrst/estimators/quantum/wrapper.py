"""T_theta(x,y) -> scalar - wrapper nn.Module cho mach luong tu (circuit.py), CUNG
CHU KY forward(y_t, x_lag, y_lag, mask) voi MaskedStatisticsNetwork (Nhip 1) va
FourierFeatureStatisticsNetwork (Pha P'.0) - cam duoc thang vao
estimate_te_from_window(), AmortizedTEEstimator, train_amortized(model_factory=...)
va src/pqrst/evaluation/feature_space.py (qua forward_blocks()) MA KHONG SUA GI
THEM o cac noi do - dung theo dung nguyen tac "hoan doi co kiem soat" cua Nhip 2
(docs/NHIP2_GUIDE.md).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from pqrst.estimators.quantum.circuit import make_circuit


class QuantumStatisticsNetwork(nn.Module):
    """T_theta dang mach luong tu data re-uploading - xem circuit.py de biet chi
    tiet + canh bao quan trong ve entangling (BAT BUOC doc truoc khi doi thiet ke).

    Tham so hoc duoc: encode_scale (n_qubits), theta (n_layers*n_qubits*2), readout
    (n_qubits+1). Vi du n_qubits=6, n_layers=4: 6 + 48 + 7 = 61 tham so.
    """

    def __init__(self, n_qubits: int = 6, n_layers: int = 4, device_name: str = "lightning.qubit"):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Gan CO DINH (khong hoc) tung qubit cho 1 trong 4 dac trung dau vao
        # (y_t=0, x_lag_masked=1, y_lag=2, mask=3), lap vong tron neu n_qubits>4 -
        # dam bao CA X (x_lag) va Y (y_t, y_lag) deu duoc encode vao it nhat 1
        # qubit, roi entangling (circuit.py) se tron chung voi nhau.
        feature_idx = torch.tensor([q % 4 for q in range(n_qubits)], dtype=torch.long)
        self.register_buffer("feature_idx", feature_idx)

        self.encode_scale = nn.Parameter(torch.rand(n_qubits) * 1.5 + 0.5)
        self.theta = nn.Parameter(torch.randn(n_layers, n_qubits, 2) * 0.1)
        self.readout = nn.Linear(n_qubits, 1)

        self.circuit = make_circuit(n_qubits, n_layers, device_name)

    def _encode(self, y_t: torch.Tensor, x_lag: torch.Tensor, y_lag: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_lag_masked = x_lag * mask
        xy = torch.cat([y_t, x_lag_masked, y_lag, mask], dim=-1)  # (batch, 4)
        raw = xy[:, self.feature_idx]  # (batch, n_qubits) - gather theo feature_idx
        return raw * self.encode_scale  # (batch, n_qubits)

    def forward(
        self,
        y_t: torch.Tensor,
        x_lag: torch.Tensor,
        y_lag: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        inputs = self._encode(y_t, x_lag, y_lag, mask)
        z_list = self.circuit(inputs, self.theta)  # list of n_qubits tensors (batch,)
        z_stack = torch.stack(z_list, dim=-1)  # (batch, n_qubits)
        out = self.readout(z_stack)
        return out.squeeze(-1)

    def forward_blocks(
        self,
        y_t: torch.Tensor,
        x_lag: torch.Tensor,
        y_lag: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Cung interface voi MaskedStatisticsNetwork/FourierFeatureStatisticsNetwork
        de dung duoc voi src/pqrst/evaluation/feature_space.py (PCA-theo-khoi,
        Pha T.3b) khong can sua gi o module do. 'block0_input' = 4 dac trung co
        dien goc; 'block1' = vector <Z> tung qubit SAU toan bo mach (truoc readout)."""
        x_lag_masked = x_lag * mask
        xy = torch.cat([y_t, x_lag_masked, y_lag, mask], dim=-1)
        inputs = self._encode(y_t, x_lag, y_lag, mask)
        z_list = self.circuit(inputs, self.theta)
        z_stack = torch.stack(z_list, dim=-1)
        return {"block0_input": xy, "block1": z_stack}
