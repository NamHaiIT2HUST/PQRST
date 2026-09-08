"""Bo sinh du lieu ghep noi tuan hoan (2 dao dong pha-ghep, kieu Kuramoto don gian hoa).

Mo phong gan hon voi ghep noi sinh ly tim-nao so voi VAR. Mo hinh de xuat:
    phi_x[t+1] = phi_x[t] + omega_x + eps_x[t]
    phi_y[t+1] = phi_y[t] + omega_y + K * sin(phi_x[t] - phi_y[t]) + eps_y[t]
    x[t] = sin(phi_x[t])
    y[t] = sin(phi_y[t])

K la cuong do ghep noi pha X -> Y. K == 0 nghia la khong ghep noi (2 dao dong doc lap).
Khong co cong thuc dong cho TE - dung pseudo ground-truth (N rat lon), giong nhu
var_nonlinear.py. Xem chi tiet trong docs/PHASE_P_GUIDE.md, muc 2.3.
"""

from __future__ import annotations

import numpy as np


def generate_periodic_coupling(
    n_samples: int,
    omega_x: float,
    omega_y: float,
    coupling_k: float,
    noise_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sinh 1 cap chuoi (x, y) tu mo hinh 2 dao dong pha-ghep o tren.

    Args:
        n_samples: do dai chuoi thoi gian can sinh.
        omega_x: tan so goc rieng cua dao dong X (radian/step).
        omega_y: tan so goc rieng cua dao dong Y (radian/step).
        coupling_k: cuong do ghep noi pha X -> Y.
        noise_std: do lech chuan nhieu Gaussian cong vao pha moi buoc.
        seed: seed cho RNG.

    Returns:
        (x, y): mang 1D shape (n_samples,) moi mang, gia tri trong [-1, 1] (= sin cua pha).

    TODO(ban tu code):
        - phi_x, phi_y nen duoc mod 2*pi moi buoc de tranh tran so (khong bat buoc ve toan
          hoc vi sin() tuan hoan, nhung giup on dinh so hoc khi N lon).
        - Ground-truth TE: viet script rieng tinh pseudo ground-truth o N lon, luu vao
          configs/synthetic/periodic_coupling.yaml, tuong tu var_nonlinear.
    """
    rng = np.random.default_rng(seed)
    burn_in = 100
    total_samples = n_samples + burn_in
    
    phi_x = np.zeros(total_samples)
    phi_y = np.zeros(total_samples)
    
    eps_x = rng.normal(0, noise_std, total_samples)
    eps_y = rng.normal(0, noise_std, total_samples)
    
    for t in range(0, total_samples - 1):
        phi_x[t+1] = (phi_x[t] + omega_x + eps_x[t]) % (2 * np.pi)
        phi_y[t+1] = (phi_y[t] + omega_y + coupling_k * np.sin(phi_x[t] - phi_y[t]) + eps_y[t]) % (2 * np.pi)
        
    x = np.sin(phi_x)
    y = np.sin(phi_y)
    
    return x[burn_in:], y[burn_in:]
