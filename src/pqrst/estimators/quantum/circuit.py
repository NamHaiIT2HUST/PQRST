"""Pha P' (Nhip 2) - mach luong tu kieu data re-uploading dung lam T_theta, thay
cho T_phi (MLP, Nhip 1) tai DUNG 1 diem noi trong src/pqrst/estimators/ (khong
sua gi khac - xem docs/NHIP2_GUIDE.md muc 3).

Thiet ke (theo dung nguyen tac data re-uploading, Perez-Salinas et al. 2020, va
ly thuyet Fourier cua Schuld, Sweke & Meyer 2021 - da dung lam co so cho
FourierFeatureStatisticsNetwork o Pha P'.0): moi lop gom
    (1) ma hoa du lieu: RY(feature * encode_scale[qubit]) - encode_scale HOC DUOC,
        feature la 1 trong 4 dau vao co dien (y_t, x_lag_masked, y_lag, mask),
    (2) xoay co tham so: RY(theta1), RZ(theta2) HOC DUOC,
    (3) entangling: CNOT theo vong (CNOT(q, (q+1) mod n_qubits) voi moi q).

CANH BAO QUAN TRONG (rut ra tu bug THAT da tim va sua o Pha P'.0 - xem
classical_fourier.py va docs/NHIP2_GUIDE.md muc 2.1): neu BO buoc (3) hoac dat
CNOT SAI cho khong noi 2 qubit encode 2 dac trung KHAC NHAU (vd chi CNOT giua
2 qubit CUNG encode y_t), ket qua do (vd <Z> tung qubit doc lap) van co the la
HAM TACH ROI CONG TINH cua (y_t, x_lag, y_lag, mask) - va theo dung bat dang
thuc Jensen da chung minh o P'.0, DV bound se bi chan o 0 VI LY DO TOAN HOC
GIONG HET, khong lien quan gi den "mach luong tu chua du manh". Vong CNOT day
du (ket noi TAT CA qubit voi nhau qua cac lop) la dieu kien toi thieu de tranh
loi nay - da kiem tra bang test rieng (xem tests/test_quantum_circuit.py).

Hieu suat (quan trong khi lap ke hoach Q'): mo phong co dien tren CPU (khong co
GPU trong moi truong nay) qua PennyLane lightning.qubit, dung tham so hoa
(parameter broadcasting) de xu ly ca 1 cua so N mau trong 1 lan goi QNode.
Do o P'.0 (batch=200, 6 qubit, 4 lop): ~1.4s/lan goi FORWARD (chua tinh nguoc).
=> train tren TOAN BO corpus 27.000 cua so nhu Pha R (T_phi) la KHONG kha thi
tren may khong GPU (uoc tinh hang chuc gio/epoch) - Pha Q' PHAI thu nho quy mo
(it cua so hon, it epoch hon, hoac chi chung minh khai niem tren 1 tap con) -
xem docs/NHIP2_GUIDE.md muc 4 (da cap nhat canh bao nay).
"""
from __future__ import annotations

import pennylane as qml
import torch


def make_circuit(n_qubits: int, n_layers: int, device_name: str = "lightning.qubit"):
    """Tra ve 1 QNode(inputs, theta) -> list[Tensor] (n_qubits phan tu, moi phan tu
    shape (batch,)) = <PauliZ> tung qubit sau n_layers lop data re-uploading.

    Args:
        n_qubits: so qubit (khuyen nghi 6-8, xem docs/NHIP2_GUIDE.md).
        n_layers: so lop re-uploading (khuyen nghi 3-5).
        device_name: ten device PennyLane ("lightning.qubit" nhanh hon
            "default.qubit" tren CPU cho mach nho nay).

    inputs: Tensor shape (batch, n_qubits) - GOC quay ma hoa da nhan san voi
        encode_scale (xem wrapper.py) - ham nay KHONG tu nhan scale.
    theta: Tensor shape (n_layers, n_qubits, 2) - tham so hoc duoc (RY, RZ moi qubit
        moi lop).
    """
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def circuit(inputs: torch.Tensor, theta: torch.Tensor):
        for l in range(n_layers):
            for q in range(n_qubits):
                qml.RY(inputs[:, q], wires=q)
            for q in range(n_qubits):
                qml.RY(theta[l, q, 0], wires=q)
                qml.RZ(theta[l, q, 1], wires=q)
            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])
        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    return circuit
