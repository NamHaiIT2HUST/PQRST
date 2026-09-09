"""Sanity check dinh huong TE(tim->nao) vs TE(nao->tim) + kiem dinh y nghia thong ke.

Day la TIEU CHI THOAT chinh cua Pha S (roadmap goc): tai lap duoc
TE(tim->nao) > TE(nao->tim) tren it nhat 1 bo du lieu co nhan ro (slpdb hoac capslpdb).

*** DOC docs/PHASE_S_GUIDE.md MUC 3.2 TRUOC KHI DUNG MODULE NAY ***
Nhieu truong tim (cardiac field artifact) co the lam sanity check DAT vi ly do hoan
toan sai. PHAI chay eeg.detect_cardiac_artifact() truoc va bao cao ket qua do KEM
theo ket qua sanity check - khong duoc bao cao rieng le.
"""

from __future__ import annotations

import numpy as np


def bidirectional_te(
    windows_forward: list,
    windows_backward: list,
    estimator,
) -> dict:
    """Tinh TE theo ca 2 chieu tren cung du lieu.

    Args:
        windows_forward: cac Window voi source=tim, target=nao (chieu tim->nao).
        windows_backward: cac Window voi source=nao, target=tim (chieu nguoc lai).
            PHAI duoc tao tu CUNG doan du lieu, chi doi vai tro 2 kenh - neu dung 2
            doan khac nhau thi phep so sanh khong con cong bang.
        estimator: BaseTEEstimator (T_phi da chuan hoa, hoac KSG).

    Returns:
        dict gom "te_forward_mean", "te_backward_mean", "te_forward_values",
        "te_backward_values", "difference" (= forward - backward), va so cua so dung.

    TODO(ban tu code):
        - Chay estimator tren tung cua so o ca 2 danh sach, thu thap gia tri tho
          (giu lai TUNG gia tri, khong chi trung binh - can cho bootstrap/CI o duoi).
        - Bat loi tung cua so, ghi NaN va dem, khong de vo vong lap (giong grid.py).
    """
    raise NotImplementedError


def permutation_test_te(
    windows: list,
    estimator,
    n_permutations: int = 500,
    seed: int = 42,
) -> dict:
    """Kiem dinh hoan vi: TE quan sat duoc co lon hon muc ngau nhien khong?

    VI SAO BAT BUOC: uoc luong TE gan nhu luon ra so DUONG do bias huu han mau, ke ca
    khi khong co ghep noi that. Vi vay "TE(A->B) > 0" hay "TE(A->B) > TE(B->A) mot
    chut" deu CHUA chung minh duoc dieu gi neu khong co phan phoi null de doi chieu.

    Args:
        windows: cac Window can kiem dinh.
        estimator: BaseTEEstimator.
        n_permutations: so lan hoan vi (>=500 theo roadmap Pha T).
        seed: de tai lap.

    Returns:
        dict gom "te_observed", "null_distribution", "p_value", "significant".

    TODO(ban tu code):
        - TE quan sat: chay estimator tren du lieu nguyen ban.
        - Phan phoi null: moi lan hoan vi, PHA VO quan he thoi gian giua source va
          target NHUNG GIU NGUYEN phan phoi bien cua tung kenh. Cach dung: hoan vi
          THU TU CAC CUA SO cua kenh nguon (ghep cua so nguon i voi cua so dich j!=i),
          hoac dich vong (circular shift) chuoi nguon mot luong ngau nhien.
          KHONG hoan vi ngau nhien tung diem trong cua so - lam vay se pha luon cau
          truc tu tuong quan cua chinh kenh nguon, tao null qua de, p-value se nho
          mot cach gia tao.
        - p_value = ty le mau null >= te_observed (one-sided).
    """
    raise NotImplementedError


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 500,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Khoang tin cay bootstrap cho trung binh cua cac uoc luong TE.

    Args:
        values: cac gia tri TE tren tung cua so.
        n_bootstrap: so lan lay mau lai (>=500 theo roadmap).
        confidence: muc tin cay.
        seed: de tai lap.

    Returns:
        (ci_low, ci_high).

    TODO(ban tu code): lay mau co hoan lai n_bootstrap lan, moi lan tinh trung binh,
    roi lay percentile. Dung np.random.default_rng(seed).
    """
    raise NotImplementedError


def run_sanity_check(
    windows_forward: list,
    windows_backward: list,
    estimator,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> dict:
    """Chay day du sanity check va tra ve ket luan ro rang.

    Returns:
        dict gom ket qua bidirectional_te, khoang tin cay 2 chieu, va:
            "passed": bool - TE(tim->nao) > TE(nao->tim) VA khoang tin cay cua hieu
                khong chua 0 (tuc la khac biet co y nghia thong ke, khong chi la
                chenh lech ngau nhien).

    TODO(ban tu code):
        - Goi bidirectional_te, tinh CI cho ca 2 chieu VA cho hieu 2 chieu.
        - "passed" phai dua tren CI CUA HIEU, khong phai chi so sanh 2 trung binh -
          2 trung binh chenh nhau chut it ma CI chong lan nhau thi chua ket luan duoc.
        - In ket luan bang tieng Viet ro rang, KHONG chi tra ve so.
        - Neu khong dat: nhac lai thu tu kiem tra o docs/PHASE_S_GUIDE.md muc 3.4
          (dong bo hoa -> ectopic -> nhieu tim -> do dai cua so -> doi chieu KSG).
    """
    raise NotImplementedError
