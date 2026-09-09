"""Tai 4 bo du lieu sinh ly that tu PhysioNet.

THONG TIN DA XAC MINH truc tiep tren trang PhysioNet (khong dua vao tri nho) -
xem bang day du trong docs/PHASE_S_GUIDE.md muc 1:

    slpdb      (MIT-BIH Polysomnographic): ECG + EEG + ho hap, 18 ban ghi, 632 MB, WFDB
    capslpdb   (CAP Sleep):                EEG + ECG + ..., 108 ban ghi, 40.1 GB, EDF
    fantasia:                              ECG + ho hap + huyet ap, 40 nguoi, 293 MB, WFDB
    apnea-ecg:                             ECG (8/70 ban ghi co them ho hap), 581 MB, WFDB

*** CANH BAO QUAN TRONG NHAT ***
Fantasia va Apnea-ECG KHONG CO EEG. Khong the dung 2 bo nay cho luan diem tim-nao.
Dung mat thoi gian tim kenh EEG trong 2 bo do - no khong ton tai. Vai tro that cua
tung bo xem docs/PHASE_S_GUIDE.md muc 1.

CAP Sleep nang 40 GB nhung KHONG CAN tai het - wfdb cho phep tai tung ban ghi rieng.
Khuyen nghi: chi tai 16 ban ghi nguoi khoe manh n1-n16 (doi chung sach nhat).
"""

from __future__ import annotations

from pathlib import Path


def download_wfdb_database(
    db_name: str,
    dest_dir: Path,
    records: list[str] | None = None,
) -> Path:
    """Tai 1 bo du lieu (hoac 1 phan) tu PhysioNet ve dia.

    Args:
        db_name: ten bo tren PhysioNet, vd "slpdb", "fantasia", "apnea-ecg", "capslpdb".
        dest_dir: thu muc dich (nen la data/raw/physionet/<db_name>).
        records: danh sach ban ghi cu the can tai. None = tai toan bo.
            VOI capslpdb PHAI truyen danh sach (vd ["n1","n2",...,"n16"]) - tai het
            la 40 GB, khong can thiet cho sanity check.

    Returns:
        Duong dan thu muc chua du lieu da tai.

    TODO(ban tu code):
        - Dung wfdb.dl_database(db_name, dl_dir, records=...) cho WFDB.
        - KIEM TRA TRUOC KHI TAI: neu thu muc da co du file thi bo qua (idempotent) -
          tranh tai lai hang GB moi lan chay lai notebook.
        - In ra dung luong da tai va so ban ghi de nguoi dung theo doi.
        - Bat loi mang va bao ro rang (tai 40GB de dut giua chung).
    """
    raise NotImplementedError


def list_available_records(db_dir: Path) -> list[str]:
    """Liet ke cac ban ghi da tai ve trong 1 thu muc.

    TODO(ban tu code): quet file .hea (WFDB) hoac .edf (CAP), tra ve ten ban ghi
    (khong ke duoi mo rong), da sap xep.
    """
    raise NotImplementedError
