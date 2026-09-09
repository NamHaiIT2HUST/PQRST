"""Dong bo hoa 2 kenh + cat cua so -> tai su dung dataclass Window cua Pha R.

Nho tra ve dung Window, toan bo ha tang danh gia da co (grid.py, cac estimator, sanity
check) chay duoc tren du lieu that ma KHONG phai sua gi - dung tinh than "hoan doi mot
khoi" cua roadmap.

*** DAY LA MODULE DE SAI NHAT PHA S ***
Roadmap goc ghi ro: "Neu sanity check khong dat, kiem tra pipeline dong bo hoa TRUOC
khi nghi ngo mo hinh."
"""

from __future__ import annotations

import numpy as np

from pqrst.data.synthetic.corpus import Window


def make_windows_from_synced_series(
    source: np.ndarray,
    target: np.ndarray,
    grid_fs: float,
    window_seconds: float,
    config_name: str,
    record_id: str,
    step_seconds: float | None = None,
) -> list[Window]:
    """Cat 2 chuoi DA DONG BO thanh cac Window de dua vao estimator.

    Args:
        source: chuoi nguon X (vd RR interval da noi suy len luoi).
        target: chuoi dich Y (vd EEG band power tren cung luoi). PHAI cung do dai
            va cung goc thoi gian voi source.
        grid_fs: tan so luoi (Hz), dung de doi window_seconds -> so mau.
        window_seconds: do dai cua so (giay). KHUYEN NGHI 30.0 - xem ly do o duoi.
        config_name: nhan cau hinh (vd "slpdb_rr_to_alpha").
        record_id: ma ban ghi, de truy nguoc cua so ve ban ghi goc.
        step_seconds: buoc truot. None = khong chong lan (= window_seconds).

    Returns:
        list[Window]. Luu y: te_ground_truth = NaN vi DU LIEU THAT KHONG CO DAP AN -
        day la khac biet can ban so voi Pha P/Q/R. Moi ham tinh bias/MSE se khong dung
        duoc; chi so sanh TUONG DOI giua 2 chieu va giua cac estimator moi co nghia.

    CHON window_seconds - noi truc tiep voi ket qua Pha R:
        Pha R da xac lap (on dinh qua 3 thi nghiem): T_phi thang KSG o N>=50, thua o
        N<30. Voi grid_fs=4Hz:
            30 giay -> N=120  -> vung T_phi manh  <-- KHUYEN NGHI
            60 giay -> N=240  -> vung T_phi manh
            10 giay -> N=40   -> ranh gioi, can than
        30 giay con dung bang 1 epoch cham giai doan giac ngu tieu chuan -> khop voi
        nhan co san cua slpdb/capslpdb.

    TODO(ban tu code):
        - n_samples = int(window_seconds * grid_fs)
        - assert len(source) == len(target) - neu lech la dau hieu dong bo hoa sai.
        - Voi moi cua so: y_t = target[1:], x_lag = source[:-1], y_lag = target[:-1]
          (dung y het quy uoc cua corpus.py de tuong thich hoan toan).
        - Bo cua so co NaN (doan du lieu mat/bi loai do artifact) - dem va bao cao
          so cua so bi bo.
        - params: luu grid_fs, window_seconds, record_id de truy nguoc duoc.
    """
    raise NotImplementedError


def verify_synchronization(
    source: np.ndarray,
    target: np.ndarray,
    grid_fs: float,
    estimator,
    shift_seconds: float = 10.0,
) -> dict:
    """*** TEST BAT BUOC: xac nhan pipeline that su do QUAN HE THOI GIAN ***

    Y tuong: dich nhan tao 1 kenh di vai giay -> quan he thoi gian bi pha vo -> TE
    phai GIAM RO. Neu TE khong doi khi dich, nghia la pipeline dang do mot thu gi do
    khac (vd chi la tuong quan tinh giua 2 phan phoi), va moi ket luan ve "huong ghep
    noi" deu vo nghia.

    Day la test re nhat de bat loi dong bo hoa - lam TRUOC khi chay toan bo phan tich.

    Args:
        source, target: 2 chuoi da dong bo.
        grid_fs: tan so luoi.
        estimator: bat ky BaseTEEstimator nao (nen dung KSG cho test nay - no khong
            phu thuoc checkpoint da train, nen loai duoc 1 bien so).
        shift_seconds: do dich nhan tao.

    Returns:
        dict gom "te_aligned", "te_shifted", "ratio" (= te_shifted/te_aligned),
        va "passed" (True neu te_shifted < te_aligned ro ret, vd ratio < 0.7).

    TODO(ban tu code):
        - Tinh TE tren cap (source, target) nhu binh thuong.
        - Dich source di shift_seconds*grid_fs mau, cat cho cung do dai, tinh lai TE.
        - Nen lap tren nhieu cua so roi lay trung binh - 1 cua so don le qua nhieu.
        - Neu KHONG passed: DUNG LAI, dieu tra dong bo hoa, dung chay tiep phan tich.
    """
    raise NotImplementedError
