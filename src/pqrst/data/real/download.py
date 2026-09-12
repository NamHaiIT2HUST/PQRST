"""Tai du lieu tu PhysioNet.

SUA (review vong 1): ban truoc khong bat loi, khong xac minh sau khi tai - da quan sat
thuc te 1 lan "in ra thanh cong" nhung KHONG CO file nao tren dia. Da them idempotent +
xac minh that.

SUA (review vong 2 - "sao chay mai khong xong"): da do truc tiep toc do mang toi
PhysioNet ~56 KB/s (rat cham, khong phai loi code/thu vien) bang urllib doc thang
1 file .dat, khong qua wfdb. Voi toc do nay, tai ca database trong 1 lenh
wfdb.dl_database() se IM LANG (khong in gi) trong hang gio dong ho - nhin giong bi
treo dù no van dang chay dung. Gio: (1) tai TUNG BAN GHI mot thay vi goi 1 lan cho ca
danh sach, in tien do sau MOI ban ghi (khong phai doi het moi thay gi); (2) 1 luong
nen theo doi dung luong thu muc dich moi 15s de bao "van dang chay, da tai X MB" -
nguoi dung biet chac la con song, khong phai doan.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import wfdb


def list_available_records(name: str) -> list[str]:
    """Danh sach ban ghi CO TREN PHYSIONET (goi API online, khong phai da tai ve)."""
    return wfdb.get_record_list(name)


def list_local_records(db_dir: Path) -> list[str]:
    """Danh sach ban ghi DA TAI VE thanh cong (co du file .hea + du lieu) trong 1
    thu muc cuc bo. Dung ham nay de kiem tra thuc te, KHONG dung print() cua qua
    trinh tai de ket luan da thanh cong."""
    db_dir = Path(db_dir)
    if not db_dir.exists():
        return []
    names = sorted(p.stem for p in db_dir.glob("*.hea"))
    complete = []
    for name in names:
        has_data = (db_dir / f"{name}.dat").exists() or (db_dir / f"{name}.edf").exists()
        if has_data:
            complete.append(name)
    return complete


def _dir_size_mb(d: Path) -> float:
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6


def _progress_reporter(out_dir: Path, stop_event: threading.Event, interval_s: float = 15.0) -> None:
    """Chay trong 1 luong nen: cu moi interval_s giay, in dung luong thu muc dich +
    toc do trung binh ke tu lan in truoc - de nguoi dung biet CHAC la qua trinh van
    dang chay (khong phai treo), du wfdb.dl_database() khong tu in tien do theo byte."""
    last_mb = _dir_size_mb(out_dir)
    last_t = time.time()
    while not stop_event.wait(interval_s):
        now_mb = _dir_size_mb(out_dir)
        now_t = time.time()
        rate_kbs = (now_mb - last_mb) * 1000 / max(now_t - last_t, 1e-6)
        print(f"    ... van dang tai, hien {now_mb:.1f} MB trong thu muc "
              f"(toc do ~{rate_kbs:.0f} KB/s trong {interval_s:.0f}s vua qua)")
        last_mb, last_t = now_mb, now_t


def download_wfdb_database(
    name: str,
    records: list[str] | None = None,
    out_dir: str | Path | None = None,
    progress_interval_s: float = 15.0,
) -> dict:
    """Tai 1 bo (hoac 1 phan) tu PhysioNet, TUNG BAN GHI MOT, co bao tien do va
    xac minh thuc su sau khi tai.

    Args:
        name: ten bo tren PhysioNet (vd 'slpdb', 'fantasia', 'apnea-ecg', 'capslpdb').
        records: danh sach ban ghi can tai. None = tai het.
        out_dir: thu muc dich.
        progress_interval_s: chu ky in tien do (giay). Dat 0 de tat bao tien do (vd
            khi chay trong test tu dong).

    Returns:
        dict: {"requested": [...], "already_present": [...], "newly_downloaded": [...],
               "missing": [...], "failed": [...]}. "missing"/"failed" PHAI duoc kiem
        tra - dung tin vao viec ham nay chay het khong loi la du (mang cham co the
        khien 1 ban ghi giua chung loi ma cac ban ghi khac van tai binh thuong).

    Luu y toc do: da do thuc te toc do toi PhysioNet co the rat cham (~50-60 KB/s
    tuy mang) - 1 ban ghi vai chuc MB co the mat vai chuc phut. Day la gioi han mang,
    KHONG phai loi code. Neu qua cham, can nhac: thu vao gio khac (server/mang do tai
    thay doi theo gio), kiem tra VPN/proxy dang bat, hoac chi tai truoc slpdb (bo
    QUAN TRONG NHAT, 632MB) va de Fantasia/Apnea-ECG/CAP chay nen trong luc lam viec
    khac.
    """
    if out_dir is None:
        out_dir = Path.cwd() / "data" / "raw" / name
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_records = records if records is not None else list_available_records(name)

    already = set(list_local_records(out_dir))
    to_fetch = [r for r in target_records if r not in already]

    failed = []
    if to_fetch:
        print(f"Dang tai {len(to_fetch)}/{len(target_records)} ban ghi cua '{name}' "
              f"vao {out_dir} (da co san {len(already)} ban ghi).")
        print("Tai TUNG ban ghi mot de bao tien do ro rang - mang cham co the mat "
              "vai chuc phut MOI ban ghi, day la binh thuong, khong phai treo.")

        for idx, rec in enumerate(to_fetch, 1):
            print(f"[{idx}/{len(to_fetch)}] Dang tai '{rec}'...")
            t0 = time.time()

            stop_event = threading.Event()
            reporter = None
            if progress_interval_s > 0:
                reporter = threading.Thread(
                    target=_progress_reporter, args=(out_dir, stop_event, progress_interval_s),
                    daemon=True,
                )
                reporter.start()

            try:
                wfdb.dl_database(name, dl_dir=str(out_dir), records=[rec])
            except Exception as e:
                print(f"    LOI khi tai '{rec}': {type(e).__name__}: {e} - bo qua, tiep tuc ban ghi tiep theo.")
                failed.append(rec)
            finally:
                stop_event.set()
                if reporter is not None:
                    reporter.join(timeout=1.0)

            print(f"    Xong '{rec}' sau {time.time()-t0:.1f}s.")
    else:
        print(f"'{name}': tat ca {len(target_records)} ban ghi yeu cau da co san, bo qua tai.")

    present_after = set(list_local_records(out_dir))
    missing = [r for r in target_records if r not in present_after]

    result = {
        "requested": target_records,
        "already_present": sorted(already),
        "newly_downloaded": sorted(present_after - already),
        "missing": missing,
        "failed": failed,
    }

    if missing:
        print(f"⚠️  CANH BAO: {len(missing)}/{len(target_records)} ban ghi KHONG co du "
              f"file sau khi tai: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    else:
        print(f"✅ Xac minh: du {len(target_records)} ban ghi da co day du file.")

    return result
