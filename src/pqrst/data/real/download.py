import os
import time
import threading
from pathlib import Path
import wfdb
import urllib.request

def list_available_records(name: str) -> list[str]:
    """Danh sach ban ghi CO TREN PHYSIONET."""
    if name == 'capslpdb':
        try:
            return wfdb.get_record_list(name)
        except:
            return [f'n{i}' for i in range(1, 17)]
    return wfdb.get_record_list(name)

def list_local_records(db_dir: Path, name: str) -> list[str]:
    """Danh sach ban ghi DA TAI VE thanh cong."""
    db_dir = Path(db_dir)
    if not db_dir.exists():
        return []
    
    if name == 'capslpdb':
        return sorted(p.stem for p in db_dir.glob("*.edf"))
    
    names = sorted(p.stem for p in db_dir.glob("*.hea"))
    complete = []
    for nm in names:
        has_data = (db_dir / f"{nm}.dat").exists() or (db_dir / f"{nm}.edf").exists()
        has_other = (db_dir / f"{nm}.apn").exists() or (db_dir / f"{nm}.qrs").exists()
        if has_data or has_other:
            complete.append(nm)
    return complete

def _dir_size_mb(d: Path) -> float:
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6

def _progress_reporter(out_dir: Path, stop_event: threading.Event, interval_s: float = 15.0) -> None:
    last_mb = _dir_size_mb(out_dir)
    last_t = time.time()
    while not stop_event.wait(interval_s):
        now_mb = _dir_size_mb(out_dir)
        now_t = time.time()
        rate_kbs = (now_mb - last_mb) * 1000 / max(now_t - last_t, 1e-6)
        print(f"    ... van dang tai, hien {now_mb:.1f} MB trong thu muc (toc do ~{rate_kbs:.0f} KB/s)")
        last_mb, last_t = now_mb, now_t

def download_wfdb_database(name: str, records: list[str] | None = None, out_dir: str | Path | None = None, progress_interval_s: float = 15.0) -> dict:
    if out_dir is None:
        out_dir = Path.cwd() / "data" / "raw" / name
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_records = records if records is not None else list_available_records(name)
    already = set(list_local_records(out_dir, name))
    to_fetch = [r for r in target_records if r not in already]
    failed = []

    if to_fetch:
        print(f"Dang tai {len(to_fetch)}/{len(target_records)} ban ghi cua '{name}' vao {out_dir} ...")
        for idx, rec in enumerate(to_fetch, 1):
            print(f"[{idx}/{len(to_fetch)}] Dang tai '{rec}'...")
            t0 = time.time()
            stop_event = threading.Event()
            reporter = None
            if progress_interval_s > 0:
                reporter = threading.Thread(target=_progress_reporter, args=(out_dir, stop_event, progress_interval_s), daemon=True)
                reporter.start()
            
            try:
                if name == 'capslpdb':
                    url = f"https://physionet.org/files/capslpdb/1.0.0/{rec}.edf"
                    file_path = out_dir / f"{rec}.edf"
                    urllib.request.urlretrieve(url, str(file_path))
                else:
                    wfdb.dl_database(name, dl_dir=str(out_dir), records=[rec])
            except Exception as e:
                print(f"    LOI khi tai '{rec}': {e} - bo qua.")
                failed.append(rec)
            finally:
                stop_event.set()
                if reporter is not None:
                    reporter.join(timeout=1.0)
            print(f"    Xong '{rec}' sau {time.time()-t0:.1f}s.")
    else:
        print(f"'{name}': tat ca {len(target_records)} ban ghi yeu cau da co san, bo qua tai.")

    present_after = set(list_local_records(out_dir, name))
    missing = [r for r in target_records if r not in present_after]
    
    if missing:
        print(f"⚠️  CANH BAO: {len(missing)}/{len(target_records)} ban ghi KHONG co du file sau khi tai: {missing[:10]}")
    else:
        print(f"✅ Xac minh: du {len(target_records)} ban ghi da co day du file.")

    return {
        "requested": target_records,
        "already_present": sorted(already),
        "newly_downloaded": sorted(present_after - already),
        "missing": missing,
        "failed": failed,
    }
