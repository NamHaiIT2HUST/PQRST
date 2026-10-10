"""Preprocess all 40 Fantasia records (20 Young, 20 Old).

Generates fwd and bwd window corpuses for all 40 subjects and compiles
clinical demographics and signal quality metrics into a single table.
"""

from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb
import yaml

from pqrst.data.real.cardiac import rr_from_beat_annotations, interpolate_rr
from pqrst.data.real.respiration import bandpass_filter, preprocess_respiration
from pqrst.data.real.sync import align_to_common_grid, sync_and_window
from pqrst.data.synthetic.corpus import save_corpus

BASE = Path(__file__).resolve().parent.parent
RAW_DIR = BASE / "data" / "raw" / "fantasia"
PROC_DIR = BASE / "data" / "processed" / "fantasia"
OUT_TABLE = BASE / "results" / "tables" / "fantasia_40_demographics.csv"


def get_record_demographics(record_id: str) -> dict:
    hea_file = RAW_DIR / f"{record_id}.hea"
    with open(hea_file, "r", encoding="utf-8") as f:
        content = f.read()
    match = re.search(r"#\s*Age:\s*(\d+)\s*Sex:\s*([MF])", content)
    age = int(match.group(1)) if match else None
    sex = match.group(2) if match else None
    group = "Old" if "o" in record_id else "Young"
    return {"age": age, "sex": sex, "group": group}


def process_single_record(record_id: str, cfg: dict) -> dict:
    demo = get_record_demographics(record_id)
    rec_path = RAW_DIR / record_id
    ann = wfdb.rdann(str(rec_path), "ecg")
    beat_times_raw = ann.sample / 250.0
    is_normal = np.array([s == "N" for s in ann.symbol])
    beat_times = beat_times_raw[is_normal]

    t_rr, rr = rr_from_beat_annotations(
        beat_times,
        exclude_ectopic=True,
        threshold_ratio=cfg["cardiac"]["ectopic_threshold_ratio"],
    )
    removed_frac = 1.0 - len(rr) / max(len(beat_times) - 1, 1)

    t_grid_rr, rr_grid = interpolate_rr(t_rr, rr, grid_fs=cfg["grid_fs"])
    rr_grid_filtered = bandpass_filter(
        rr_grid, cfg["grid_fs"], tuple(cfg["respiration"]["bandpass_hz"])
    )

    record = wfdb.rdrecord(str(rec_path))
    resp_idx = record.sig_name.index("RESP")
    resp_sig = record.p_signal[:, resp_idx]
    t_resp, resp_grid = preprocess_respiration(
        resp_sig,
        fs=record.fs,
        grid_fs=cfg["grid_fs"],
        bandpass=tuple(cfg["respiration"]["bandpass_hz"]),
    )

    rr_aligned, resp_aligned = align_to_common_grid(
        t_grid_rr, rr_grid_filtered, t_resp, resp_grid, grid_fs=cfg["grid_fs"]
    )

    # Naming convention in Phase S:
    # fwd = rr_to_resp (source: RR, target: Resp)
    # bwd = resp_to_rr (source: Resp, target: RR)
    windows_fwd = sync_and_window(
        rr_aligned,
        resp_aligned,
        grid_fs=cfg["grid_fs"],
        window_seconds=cfg["window_seconds"],
        record_id=record_id,
        config_name="rr_to_resp",
    )
    windows_bwd = sync_and_window(
        resp_aligned,
        rr_aligned,
        grid_fs=cfg["grid_fs"],
        window_seconds=cfg["window_seconds"],
        record_id=record_id,
        config_name="resp_to_rr",
    )

    fwd_path = PROC_DIR / f"{record_id}_fwd.npz"
    bwd_path = PROC_DIR / f"{record_id}_bwd.npz"

    # Always ensure both files exist
    save_corpus(windows_fwd, str(fwd_path))
    save_corpus(windows_bwd, str(bwd_path))

    return {
        "record_id": record_id,
        "group": demo["group"],
        "age": demo["age"],
        "sex": demo["sex"],
        "total_beats": len(beat_times_raw),
        "normal_beats": len(beat_times),
        "ectopic_fraction": round(float(removed_frac), 4),
        "aligned_samples": len(rr_aligned),
        "duration_min": round(len(rr_aligned) / cfg["grid_fs"] / 60.0, 1),
        "n_windows": len(windows_fwd),
    }


def main():
    with open(BASE / "configs" / "real" / "preprocessing.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)

    records = [
        f.stem
        for f in sorted(RAW_DIR.glob("*.hea"))
    ]
    print(f"Processing all {len(records)} Fantasia records...")

    results = []
    for rid in records:
        info = process_single_record(rid, cfg)
        print(f"  [{rid}] {info['group']} Age={info['age']} Sex={info['sex']} Windows={info['n_windows']}")
        results.append(info)

    df = pd.DataFrame(results)
    df.to_csv(OUT_TABLE, index=False)
    print(f"\nSaved all 40 demographics & metadata to: {OUT_TABLE}")
    print("\nSummary by Group:")
    print(df.groupby("group")[["age", "duration_min", "n_windows"]].agg(["count", "mean", "std"]))


if __name__ == "__main__":
    main()
