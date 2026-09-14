from __future__ import annotations
import numpy as np

def bidirectional_te(windows_forward: list, windows_backward: list, estimator) -> dict:
    """SUA (review): ban truoc dung `except: pass` nuot loi im lang, khong dem - neu
    nhieu cua so loi se am tham cho ra ket qua tren mau con lai rat nho ma khong ai
    biet. Gio dem so loi va BAO RO trong ket qua tra ve."""
    te_f = []
    te_b = []
    n_failed_f = 0
    n_failed_b = 0
    for w_f, w_b in zip(windows_forward, windows_backward):
        x_f = np.empty(w_f.n_samples + 1); y_f = np.empty(w_f.n_samples + 1)
        x_f[:-1] = w_f.x_lag; x_f[-1] = w_f.x_lag[-1]
        y_f[:-1] = w_f.y_lag; y_f[-1] = w_f.y_t[-1]
        try:
            te_f.append(estimator.estimate(x_f, y_f))
        except Exception:
            n_failed_f += 1

        x_b = np.empty(w_b.n_samples + 1); y_b = np.empty(w_b.n_samples + 1)
        x_b[:-1] = w_b.x_lag; x_b[-1] = w_b.x_lag[-1]
        y_b[:-1] = w_b.y_lag; y_b[-1] = w_b.y_t[-1]
        try:
            te_b.append(estimator.estimate(x_b, y_b))
        except Exception:
            n_failed_b += 1

    if n_failed_f or n_failed_b:
        print(f"CANH BAO bidirectional_te: {n_failed_f}/{len(windows_forward)} cua so "
              f"chieu forward va {n_failed_b}/{len(windows_backward)} cua so chieu "
              f"backward loi khi uoc luong - da bo qua, KHONG tinh vao trung binh.")

    return {
        "te_forward_mean": np.mean(te_f),
        "te_backward_mean": np.mean(te_b),
        "te_forward_values": te_f,
        "te_backward_values": te_b,
        "difference": np.mean(te_f) - np.mean(te_b),
        "n_windows": len(te_f),
        "n_failed_forward": n_failed_f,
        "n_failed_backward": n_failed_b,
    }

def bidirectional_te_per_record(
    records: dict[str, tuple[list, list]], estimator
) -> dict[str, dict]:
    """Tinh TE trung binh MOI CHIEU cho TUNG ban ghi rieng (khong gop cua so giua
    cac ban ghi) - dung lam DAU VAO cho don vi thong ke DUNG (xem
    run_sanity_check_per_record).

    Args:
        records: {record_id: (windows_forward, windows_backward)}.

    Returns:
        {record_id: {"te_forward":.., "te_backward":.., "n_windows":..}} - 1 gia tri
        TRUNG BINH cho MOI ban ghi.
    """
    out = {}
    for rid, (w_fwd, w_bwd) in records.items():
        res = bidirectional_te(w_fwd, w_bwd, estimator)
        out[rid] = {
            "te_forward": res["te_forward_mean"],
            "te_backward": res["te_backward_mean"],
            "n_windows": res["n_windows"],
        }
    return out


def run_sanity_check_per_record(
    te_forward_by_record, te_backward_by_record, n_bootstrap: int = 500, seed: int = 42
) -> dict:
    """Sanity check dung DON VI BAN GHI (khong phai cua so) lam mau thong ke.

    SUA (review - phat hien khi chuan bi cho Q1/Q2): `run_sanity_check` gop TAT CA
    cua so cua TAT CA ban ghi thanh 1 tap roi bootstrap/kiem dinh tren do - day la
    **gia lap so mau (pseudoreplication)**, mot loi thong ke kinh dien: cac cua so
    trong CUNG 1 ban ghi tuong quan voi nhau (cung 1 nguoi, cung 1 dem ghi), khong
    doc lap. Coi 17 ban ghi x ~240 cua so la "N=4049 mau doc lap" la SAI gia dinh -
    CI bootstrap tren "N=4049" phan anh SAI do khong chac chan thuc su (co the hep
    HON hoac RONG hon CI dung tuy muc do khac biet giua cac ban ghi trong du lieu cu
    the - khong co chieu co dinh, da tu kiem tra bang mo phong). Van de goc KHONG
    phai "CI hep hon" ma la GIA DINH DOC LAP SAI cho don vi mau - day la loai loi
    reviewer thong ke/sinh ly hoc se bat ngay trong 1 bai Q1/Q2.

    Ham nay lay 1 gia tri TE TRUNG BINH cho MOI ban ghi (tu bidirectional_te_per_record)
    lam don vi mau, roi bootstrap/kiem dinh tren N=so_ban_ghi - phan anh dung do
    khong chac chan thuc su. Them kiem dinh Wilcoxon signed-rank (phi tham so, phu
    hop N nho, khong gia dinh phan phoi chuan) nhu 1 kiem dinh doc lap thu 2.

    Args:
        te_forward_by_record, te_backward_by_record: 1 gia tri TE trung binh CHO
            MOI ban ghi (dai = so ban ghi, KHONG PHAI so cua so).
    """
    from scipy import stats

    fwd = np.asarray(te_forward_by_record, dtype=float)
    bwd = np.asarray(te_backward_by_record, dtype=float)
    assert len(fwd) == len(bwd), "phai co cung so ban ghi cho 2 chieu"
    diff = fwd - bwd

    ci_fwd = bootstrap_ci(fwd, n_bootstrap, seed=seed)
    ci_bwd = bootstrap_ci(bwd, n_bootstrap, seed=seed)
    ci_diff = bootstrap_ci(diff, n_bootstrap, seed=seed)

    try:
        w_stat, w_p = stats.wilcoxon(fwd, bwd, alternative="greater")
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")

    passed = bool(fwd.mean() > bwd.mean() and ci_diff[0] > 0)

    if passed:
        print(f"Sanity check (theo ban ghi, N={len(fwd)}) PASSED: TE forward > TE "
              f"backward ro ret, CI hieu so khong chua 0, Wilcoxon p={w_p:.4f}.")
    else:
        print(f"Sanity check (theo ban ghi, N={len(fwd)}) FAILED.")

    return {
        "n_records": len(fwd),
        "te_forward_mean": float(fwd.mean()),
        "te_backward_mean": float(bwd.mean()),
        "te_forward_by_record": fwd,
        "te_backward_by_record": bwd,
        "difference": float(diff.mean()),
        "ci_forward": ci_fwd,
        "ci_backward": ci_bwd,
        "ci_difference": ci_diff,
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p": float(w_p),
        "passed": passed,
    }


def permutation_test_te(windows: list, estimator, n_permutations: int = 500, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    
    te_obs = []
    x_list = []; y_list = []
    for w in windows:
        x = np.zeros(w.n_samples + 1); y = np.zeros(w.n_samples + 1)
        x[:-1] = w.x_lag; y[:-1] = w.y_lag; y[1:] = w.y_t
        te_obs.append(estimator.estimate(x, y))
        x_list.append(x)
        y_list.append(y)
        
    te_obs_mean = np.mean(te_obs)
    
    null_dist = []
    for _ in range(n_permutations):
        perm = rng.permutation(len(windows))
        te_null = []
        for i, j in enumerate(perm):
            if i == j: j = (j + 1) % len(windows)
            te_null.append(estimator.estimate(x_list[j], y_list[i]))
        null_dist.append(np.mean(te_null))
        
    p_value = np.mean(np.array(null_dist) >= te_obs_mean)
    
    return {
        "te_observed": te_obs_mean,
        "null_distribution": null_dist,
        "p_value": p_value,
        "significant": p_value < 0.05
    }

def bootstrap_ci(values: np.ndarray, n_bootstrap: int = 500, confidence: float = 0.95, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))
    
    alpha = 1.0 - confidence
    return np.percentile(means, alpha / 2 * 100), np.percentile(means, (1 - alpha / 2) * 100)

def run_sanity_check(windows_forward: list, windows_backward: list, estimator, n_bootstrap: int = 500, seed: int = 42) -> dict:
    res = bidirectional_te(windows_forward, windows_backward, estimator)
    ci_f = bootstrap_ci(res["te_forward_values"], n_bootstrap, seed=seed)
    ci_b = bootstrap_ci(res["te_backward_values"], n_bootstrap, seed=seed)
    
    diff = np.array(res["te_forward_values"]) - np.array(res["te_backward_values"])
    ci_diff = bootstrap_ci(diff, n_bootstrap, seed=seed)
    
    passed = (res["te_forward_mean"] > res["te_backward_mean"]) and (ci_diff[0] > 0)
    
    if passed:
        print("Sanity check PASSED: TE(tim->nao) lon hon ro ret so voi TE(nao->tim).")
    else:
        print("Sanity check FAILED. Vui long kiem tra:")
        print("1. Dong bo hoa thoi gian\n2. Loc ectopic\n3. Nhieu tim\n4. Do dai cua so\n5. Doi chieu KSG")
        
    res.update({
        "ci_forward": ci_f,
        "ci_backward": ci_b,
        "ci_difference": ci_diff,
        "passed": passed
    })
    return res
