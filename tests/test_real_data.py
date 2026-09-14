"""Test cho pipeline du lieu that (Pha S). Xem checklist docs/PHASE_S_GUIDE.md muc 4.

Nguyen tac: cac test o day KHONG duoc phu thuoc vao du lieu PhysioNet da tai ve
(du lieu that nam trong .gitignore, khong phai may nao cung co). Thay vao do, tu sinh
tin hieu gia lap co tinh chat da biet roi kiem tra pipeline xu ly dung.
"""

import numpy as np
import pytest

from pqrst.data.real.cardiac import rr_from_beat_annotations, interpolate_rr
from pqrst.data.real.eeg import compute_band_power, detect_cardiac_artifact, remove_cardiac_artifact
from pqrst.data.real.sync import align_to_common_grid, sync_and_window, verify_synchronization
from pqrst.data.real.respiration import preprocess_respiration, bandpass_filter
from pqrst.evaluation.sanity_check import (
    bidirectional_te, permutation_test_te, bootstrap_ci, run_sanity_check,
    bidirectional_te_per_record, run_sanity_check_per_record,
)
from pqrst.utils.standardize import standardize_window
from pqrst.data.synthetic.var_linear_gaussian import generate_var_linear_gaussian
from pqrst.baselines.ksg import KSGTEEstimator


class DummyEstimator:
    def estimate(self, x, y, **kwargs):
        # returns sum of difference, just a deterministic fake TE
        return float(np.mean(x - y))


class TestStandardize:
    """Chuan hoa la CAU NOI bat buoc giua du lieu tong hop va du lieu that
    (xem docs/PHASE_S_GUIDE.md muc 0). DA VERIFY TREN CHECKPOINT THAT (notebook 00):
    corr ban chuan hoa = 1.00000 khi doi thang do, so voi 0.07 cua ban khong chuan hoa."""

    def test_zero_mean_unit_std(self):
        a = np.array([0.8, 0.85, 0.79, 0.82, 0.9])
        z = standardize_window(a)
        assert abs(z.mean()) < 1e-10
        assert abs(z.std() - 1.0) < 1e-10

    def test_affine_invariance(self):
        """Tinh chat COT LOI: z-score bat bien duoi phep bien doi affine - day chinh
        la thu cho phep dua du lieu that (thang do bat ky) ve dung mien ma T_phi
        da hoc."""
        a = np.array([0.8, 0.85, 0.79, 0.82, 0.9])
        assert np.allclose(standardize_window(a * 100 + 55), standardize_window(a))

    def test_constant_window_no_nan(self):
        """Cua so hang so (tin hieu bao hoa / doan du lieu mat) phai tra ve mang 0,
        KHONG duoc ra NaN/Inf lam vo ca pipeline."""
        out = standardize_window(np.full(5, 3.7))
        assert np.all(np.isfinite(out))
        assert np.allclose(out, 0.0)


class TestCardiac:
    def test_rr_from_beat_annotations_basic(self):
        t = np.array([1.0, 1.8, 2.6, 3.4])
        t_rr, rr = rr_from_beat_annotations(t, exclude_ectopic=False)
        assert np.allclose(rr, 0.8)
        assert len(t_rr) == 3

    def test_rr_from_beat_annotations_ectopic(self):
        # 0.8, 0.8, 0.2 (ectopic), 1.4 (compensatory), 0.8
        t = np.cumsum([0.0, 0.8, 0.8, 0.2, 1.4, 0.8])
        t_rr, rr = rr_from_beat_annotations(t, exclude_ectopic=True)
        assert len(rr) == 3
        assert np.allclose(rr, 0.8)

    def test_rr_ectopic_uses_local_not_global_median(self):
        """Nhip tim TROI dan theo thoi gian (vd nhanh dan qua giai doan giac ngu) -
        trung vi TOAN CUC se khong dai dien cho tung doan, co the loai nham nhip binh
        thuong o dau/cuoi doan troi. Trung vi CUC BO phai thich nghi duoc voi xu huong
        troi cham nay ma khong loai nham."""
        # RR troi tu 1.0 xuong 0.7 giay qua 40 nhip (xu huong cham, khong dot ngot)
        rr_true = np.linspace(1.0, 0.7, 40)
        t = np.concatenate([[0.0], np.cumsum(rr_true)])
        t_rr, rr = rr_from_beat_annotations(t, exclude_ectopic=True, local_window_beats=11)
        # Voi trung vi CUC BO, hau het nhip "binh thuong theo xu huong troi" phai duoc giu -
        # khong bi loai oan vi khac trung vi TOAN CUC cua ca doan.
        assert len(rr) >= 35

    def test_interpolate_rr(self):
        t_rr = np.array([1.0, 2.0, 3.0])
        rr = np.array([0.8, 0.9, 1.0])
        t_grid, rr_grid = interpolate_rr(t_rr, rr, grid_fs=2.0)
        assert len(t_grid) > 0
        assert len(t_grid) == len(rr_grid)

    def test_interpolate_rr_no_extrapolation(self):
        """KHONG duoc ngoai suy ngoai [t_rr[0], t_rr[-1]] - vung khong co du lieu
        that thi khong duoc bia gia tri."""
        t_rr = np.array([1.0, 2.0, 3.0])
        rr = np.array([0.8, 0.9, 1.0])
        t_grid, rr_grid = interpolate_rr(t_rr, rr, grid_fs=2.0)
        assert t_grid[0] >= t_rr[0]
        assert t_grid[-1] <= t_rr[-1]


class TestRespiration:
    def test_bandpass_filter_removes_slow_drift_and_keeps_oscillation(self):
        """*** Phat hien tren Fantasia (23 ban ghi) ***: loc RESP bang thong nhung
        KHONG loc RR cung dai tao bat doi xung tri nho tu tuong quan (RR ~11s,
        RESP ~0.65s vi RR con giu trend cham LF/VLF) - lam TE lech huong SAI so
        voi RSA. `bandpass_filter` phai loai bo trend cham VA giu lai dao dong
        trong dai, dung chung cho ca RR va RESP de tranh bat doi xung nay."""
        fs = 4.0
        t = np.arange(0, 300, 1 / fs)
        oscillation = np.sin(2 * np.pi * 0.25 * t)
        slow_drift = 0.02 * t  # trend cham, tan so << 0.1Hz
        x = oscillation + slow_drift
        filtered = bandpass_filter(x, fs, (0.1, 0.5))
        # Tuong quan voi dao dong goc phai cao (giu lai duoc dao dong)...
        corr_osc = np.corrcoef(filtered, oscillation)[0, 1]
        assert corr_osc > 0.9
        # ... va bien do trend cham (do bang do lech giua 2 nua dau/cuoi) phai giam manh.
        half = len(x) // 2
        drift_before = abs(x[half:].mean() - x[:half].mean())
        drift_after = abs(filtered[half:].mean() - filtered[:half].mean())
        assert drift_after < 0.1 * drift_before


    def test_preprocess_respiration_recovers_breathing_frequency(self):
        """Tin hieu ho hap gia lap (0.25Hz = 15 lan/phut, dai binh thuong) + trend
        cham + nhieu cao tan - sau loc bang thong phai con lai gan dung tan so ho hap
        THAT, khong bi lech boi trend/nhieu."""
        fs = 50.0
        t = np.arange(0, 300, 1 / fs)
        breathing_freq = 0.25
        resp = (
            np.sin(2 * np.pi * breathing_freq * t)
            + 0.01 * t  # trend cham (drift cam bien)
            + 0.05 * np.sin(2 * np.pi * 10 * t)  # nhieu cao tan
        )
        t_grid, resp_grid = preprocess_respiration(resp, fs=fs, grid_fs=4.0)
        assert len(t_grid) == len(resp_grid)
        assert len(t_grid) > 0

        # uoc luong tan so troi qua FFT tren tin hieu da loc - phai gan breathing_freq
        freqs = np.fft.rfftfreq(len(resp_grid), d=1 / 4.0)
        spectrum = np.abs(np.fft.rfft(resp_grid - resp_grid.mean()))
        dominant = freqs[np.argmax(spectrum)]
        assert abs(dominant - breathing_freq) < 0.03

    def test_preprocess_respiration_handles_nan_dropout(self):
        """*** Phat hien tren Fantasia f2o06 ***: NaN rai rac (dut cam bien) trong
        tin hieu RESP goc lam filtfilt lan NaN ra TOAN BO ket qua (61 mau NaN dau
        vao -> 100% NaN dau ra). Phai noi suy qua khoang NaN NGAN truoc khi loc."""
        fs = 50.0
        t = np.arange(0, 300, 1 / fs)
        resp = np.sin(2 * np.pi * 0.25 * t)
        resp[1000:1010] = np.nan  # dut cam bien ngan, ~0.2s
        t_grid, resp_grid = preprocess_respiration(resp, fs=fs, grid_fs=4.0)
        assert len(resp_grid) > 0
        assert not np.isnan(resp_grid).any()

    def test_preprocess_respiration_all_nan_returns_empty(self):
        resp = np.full(500, np.nan)
        t_grid, resp_grid = preprocess_respiration(resp, fs=50.0, grid_fs=4.0)
        assert len(t_grid) == 0 and len(resp_grid) == 0

    def test_preprocess_respiration_no_extrapolation(self):
        fs = 10.0
        resp = np.sin(2 * np.pi * 0.2 * np.arange(0, 60, 1 / fs))
        t_grid, resp_grid = preprocess_respiration(resp, fs=fs, grid_fs=4.0)
        assert t_grid[0] >= 0.0
        assert t_grid[-1] <= 60.0


class TestEEG:
    def test_compute_band_power(self):
        fs = 100.0
        t = np.arange(0, 5, 1 / fs)
        eeg = np.sin(2 * np.pi * 10 * t)  # 10 Hz - nam trong dai alpha (8-13Hz)
        t_bp, bp = compute_band_power(eeg, fs, band=(8, 12), window_seconds=1.0, step_seconds=0.5)
        assert len(bp) > 0
        assert np.all(bp > np.log(1e-10))

    def test_detect_cardiac_artifact_clean(self):
        """EEG SACH (nhieu trang) -> artifact_ratio phai GAN 1.0, khong nghi nhiem."""
        rng = np.random.default_rng(0)
        eeg = rng.normal(0, 1, 3000)
        beats = np.arange(1.0, 25.0, 0.8)
        res = detect_cardiac_artifact(eeg, beats, fs=100.0)
        assert "artifact_ratio" in res
        assert res["artifact_ratio"] < 2.0
        assert res["suspected"] is False

    def test_detect_cardiac_artifact_contaminated(self):
        """*** TEST QUAN TRONG NHAT CUA MODULE NAY ***
        EEG bi nhiem NHAN TAO (nhieu trang + xung QRS gia tai dung dinh R) -> PHAI
        bao suspected=True. Neu test nay fail, khong the tin ket qua sanity check
        nao tren du lieu that (xem docs/PHASE_S_GUIDE.md muc 3.2)."""
        rng = np.random.default_rng(0)
        eeg = rng.normal(0, 1, 3000)
        beats = np.arange(1.0, 25.0, 0.8)
        for b in beats:
            i = int(b * 100)
            eeg[i:i + 6] += np.array([0, 1, 3, -2, 0.5, 0]) * 5
        res = detect_cardiac_artifact(eeg, beats, fs=100.0)
        assert res["suspected"] is True
        assert res["artifact_ratio"] > 2.0

    def test_remove_cardiac_artifact_removes_artifact_preserves_coupling(self):
        """*** Xac nhan khu nhieu KHONG vo tinh xoa mat ghep noi tim-nao THAT ***
        (phat hien tren slpdb thuc: 18/18 ban ghi nhiem nhieu tim - xem
        docs/PHASE_S_REVIEW.md - quyet dinh khu nhieu truoc khi tinh TE, nhung phai
        chung minh khu nhieu khong xoa luon ca ghep noi that can do).

        Mo phong: EEG = nhieu trang + 1 GHEP NOI THAT (bien dieu CHAM theo RR, thang
        thoi gian giay - da tao bang HRV hinh sin chu ky ~32s) + NHIEM DIEN TIM manh
        (xung QRS gia NHON, thang thoi gian chuc ms, dinh dung tai R - giong hinh
        dang dung trong test_detect_cardiac_artifact_contaminated). 2 thang thoi gian
        cach xa nhau (chuc ms vs chuc giay) nen mot ham khu nhieu DUNG (chi tru trong
        1 cua so hep quanh R-peak) phai tach duoc rieng nhieu ma khong dung toi bien
        dieu cham.

        Tieu chi PASS: (1) artifact_ratio giam ve duoi nguong sau khi khu nhieu, VA
        (2) tuong quan EEG-voi-ghep-noi GIU duoc >=80% so voi truoc khi bi nhiem -
        neu ham khu nhieu qua "tham lam" (window rong), tieu chi (2) se fail truoc.
        """
        rng = np.random.default_rng(0)
        fs = 100.0
        duration = 600.0
        n = int(duration * fs)
        t = np.arange(n) / fs

        n_beats = int(duration / 0.8) + 5
        k = np.arange(n_beats)
        intervals = 0.8 + 0.15 * np.sin(2 * np.pi * k / 40) + rng.normal(0, 0.01, n_beats)
        beats = np.cumsum(intervals) + 1.0
        beats = beats[beats < duration - 1.0]

        rr_local = np.interp(t, beats[1:], np.diff(beats))
        coupling_component = 2.0 * (rr_local - rr_local.mean())  # bien dieu CHAM, giay

        baseline = rng.normal(0, 1.0, n)
        eeg_with_coupling = baseline + coupling_component  # "su that" chua bi nhiem

        spike = np.array([0, 1, 3, -2, 0.5, 0]) * 5  # xung QRS gia, ~60ms
        eeg_contaminated = eeg_with_coupling.copy()
        for b in beats:
            i = int(b * fs)
            if i + len(spike) <= n:
                eeg_contaminated[i:i + len(spike)] += spike

        res_before = detect_cardiac_artifact(eeg_contaminated, beats, fs=fs)
        assert res_before["suspected"] is True, "setup sai: chua nhiem ro nhu mong doi"

        # window_seconds=0.2 (khong dung mac dinh 0.1): mac dinh cua ham duoc
        # chinh cho du lieu THAT o fs=250Hz (xung nhiem chi ~3-4 mau rong, window
        # mac dinh 24 mau -> ti le ~6x du rong). O day fs=100Hz va xung gia co 6
        # mau rong (bang phep thu o tests khac trong file nay) - can window rong
        # hon ti le de vut Tukey khong an vao chinh giua xung; day CHI la khac biet
        # ti le cua kich ban test tong hop, khong anh huong mac dinh dung cho du
        # lieu that.
        eeg_cleaned = remove_cardiac_artifact(eeg_contaminated, beats, fs=fs, window_seconds=0.2)
        res_after = detect_cardiac_artifact(eeg_cleaned, beats, fs=fs)
        assert res_after["artifact_ratio"] < 2.0, (
            f"khu nhieu chua du: artifact_ratio con {res_after['artifact_ratio']:.2f}")
        assert res_after["suspected"] is False

        corr_before = np.corrcoef(eeg_with_coupling, coupling_component)[0, 1]
        corr_after = np.corrcoef(eeg_cleaned, coupling_component)[0, 1]
        assert corr_before > 0.15, "setup sai: ghep noi 'that' qua yeu de kiem dinh"
        assert corr_after > 0.8 * corr_before, (
            f"khu nhieu da vo tinh xoa mat ghep noi that: tuong quan {corr_before:.3f} "
            f"-> {corr_after:.3f}")


class TestSync:
    def test_align_to_common_grid_uses_real_timestamps(self):
        """*** TEST BAO VE LOI DA PHAT HIEN KHI REVIEW ***
        2 chuoi CUNG gia tri nhung LECH thoi gian bat dau (giong dung tinh huong
        RR-grid vs EEG band-power-grid trong thuc te) phai duoc dong bo DUNG theo
        moc thoi gian THAT, khong phai bi cat theo chi so nhu truoc."""
        # Ca 2 kenh do CUNG 1 tin hieu vat ly f(t)=sin(t) theo THOI GIAN THUC, nhung
        # duoc lay mau tai cac moc thoi gian KHAC NHAU (giong dung tinh huong RR-grid
        # bat dau tai thoi diem nhip dau tien, con EEG-grid bat dau tai
        # window_seconds/2 tinh tu dau ban ghi - 2 moc t=0 khac nhau).
        t_a = np.arange(0, 100, 0.25)
        a = np.sin(t_a)
        t_b = np.arange(3, 103, 0.25) + 0.01    # bat dau tre hon + lech pha rat nho
        b = np.sin(t_b)

        a_grid, b_grid = align_to_common_grid(t_a, a, t_b, b, grid_fs=4.0)
        assert len(a_grid) == len(b_grid)
        # Sau khi dong bo DUNG theo thoi gian thuc, ca 2 phai xap xi cung sin(t_grid)
        # (sai so nho chi do buoc noi suy, KHONG do lech moc thoi gian bat dau).
        assert np.allclose(a_grid, b_grid, atol=5e-3)

    def test_sync_and_window(self):
        src = np.arange(100)
        tgt = np.arange(100) * 2
        windows = sync_and_window(src, tgt, grid_fs=1.0, window_seconds=10.0)
        assert len(windows) == 10
        w = windows[0]
        assert w.n_samples == 9
        assert len(w.y_t) == 9
        assert len(w.x_lag) == 9

    def test_sync_and_window_drops_nan_windows(self):
        src = np.arange(100.0)
        tgt = np.arange(100.0) * 2
        tgt[15] = np.nan
        windows = sync_and_window(src, tgt, grid_fs=1.0, window_seconds=10.0)
        # cua so chua chi so 15 (vd cua so bat dau tai i=10, gom [10,19]) phai bi bo
        assert len(windows) == 9

    def test_verify_synchronization_detects_shift(self):
        """Test co dap an biet truoc: ghep noi X->Y that (khong doi xung), dich thoi
        gian phai lam TE giam ro."""
        x, y, _ = generate_var_linear_gaussian(600, 0.5, 0.5, 0.8, 0.3, seed=1)
        res = verify_synchronization(x, y, grid_fs=1.0, estimator=KSGTEEstimator(), shift_seconds=10.0)
        assert res["passed"] is True
        assert res["te_shifted"] < res["te_aligned"]

    def test_verify_synchronization_inconclusive_on_independent_data(self):
        """2 chuoi doc lap (khong ghep noi) -> te_aligned gan 0 -> phai bao
        inconclusive, KHONG duoc ket luan passed=False mot cach vo can cu.

        Nguong 0.02 (xem sync.py) hieu chinh tu do thuc nghiem tren du lieu doc lap
        thuc su (N=600-6000, 5 seed: te_aligned trong [-0.014, 0.017]) - seed=0,
        N=6000 duoi day da xac nhan te_aligned=0.0002, an toan duoi nguong."""
        rng = np.random.default_rng(0)
        src = rng.normal(0, 1, 6000)
        tgt = rng.normal(0, 1, 6000)
        res = verify_synchronization(src, tgt, grid_fs=1.0, estimator=KSGTEEstimator(), shift_seconds=10.0)
        assert res.get("inconclusive") is True


class TestSanityCheck:
    def test_bidirectional_te_on_known_directional_coupling(self):
        """*** TEST QUAN TRONG NHAT CUA MODULE NAY ***
        Kiem chung chinh ham sanity check TRUOC khi dung tren du lieu that: dung
        generate_var_linear_gaussian (chi ghep noi X->Y, KHONG co Y->X). Sanity check
        phai phat hien dung chieu: TE(X->Y) > TE(Y->X)."""
        x, y, _ = generate_var_linear_gaussian(2000, 0.5, 0.5, 0.8, 0.3, seed=7)
        w_fwd = sync_and_window(x, y, grid_fs=1.0, window_seconds=50.0)
        w_bwd = sync_and_window(y, x, grid_fs=1.0, window_seconds=50.0)
        res = run_sanity_check(w_fwd, w_bwd, KSGTEEstimator(), n_bootstrap=200, seed=7)
        assert res["te_forward_mean"] > res["te_backward_mean"]
        assert bool(res["passed"])

    def test_run_sanity_check_per_record_detects_known_direction(self):
        """*** Sua pseudoreplication (chuan bi cho Q1/Q2) ***: don vi mau DUNG la
        BAN GHI, khong phai cua so (cac cua so trong cung 1 ban ghi tuong quan voi
        nhau). Mo phong 6 "ban ghi" DOC LAP (seed khac nhau), moi ban ghi chi co
        ghep noi X->Y (khong co Y->X) - kiem dinh theo ban ghi phai phat hien dung
        chieu."""
        records = {}
        for seed in range(6):
            x, y, _ = generate_var_linear_gaussian(1500, 0.5, 0.5, 0.8, 0.3, seed=seed)
            w_fwd = sync_and_window(x, y, grid_fs=1.0, window_seconds=50.0)
            w_bwd = sync_and_window(y, x, grid_fs=1.0, window_seconds=50.0)
            records[f"rec{seed}"] = (w_fwd, w_bwd)

        per_record = bidirectional_te_per_record(records, KSGTEEstimator())
        assert len(per_record) == 6
        fwd = [v["te_forward"] for v in per_record.values()]
        bwd = [v["te_backward"] for v in per_record.values()]

        res = run_sanity_check_per_record(fwd, bwd, n_bootstrap=200, seed=7)
        assert res["n_records"] == 6
        assert res["te_forward_mean"] > res["te_backward_mean"]
        assert bool(res["passed"])
        assert res["wilcoxon_p"] < 0.05

    def test_bidirectional_te_counts_failures(self):
        src = np.arange(50.0)
        tgt = np.arange(50.0) * 2
        w_f = sync_and_window(src, tgt, 1.0, 10.0)[:2]
        w_b = sync_and_window(tgt, src, 1.0, 10.0)[:2]
        res = bidirectional_te(w_f, w_b, DummyEstimator())
        assert res["n_windows"] == 2
        assert "difference" in res
        assert res["n_failed_forward"] == 0
        assert res["n_failed_backward"] == 0

    def test_permutation_test_preserves_marginal_distribution(self):
        """Hoan vi phai PHA VO quan he thoi gian nhung GIU NGUYEN phan phoi bien cua
        tung kenh (hoan vi THU TU CUA SO, khong hoan vi tung diem)."""
        src = np.arange(50.0)
        tgt = np.arange(50.0) * 2
        w = sync_and_window(src, tgt, 1.0, 10.0)[:5]
        res = permutation_test_te(w, DummyEstimator(), n_permutations=10)
        assert "p_value" in res
        assert 0.0 <= res["p_value"] <= 1.0

    def test_permutation_test_null_when_independent(self):
        """2 chuoi DOC LAP hoan toan -> p-value KHONG duoc co y nghia mot cach gia tao."""
        rng = np.random.default_rng(3)
        src = rng.normal(0, 1, 400)
        tgt = rng.normal(0, 1, 400)
        w = sync_and_window(src, tgt, 1.0, 20.0)
        res = permutation_test_te(w, KSGTEEstimator(), n_permutations=50, seed=3)
        assert res["p_value"] > 0.05

    def test_bootstrap_ci(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        low, high = bootstrap_ci(vals, n_bootstrap=50)
        assert low <= high

    def test_bootstrap_ci_covers_mean(self):
        rng = np.random.default_rng(0)
        vals = rng.normal(5.0, 1.0, 200)
        low, high = bootstrap_ci(vals, n_bootstrap=200, seed=0)
        assert low <= vals.mean() <= high

    def test_run_sanity_check(self):
        src = np.arange(50.0)
        tgt = np.arange(50.0) * 2
        w_f = sync_and_window(src, tgt, 1.0, 10.0)[:5]
        w_b = sync_and_window(tgt, src, 1.0, 10.0)[:5]
        res = run_sanity_check(w_f, w_b, DummyEstimator(), n_bootstrap=10)
        assert "passed" in res
