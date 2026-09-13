from __future__ import annotations
import numpy as np

EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

def compute_band_power(eeg: np.ndarray, fs: float, band: tuple[float, float], window_seconds: float = 2.0, step_seconds: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    from scipy.signal import welch
    from scipy.integrate import trapezoid  # SUA: np.trapz bi go khoi numpy 2.x (AttributeError)
    n_samples = len(eeg)
    window_samples = int(window_seconds * fs)
    step_samples = int(step_seconds * fs)

    times = []
    powers = []

    for i in range(0, n_samples - window_samples + 1, step_samples):
        segment = eeg[i:i+window_samples]
        f, Pxx = welch(segment, fs, nperseg=window_samples)
        idx = np.logical_and(f >= band[0], f <= band[1])
        power = trapezoid(Pxx[idx], f[idx])
        powers.append(np.log(power + 1e-12))  # log power
        times.append((i + window_samples/2) / fs)

    return np.array(times), np.array(powers)

def _locked_average(eeg: np.ndarray, lock_times_samples: np.ndarray, half_win: int) -> np.ndarray | None:
    segments = [
        eeg[i - half_win : i + half_win]
        for i in lock_times_samples
        if i - half_win >= 0 and i + half_win < len(eeg)
    ]
    if not segments:
        return None
    return np.mean(segments, axis=0)


def remove_cardiac_artifact(
    eeg: np.ndarray,
    beat_times: np.ndarray,
    fs: float,
    window_seconds: float = 0.1,
    max_jitter_seconds: float = 0.02,
) -> np.ndarray:
    """Khu nhieu dien tim bang tru template khoa-theo-R-peak (phuong phap Woody:
    can chinh jitter truoc, roi tru scale rieng cho tung nhip).

    QUAN TRONG - vi sao window_seconds MAC DINH RAT HEP (100ms, so voi 1000ms cua
    detect_cardiac_artifact): da do thuc nghiem tren slp01a (xem PHASE_S_REVIEW.md)
    rang do rong THAT cua xung nhiem chi ~12-16ms, dinh dong thoi voi QRS cua ECG
    (khong phai heartbeat-evoked potential, vi HEP dinh muon ~200-300ms va rong hon
    nhieu). Dung window RONG de tru se xoa luon ca EEG THAT xung quanh (vd tin hieu
    ghep noi tim-nao dang can do) - day chinh la nguy co "vo tinh xoa mat ghep noi
    that" khi khu nhieu. Window HEP (100ms, ~25 mau o 250Hz) chi vua du de trum kin
    xung nhon do duoc, khong dung toi cac dao dong EEG cham hon (giay) noi mang
    thong tin ghep noi tim-nao thuc su.

    SUA (review - chay thuc te tren 18/18 ban ghi slpdb): ban dau CHI tru template
    tai dung vi tri annotation, scale rieng tung nhip, KHONG can chinh jitter - ket
    qua THAT SU KHONG ON DINH: nhieu ban ghi giam manh (vd slp45: 20.8->3.9) nhung
    vai ban ghi lai TANG nhiem sau khi "khu" (vd slp32: 15.4->48.4). Them Woody's
    method (can chinh jitter +-max_jitter mau bang tuong quan chuan hoa TRUOC khi
    xay lai template) KHONG sua duoc - van tang o dung nhung ban ghi do (slp32:
    15.4->53.0). Dieu tra sau: in truc tiep gia tri template cua cac ban ghi "tang"
    thi thay KHONG phai xung nhon nhu slp01a/slp45, ma la mot muc GAN-HANG-SO
    (edge_offset/std ~6-10, so voi ~0.2-0.6 o cac ban ghi giam duoc) - template
    KHONG ve gan 0 o 2 dau cua so. Tru truc tiep 1 template nhu vay (cua so hinh
    chu nhat, khong vut) tao ra BUOC NHAY (discontinuity) o bien +-window_seconds/2,
    va buoc nhay nay LAI dung dan theo nhip -> chinh no bi phat hien nhu 1 nhiem moi,
    doi khi manh hon nhiem goc. Fix dung: VUT (window) template bang ham Hann truoc
    khi tru - Hann bang dung 0 tai 2 dau cua so, dam bao KHONG BAO GIO tao buoc
    nhay bat ke template co "gan hang so" hay khong, chi tru manh nhat o giua cua
    so (dung noi xung QRS tap trung).

    Cac buoc: (1) xay template THO tu vi tri annotation goc, vut Tukey (alpha=0.5:
    50% giua GIU NGUYEN bien do - dung noi xung QRS tap trung, 25% moi ben vut dan
    ve dung 0 - khong bao gio tao buoc nhay bat ke template co dang gi), (2) voi
    moi nhip, tim do lech +-max_jitter mau khop tot nhat (tuong quan chuan hoa) voi
    template tho, (3) XAY LAI template tu cac vi tri DA can chinh, vut Tukey lai,
    (4) tru template da vut nay tai vi tri da can chinh, scale rieng tung nhip.

    Da verify bang mo phong (xem tests/test_real_data.py
    test_remove_cardiac_artifact_removes_artifact_preserves_coupling): tren du lieu
    tong hop co CA nhiem QRS manh VA 1 ghep noi tim-nao THAT (bien dieu EEG cham theo
    RR), ham nay xoa duoc nhiem (artifact_ratio ve gan muc nen) MA VAN GIU duoc tuong
    quan ghep noi that gan nhu khong doi.
    """
    half_win = int((window_seconds / 2) * fs)
    max_jitter = max(int(max_jitter_seconds * fs), 0)
    beat_samples = (beat_times * fs).astype(int)
    from scipy.signal.windows import tukey
    # alpha=0.5: 50% giua GIU NGUYEN bien do (dung noi xung QRS tap trung), 25% moi
    # ben vut ve dung 0 (khong bao gio tao buoc nhay). Hann (vut toan bo, khong co
    # doan giua giu nguyen) qua manh, lam yeu ca phan giua thuc su can tru - da thu
    # va lam fail test tren du lieu tong hop (khu khong het xung QRS gia manh).
    taper = tukey(2 * half_win, alpha=0.5)

    coarse_template = _locked_average(eeg, beat_samples, half_win)
    if coarse_template is None:
        return eeg.copy()
    coarse_template = coarse_template * taper
    coarse_norm = float(np.linalg.norm(coarse_template))
    if coarse_norm <= 0:
        return eeg.copy()

    aligned_positions = []
    for i in beat_samples:
        lo, hi = i - half_win - max_jitter, i + half_win + max_jitter
        if lo < 0 or hi > len(eeg):
            continue
        best_j, best_corr = 0, -np.inf
        for j in range(-max_jitter, max_jitter + 1):
            seg = eeg[i + j - half_win : i + j + half_win] * taper
            seg_norm = float(np.linalg.norm(seg))
            corr = float(np.dot(seg, coarse_template)) / (seg_norm * coarse_norm + 1e-12)
            if corr > best_corr:
                best_corr, best_j = corr, j
        aligned_positions.append(i + best_j)

    if not aligned_positions:
        return eeg.copy()
    aligned_positions = np.array(aligned_positions)
    refined_template = _locked_average(eeg, aligned_positions, half_win)
    if refined_template is None:
        return eeg.copy()
    refined_template = refined_template * taper

    cleaned = eeg.copy()
    denom = float(np.dot(refined_template, refined_template))
    if denom <= 0:
        return cleaned
    for i in aligned_positions:
        lo, hi = i - half_win, i + half_win
        seg = eeg[lo:hi]
        scale = float(np.dot(seg, refined_template) / denom)
        cleaned[lo:hi] = seg - scale * refined_template
    return cleaned


def remove_cardiac_artifact_regression(
    eeg: np.ndarray,
    ecg: np.ndarray,
    fs: float,
    max_lag_seconds: float = 0.02,
) -> np.ndarray:
    """Khu nhieu dien tim bang HOI QUY TUYEN TINH EEG theo chinh dang song ECG THAT
    (khong phai 1 template trung binh co dinh).

    SUA (review - `remove_cardiac_artifact` bang template trung binh KHONG DU): da
    thu 3 lan cai tien (template tho -> can chinh jitter Woody -> vut Tukey chong
    buoc nhay), van khong dua BAT KY ban ghi nao trong 18 ban ghi slpdb ve duoi
    nguong (thap nhat ~5.9, nguong 2.0), va 6/18 ban ghi con TE HON. Nguyen nhan:
    slpdb chi co 1 kenh EEG nen KHONG the dung ICA (phuong phap manh nhat, can
    nhieu kenh doc lap); va bien do/hinh dang nhiem THAT co the khac nhau giua cac
    nhip (do ho hap, chuyen dong, tro khang da thay doi...) nhieu hon mot template
    CO DINH (du da can chinh jitter+vut) co the mo ta.

    Y tuong khac han: KHONG gia dinh 1 hinh dang co dinh. Dung TRUC TIEP gia tri
    ECG THAT tai moi thoi diem (va vai mau lan can, +-max_lag_seconds) lam BIEN HOI
    QUY de du doan phan EEG bi nhiem, qua 1 phep hoi quy tuyen tinh (FIR ngan) fit
    TREN TOAN BO ban ghi cung luc:

        EEG_nhiem_du_doan(t) = sum_{k=-K}^{K} w_k * ECG(t + k)

    roi tru phan du doan nay. Vi ECG(t) tai moi thoi diem PHAN ANH DUNG bien do/hinh
    dang thuc te cua tung nhip (khong phai trung binh ca ban ghi), hoi quy nay tu
    dieu chinh theo tung nhip ma khong can gia dinh hinh dang co dinh.

    Vi sao AN TOAN cho ghep noi tim-nao THAT: bo loc CHI dai +-max_lag_seconds
    (mac dinh 20ms, ~5 mau o 250Hz) - qua ngan de mo hinh hoa bat ky dao dong CHAM
    (giay) nao. Va ECG(t) gan nhu bang 0 giua 2 nhip (ngoai doan QRS), nen hoi quy
    CHI co the anh huong tai chinh cac thoi diem QRS - khong dung duoc toi cac
    doan EEG xa nhip tim, noi ghep noi tim-nao (qua cong suat dai tan, thang thoi
    gian giay) thuc su nam.

    Da verify bang mo phong (xem tests/test_real_data.py
    test_remove_cardiac_artifact_regression_removes_artifact_preserves_coupling).
    """
    assert len(eeg) == len(ecg), "eeg va ecg phai cung do dai (cung luoi mau)"
    max_lag = max(int(max_lag_seconds * fs), 1)
    n = len(eeg)
    lo, hi = max_lag, n - max_lag
    if hi <= lo:
        return eeg.copy()

    lags = range(-max_lag, max_lag + 1)
    X = np.column_stack([ecg[lo + lag : hi + lag] for lag in lags])
    y = eeg[lo:hi]
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ w

    cleaned = eeg.copy()
    cleaned[lo:hi] = y - pred
    return cleaned


def remove_cardiac_artifact_ica(
    raw,
    eeg_channels: list[str],
    ecg_channel: str,
    n_components: int | float | None = None,
    rng: int = 42,
):
    """Khu nhieu tim khoi EEG bang ICA (mne.preprocessing.ICA + find_bads_ecg) -
    PHUONG PHAP CHUAN, cong bo rong rai trong tai lieu EEG, khac han 4 lan thu tren
    slpdb (template trung binh + Woody jitter + Tukey taper + hoi quy ECG toan cuc -
    xem docstring remove_cardiac_artifact/remove_cardiac_artifact_regression) deu
    KHONG DU vi slpdb chi co 1 kenh EEG (ICA can nhieu kenh DOC LAP de tach nguon).

    capslpdb co 9 kenh EEG differential dong thoi -> du de ICA tach 1 (hoac vai)
    thanh phan doc lap tuong ung voi nguon dien tim, roi loai bo CHINH thanh phan do
    (khong dung template/hoi quy mo hinh hoa nhiem tren TUNG kenh rieng le).

    Args:
        raw: mne.io.Raw (da preload) chua CA kenh EEG va kenh ECG.
        eeg_channels: ten cac kenh EEG dung de fit ICA (vd 9 kenh differential cua
            capslpdb).
        ecg_channel: ten kenh ECG dung de tim thanh phan ICA tuong quan voi tim
            (find_bads_ecg tu do QRS tren kenh nay, KHONG can beat_times truyen vao).
        n_components: so thanh phan ICA (None = giu het, mne tu chon theo rank).

    Returns:
        (cleaned_raw, ecg_component_indices, scores): cleaned_raw la mne.io.Raw
        CHI gom eeg_channels, da loai bo thanh phan lien quan tim. ecg_component_indices
        la danh sach index thanh phan ICA bi loai (rong = khong tim thay thanh phan
        nao tuong quan du manh - CAN kiem tra lai bang detect_cardiac_artifact, KHONG
        duoc mac dinh tin la da khu duoc).
    """
    import mne

    eeg_raw = raw.copy().pick(eeg_channels)
    # ICA can EEG da loc high-pass (>=1Hz) de hoat dong tot - mne canh bao ro dieu
    # nay (drift cham lam giam chat luong tach nguon). Fit tren ban da loc, nhung
    # AP DUNG (subtract) unmixing tim duoc len ban KHONG loc (giu nguyen dai tan
    # cham that su trong ket qua cuoi, chi muon ICA "nhin" ro hon luc fit).
    eeg_for_fit = eeg_raw.copy().filter(l_freq=1.0, h_freq=None, verbose="ERROR")
    ica = mne.preprocessing.ICA(n_components=n_components, rng=rng, max_iter="auto")
    ica.fit(eeg_for_fit)
    ecg_inds, scores = ica.find_bads_ecg(raw, ch_name=ecg_channel, method="correlation")
    ica.exclude = ecg_inds

    cleaned = eeg_raw.copy()
    ica.apply(cleaned)
    return cleaned, ecg_inds, scores


def detect_cardiac_artifact(
    eeg: np.ndarray,
    beat_times: np.ndarray,
    fs: float,
    window_seconds: float = 1.0,
    n_random_controls: int = 200,
    artifact_ratio_threshold: float = 2.0,
    seed: int = 42,
) -> dict:
    """*** KIEM TRA BAT BUOC TRUOC KHI TIN KET QUA SANITY CHECK *** (xem
    docs/PHASE_S_GUIDE.md muc 3.2).

    SUA (review): ban truoc chia bien do dinh-dinh cua trung binh khoa-theo-R-peak
    cho DO LECH CHUAN CUA CHINH TRUNG BINH DO - day khong phai doi chung dung, vi ngay
    ca voi EEG sach (nhieu trang), duong trung binh van co dao dong ngau nhien nho va
    ty so nay khong duoc chuan hoa dung theo ky vong duoi gia thuyet null. Gio dung
    DOI CHUNG NGAU NHIEN THAT: lap lai viec khoa-trung-binh nhieu lan tren cac thoi
    diem NGAU NHIEN (khong lien quan nhip tim) de uoc luong phan phoi "muc nen" that,
    roi so bien do dinh-dinh cua ban khoa-theo-R-peak voi phan phoi do.
    """
    half_win = int((window_seconds / 2) * fs)
    beat_samples = (beat_times * fs).astype(int)

    locked_avg = _locked_average(eeg, beat_samples, half_win)
    if locked_avg is None:
        return {"has_artifact": False, "suspected": False, "peak_amplitude": 0.0,
                "baseline_amplitude": 0.0, "artifact_ratio": 0.0,
                "locked_average": None, "message": "Khong du du lieu de kiem tra."}

    peak_amplitude = float(np.max(locked_avg) - np.min(locked_avg))

    rng = np.random.default_rng(seed)
    n_beats = len(beat_samples)
    valid_range = len(eeg) - 2 * half_win
    baseline_peaks = []
    for _ in range(n_random_controls):
        random_samples = rng.integers(half_win, half_win + max(valid_range, 1), size=n_beats)
        rand_avg = _locked_average(eeg, random_samples, half_win)
        if rand_avg is not None:
            baseline_peaks.append(float(np.max(rand_avg) - np.min(rand_avg)))

    baseline_amplitude = float(np.mean(baseline_peaks)) if baseline_peaks else 1e-12
    artifact_ratio = peak_amplitude / (baseline_amplitude + 1e-12)
    suspected = artifact_ratio > artifact_ratio_threshold

    return {
        "has_artifact": suspected,
        "suspected": suspected,
        "peak_amplitude": peak_amplitude,
        "baseline_amplitude": baseline_amplitude,
        "artifact_ratio": float(artifact_ratio),
        "locked_average": locked_avg,
        "message": (
            f"artifact_ratio={artifact_ratio:.2f} "
            f"({'VUOT NGUONG '+str(artifact_ratio_threshold)+' - NGHI NHIEM TIM' if suspected else 'trong nguong binh thuong'})"
        ),
    }
