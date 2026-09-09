# Roadmap chi tiết — Dự án PQRST (Q-BHC / AQNE-TE)

> Bản gốc của mentor/plan cấp cao nằm ở [`PQRST_Roadmap_goc.md`](PQRST_Roadmap_goc.md). File này **triển khai chi tiết hơn** thành các task cụ thể, ánh xạ trực tiếp vào cấu trúc repo (`src/`, `configs/`, `scripts/`, `tests/`), để mỗi pha có một checklist hành động được, không chỉ mục tiêu trừu tượng.
>
> Quy ước đặt tên: pha của Nhịp 1 (cổ điển) dùng chữ P, Q, R, S, T; pha tương ứng của Nhịp 2 (lượng tử) dùng P′, Q′, R′, S′, T′ và **tái sử dụng gần như toàn bộ code/hạ tầng** của pha cùng chữ cái ở Nhịp 1.

---

## Cách đọc file này

Mỗi pha có 4 phần:
1. **Input** — cái gì phải có sẵn trước khi bắt đầu pha (từ pha trước, hoặc từ bên ngoài).
2. **Việc cần làm** — danh sách task, mỗi task trỏ tới file/module cụ thể trong repo.
3. **Output / Deliverable** — sản phẩm hữu hình khi xong (file, bảng số liệu, hình vẽ, báo cáo).
4. **Tiêu chí thoát (exit criteria)** — điều kiện định lượng để coi là "xong", giữ nguyên từ bản gốc của mentor.

Pha P (Nhịp 1) được triển khai **chi tiết ở mức thực thi** trong [`PHASE_P_GUIDE.md`](PHASE_P_GUIDE.md) — file này chỉ tóm tắt nó, và đi sâu đồng đều hơn cho các pha còn lại.

---

# NHỊP 1 — Chu kỳ cổ điển (MINE gốc, MLP + Donsker–Varadhan)

## Pha P — Khởi động (1–1.5 tuần)

Chi tiết đầy đủ: [`PHASE_P_GUIDE.md`](PHASE_P_GUIDE.md). Tóm tắt:

- Dựng khung repo (`src/pqrst/...`, `configs/`, `tests/`) — **đã làm xong, xem bên dưới**.
- Cài môi trường Python (venv, không dùng conda vì máy không có sẵn) + PyTorch + IDTxl/JIDT (cần JDK, máy đã có JDK 22/25).
- Viết 3 bộ sinh dữ liệu tổng hợp: VAR tuyến tính Gaussian, VAR phi tuyến, ghép nối tuần hoàn.
- Bọc 3 baseline cổ điển (KSG, symbolic TE, binning) sau một interface chung `estimate_te(x, y, method=...)`.
- Script validate baseline vs ground-truth.

**Output:** repo chạy được `pytest`, script `scripts/validate_baselines.py` in ra bảng sai số baseline vs ground-truth.

**Exit:** 3 baseline khớp ground-truth trong sai số chấp nhận được (đề xuất: |bias| < 0.05 nat hoặc < 10% giá trị TE thật, tuỳ cấu hình — quyết định cụ thể khi có số liệu đầu tiên).

---

## Pha Q — MINE cổ điển (1–2 tuần)

Chi tiết đầy đủ (kiến trúc mạng, công thức loss, spec bài toán smoke test đã verify bằng Monte Carlo, checklist thoát): [`PHASE_Q_GUIDE.md`](PHASE_Q_GUIDE.md).

**Input:** Pha P xong — có bộ sinh dữ liệu tổng hợp đơn giản nhất (VAR tuyến tính Gaussian, 1 cấu hình cố định) và baseline để đối chiếu.

**Việc cần làm:**
- `src/pqrst/estimators/mine/network.py` — mạng thống kê `T_φ`: MLP 2–3 lớp ẩn (đề xuất khởi điểm: input = concat(cửa sổ X, cửa sổ Y) hoặc (X, Y, Z) cho conditional sau này; hidden = [64, 64] hoặc [128, 64]; activation ELU/ReLU).
- `src/pqrst/estimators/mine/losses.py` — Donsker–Varadhan bound: `DV(T) = E_joint[T] − log(E_marginal[exp(T)])`, kèm cơ chế shuffle-batch để tạo mẫu từ phân phối marginal (hoán vị Y trong batch, giữ X cố định).
- `src/pqrst/estimators/mine/train.py` — vòng lặp huấn luyện Adam, learning-rate schedule đơn giản, early stopping theo loss trên tập validation tách bạch, log loss theo epoch.
- `scripts/train_mine_smoke_test.py` — chạy trên **một** cấu hình tổng hợp đơn giản (coupling cao, N vừa phải, ví dụ N=100) để kiểm tra vòng lặp chạy đúng.
- (Tuỳ chọn nhưng khuyến nghị) `src/pqrst/estimators/mine/ema.py` — exponential moving average bias correction cho MINE (kỹ thuật chuẩn từ paper gốc để giảm bias của ước lượng log-mean-exp).

**Output:** checkpoint model đầu tiên, plot loss curve hội tụ, giá trị MI ước lượng gần với ground-truth trên 1 cấu hình.

**Exit:** loss giảm ổn định (không NaN, không dao động không kiểm soát); `T_φ` cho MI trong khoảng sai số nhỏ so với ground-truth của cấu hình test.

**Lưu ý rủi ro:** đây là bước ít rủi ro nhất dự án (theo mentor) — MINE 2018 có rất nhiều implementation tham khảo công khai. Nếu vẫn không hội tụ sau khi thử 2-3 cấu hình siêu tham số cơ bản (learning rate, kiến trúc mạng, batch size), khả năng cao là lỗi trong bước chuẩn hoá dữ liệu hoặc cơ chế shuffle-batch — kiểm tra trước khi nghi ngờ thuật toán.

---

## Pha R — Mở rộng & kiểm định đầy đủ (2–3 tuần)

**Input:** Pha Q xong — `T_φ` hội tụ trên 1 cấu hình đơn giản.

**Việc cần làm:**
- Mở rộng bộ sinh dữ liệu (`src/pqrst/data/synthetic/*.py`) để quét lưới cấu hình: nhiều mức coupling strength, nhiều mức nhiễu (SNR), N ∈ {10, 20, 30, 50, 100, 200}.
- `scripts/generate_synthetic_corpus.py` — sinh kho ngữ liệu đầy đủ: ≥20.000 cửa sổ train, ≥2.000 cửa sổ validation **tách bạch theo seed/cấu hình**, lưu vào `data/interim/synthetic_corpus/` (định dạng đề xuất: `.npz` hoặc `.parquet`, kèm file metadata mô tả cấu hình sinh).
- `src/pqrst/estimators/mine/conditional.py` — mở rộng sang conditional MI qua đồng nhất thức `I(A;B|C) = I(A;B,C) − I(A;C)`, dùng lại cùng kiến trúc `T_φ` (train 2 lần với input khác nhau, hoặc 1 mạng nhận thêm cờ điều kiện — quyết định thiết kế cụ thể để tự làm).
- `src/pqrst/evaluation/metrics.py` — bias, variance, MSE của mỗi estimator so với ground-truth, theo từng ô lưới (N × coupling × noise).
- `scripts/compare_estimators_synthetic.py` — chạy `T_φ` + 3 baseline trên toàn bộ lưới, xuất bảng/heatmap.

**Output:** bảng so sánh đầy đủ (estimator × N × coupling × noise), heatmap bias/variance, kho ngữ liệu tổng hợp đóng băng để tái dùng ở Pha S trở đi (và toàn bộ Nhịp 2).

**Exit:** trên tập validation tổng hợp, `T_φ` có phương sai thấp hơn KSG/symbolic TE **ở vùng N nhỏ (N < 30)** — đây là bằng chứng cốt lõi của luận điểm dự án.

---

## Pha S — Dữ liệu thật (2–3 tuần)

**Input:** Pha R xong — `T_φ` đã "đóng băng" kiến trúc + quy trình train (không đổi nữa từ đây).

**Việc cần làm:**
- `src/pqrst/data/real/fantasia.py`, `apnea_ecg.py`, `mitbih_polysomnographic.py`, `cap_sleep.py` — mỗi file: hàm load + tiền xử lý cho 1 bộ dữ liệu (nguồn: PhysioNet). Cần quyết định thư viện đọc tín hiệu (đề xuất: `wfdb` cho định dạng PhysioNet `.dat/.hea`, `pyedflib`/`mne` cho `.edf` nếu có).
- `src/pqrst/data/windowing.py` — trích cửa sổ đặc trưng đồng bộ giữa 2 kênh (tim: RR-interval/HRV; não: EEG band power hoặc tương tự) — **quyết định đặc trưng cụ thể là việc khoa học, cần bạn tự thiết kế** dựa trên tài liệu 4 bộ dữ liệu.
- Xử lý artifact (loại bỏ đoạn nhiễu, ectopic beats trong ECG, v.v.) — mỗi bộ dữ liệu có đặc thù riêng, ghi chú vào `docs/references/`.
- `scripts/run_sanity_check.py` — chạy `T_φ` (và baseline) trên MIT-BIH Polysomnographic + CAP Sleep Database, kiểm tra TE(tim→não) > TE(não→tim).

**Output:** dữ liệu đã tiền xử lý trong `data/processed/`, báo cáo sanity check.

**Exit:** sanity check đạt trên ít nhất 1 bộ có nhãn rõ. Nếu fail: kiểm tra đồng bộ hoá thời gian giữa 2 kênh trước khi nghi ngờ mô hình.

---

## Pha T — Kết quả & Bản thảo checkpoint (2–3 tuần)

**Input:** Pha S xong — sanity check đạt trên ít nhất 1 bộ dữ liệu thật.

**Việc cần làm:**
- `scripts/run_main_experiment.py` — quét N trên 4 bộ dữ liệu × 4 phương pháp (`T_φ`, KSG, symbolic TE, binning).
- `src/pqrst/evaluation/bootstrap.py` — bootstrap ≥500 lần cho khoảng tin cậy.
- `src/pqrst/evaluation/permutation_test.py` — permutation test cho ý nghĩa thống kê của TE ước lượng được (so với TE = 0 dưới giả thuyết không có ghép nối).
- Viết bản thảo checkpoint (LaTeX hoặc Overleaf riêng, không nhất thiết trong repo code — có thể thêm `paper/` sau).

**Output:** bộ kết quả hoàn chỉnh (bảng + hình), bản thảo checkpoint có thể nộp độc lập.

**🚦 GATE CHUYỂN NHỊP** (giữ nguyên từ bản gốc mentor):
- Kết quả tốt → sang Nhịp 2 với câu chuyện mạnh.
- Kết quả ngang bằng → vẫn đáng sang Nhịp 2, hạ kỳ vọng.
- Hết thời gian/nguồn lực → dừng ở đây, nộp bản thảo checkpoint (Entropy / Frontiers in Network Physiology).

---

# NHỊP 2 — Chu kỳ lượng tử (hoán đổi có kiểm soát)

*Nguyên tắc xuyên suốt: **không viết lại** pipeline dữ liệu/loss/đánh giá của Nhịp 1 — chỉ thay `T_φ` (MLP) bằng `T_θ` (mạch lượng tử) tại đúng 1 điểm nối trong `src/pqrst/estimators/`.*

## Pha P′ — Chuẩn bị lượng tử (3–5 ngày)

**Việc cần làm:**
- Cài PennyLane (khuyến nghị hơn Qiskit thuần cho use-case này vì tích hợp autograd/PyTorch mượt hơn) + backend simulator (`default.qubit` hoặc `lightning.qubit` cho tốc độ).
- `src/pqrst/estimators/quantum/circuit.py` — mạch data re-uploading: 6–8 qubit, 3–5 lớp, mỗi lớp gồm (encode dữ liệu qua rotation gates) + (tham số hoá qua rotation gates có trainable params) + (entangling layer, ví dụ CNOT ring).
- `src/pqrst/estimators/quantum/wrapper.py` — hàm `T_theta(x, y) -> scalar`, cùng chữ ký (interface) với `T_phi` ở Nhịp 1 để cắm thẳng vào `train.py` đã có.

**Output:** mạch chạy được, trả về 1 giá trị vô hướng hợp lệ từ dữ liệu mẫu.

**Exit:** mạch chạy được, output hợp lệ (không NaN, gradient tính được qua ít nhất 1 bước).

## Pha Q′ — Hoán đổi có kiểm soát (1.5–3 tuần)

**Việc cần làm:**
- Cắm `T_theta` vào `src/pqrst/estimators/mine/train.py` (thêm tham số chọn estimator: `"mlp"` hoặc `"quantum"`, tái dùng 100% loss/data loader).
- Gradient qua parameter-shift rule (PennyLane hỗ trợ sẵn qua `qml.grad`/autograd interface) hoặc adjoint differentiation (nhanh hơn trên simulator, không dùng được trên hardware thật).
- Chạy trên đúng cấu hình tổng hợp đơn giản nhất đã dùng ở Pha Q (Nhịp 1) để so sánh apples-to-apples ngay từ đầu.

**Output:** loss curve của `T_theta`, so sánh trực tiếp với loss curve của `T_phi` trên cùng cấu hình.

**Exit:** loss giảm ổn định; gradient không biến mất (theo dõi norm gradient theo epoch — nếu tiến về 0 rất nhanh trong khi loss không cải thiện, đó là dấu hiệu barren plateau).

**🚦 CỔNG DỰ PHÒNG:** ≥2 cấu hình siêu tham số không hội tụ → nghi barren plateau → có thể dừng Nhịp 2, đưa vào bản thảo Nhịp 1 như phụ lục/hướng mở. Không phải rủi ro chí mạng vì Nhịp 1 đã là bản thảo hoàn chỉnh.

## Pha R′ — Kiểm định & So sánh trực tiếp (2–3 tuần)

**Việc cần làm:**
- Chạy `T_theta` trên toàn bộ kho ngữ liệu tổng hợp đã đóng băng từ Pha R (Nhịp 1) — tái dùng `data/interim/synthetic_corpus/`.
- `scripts/compare_estimators_synthetic.py` (đã viết ở Pha R) — thêm `T_theta` vào bảng so sánh 4→5 phương pháp.

**Output:** bảng số liệu trung tâm của bài báo: `T_theta` vs `T_phi` vs 3 baseline, cùng lưới N × coupling × noise.

**Exit:** có bảng so sánh đầy đủ trên tập validation tách bạch.

## Pha S′ — Dữ liệu thật với lượng tử (1–2 tuần)

**Việc cần làm:**
- Chạy `T_theta` qua đúng pipeline `data/processed/` đã có từ Pha S (Nhịp 1) — không xây lại.
- Lặp lại `scripts/run_sanity_check.py` với `T_theta`.

**Exit:** sanity check đạt; có kết quả `T_theta` trên cả 4 bộ dữ liệu thật.

## Pha T′ — Kết quả cuối & Bản thảo hoàn chỉnh (2–3 tuần)

**Việc cần làm:**
- `scripts/run_main_experiment.py` (đã viết ở Pha T) — chạy full cho cả `T_theta` và `T_phi`.
- Ablation riêng cho lượng tử: độ sâu mạch, số qubit, cách encode đặc trưng — file kết quả riêng `results/tables/quantum_ablation.csv`.
- Đóng gói: `results/checkpoints/phi_classical.pt` + `results/checkpoints/theta_quantum.pkl` (hoặc định dạng PennyLane phù hợp), kèm script tái lập.
- Bản thảo hoàn chỉnh → arXiv + tạp chí mục tiêu.

**Exit (= hoàn thành dự án):** bản thảo đầy đủ so sánh cổ điển–lượng tử, mã nguồn tái lập được từ đầu đến cuối.

---

## Bảng ánh xạ Nhịp 1 → Nhịp 2 (tái sử dụng hạ tầng)

| Thành phần | Nhịp 1 (đã xây) | Nhịp 2 (tái dùng) |
|---|---|---|
| Dữ liệu tổng hợp | `src/pqrst/data/synthetic/` + kho ở Pha R | Dùng nguyên, không đổi |
| Dữ liệu thật | `src/pqrst/data/real/` + `data/processed/` ở Pha S | Dùng nguyên, không đổi |
| Loss / shuffle-batch | `src/pqrst/estimators/mine/losses.py` | Dùng nguyên, không đổi |
| Baseline (KSG/symbolic/binning) | `src/pqrst/baselines/` | Dùng nguyên làm đối chứng |
| Estimator | `src/pqrst/estimators/mine/network.py` (`T_phi`) | Thêm `src/pqrst/estimators/quantum/` (`T_theta`), interface giống hệt |
| Evaluation (bootstrap, permutation) | `src/pqrst/evaluation/` | Dùng nguyên, không đổi |

---

## Trạng thái hiện tại

- [x] Khung thư mục repo dựng xong (xem `README.md` ở gốc repo).
- [x] **Pha P xong** — 3 bộ sinh dữ liệu tổng hợp, 3 baseline (KSG/binning/symbolic), script validate, đã qua 1 vòng review sửa lỗi công thức ground-truth. Chi tiết: [`PHASE_P_REPORT.md`](PHASE_P_REPORT.md).
- [x] **Pha Q xong** — `StatisticsNetwork`, `donsker_varadhan_loss`, `shuffle_batch`, `train_mine` implement xong, smoke test MI(X[t-1];Y[t]) đạt sai số 0.8–7.6% qua nhiều seed (ngưỡng 20%), 20/20 test pass, không phát hiện lỗi khi review. EMA bias correction không dùng (quyết định có chủ đích, để dành Pha R). Chi tiết: [`PHASE_Q_GUIDE.md`](PHASE_Q_GUIDE.md).
- [x] **Pha R xong** — corpus 27k/4k/18k cửa sổ, `MaskedStatisticsNetwork` train xong, đánh giá trên lưới đầy đủ. Review phát hiện 1 lỗi chặn (`grid.py` lệch chỉ số x khi dựng lại dữ liệu, ảnh hưởng cả 4 estimator) — đã sửa và chạy lại. Tiêu chí thoát chính thức (thắng KSG ở N<30) **chưa đạt** (9/30), nhưng phát hiện Amortized thắng rõ và cách biệt tăng dần từ N≈30 trở lên — điểm giao cắt đúng tại ranh giới tiêu chí. Chi tiết: [`PHASE_R_REPORT.md`](PHASE_R_REPORT.md).
- [x] Tài liệu hướng dẫn Pha S chi tiết ([`PHASE_S_GUIDE.md`](PHASE_S_GUIDE.md)) + khung code (`src/pqrst/data/real/`, `src/pqrst/evaluation/sanity_check.py`, `src/pqrst/utils/standardize.py`, `configs/real/`, 4 notebook `notebooks/phase_s_*.ipynb`, `tests/test_real_data.py`).
- [ ] Pha S — code thật + chạy 4 notebook. **Điều kiện tiên quyết (S0): train lại `T_φ` bất biến thang đo** — checkpoint hiện tại cho corr chỉ 0.07 với đáp án khi gặp thang đo dữ liệu thật (xem `PHASE_S_GUIDE.md` mục 0).

Tài liệu tạp chí mục tiêu và bảng rủi ro theo nhịp giữ nguyên như bản gốc của mentor — không lặp lại ở đây, xem [`PQRST_Roadmap_goc.md`](PQRST_Roadmap_goc.md).
