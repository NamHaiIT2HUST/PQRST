# Nhịp 2 — Hướng dẫn chi tiết (P′ → T′, chu kỳ lượng tử)

> Cập nhật lại `ROADMAP.md` (mục "NHỊP 2") với 2 thay đổi quan trọng rút ra từ
> chính kết quả Nhịp 1: (1) thêm 1 bước chuẩn bị TRƯỚC P′ để câu chuyện lượng tử
> đứng vững về mặt khoa học (không chỉ "thử cho biết"), (2) sửa lại lý do đầu tư
> lượng tử — KHÔNG còn là "ít tham số giúp N nhỏ" (giả thuyết này đã bị chính
> Pha R Nhịp 1 bác bỏ), mà là một câu hỏi cụ thể hơn (xem mục 1).

---

## 0. Tóm tắt thời gian

| Việc | Bản chất | Thời gian |
|---|---|---|
| **P′.0 — Bổ sung trước khi vào lượng tử (MỚI)** | Ablation cổ điển + hình thức hoá công cụ PCA + đọc lý thuyết | 3–5 ngày |
| P′ — Chuẩn bị mạch lượng tử | Cài PennyLane, viết `T_theta` | 3–5 ngày |
| Q′ — Hoán đổi có kiểm soát | Cắm `T_theta` vào `train.py`, canh chỉnh gradient | 1.5–3 tuần |
| R′ — Kiểm định & so sánh trên synthetic | Chạy lại lưới đã có với `T_theta` | 2–3 tuần |
| S′ — Dữ liệu thật với lượng tử | Chạy `T_theta` qua pipeline Pha S có sẵn | 1–2 tuần |
| T′ — Kết quả cuối & bản thảo hoàn chỉnh | Tổng hợp + ablation lượng tử + viết | 2–3 tuần |
| **Tổng cộng** | | **~9–14 tuần**, có cổng dự phòng có thể dừng sớm ở Q′ |

---

## 1. Trả lời trước: "Vì sao làm lượng tử" — lý do CŨ đã yếu, lý do MỚI là gì

**Lý do cũ (trong `PHASE_R_REPORT.md` mục 6, lúc chưa kiểm chứng):** mạch lượng
tử có rất ít tham số (6-8 qubit × 3-5 lớp ≪ MLP `[128,128,64]`) → nếu lý do
Amortized thua ở N nhỏ là do **overfit vào corpus train** (mạng lớn học quá
khớp), thì kiến trúc ít tham số hơn có thể tổng quát tốt hơn ở đúng vùng N nhỏ.

**Đã bị bác bỏ:** Pha R mục 8 (`docs/PHASE_R_REPORT.md`) đã kiểm định trực tiếp
bằng ablation cổ điển (mạng `[16,16]`, 369 tham số) — **không thắng rõ** ở
N=10-20 so với mạng chính. Nghĩa là "ít tham số" tự nó KHÔNG phải nguyên nhân
giúp tổng quát hoá tốt hơn ở N nhỏ. Nếu vẫn dùng lý do cũ để biện minh cho
lượng tử, reviewer sẽ hỏi đúng câu: "chính báo cáo của các bạn đã bác bỏ giả
thuyết này ở dạng cổ điển — sao lại kỳ vọng nó đúng ở dạng lượng tử?"

**Lý do MỚI, cụ thể hơn (dựa trên lý thuyết đã kiểm chứng, không phải suy đoán):**
mạch lượng tử kiểu **data re-uploading không chỉ "ít tham số"** — nó tạo ra một
**lớp hàm khác về bản chất**: theo Schuld, Sweke & Meyer (*Physical Review A*
103, 032430, 2021), một mạch lượng tử với data re-uploading tương đương một
**chuỗi Fourier riêng phần** của dữ liệu đầu vào, với tập tần số truy cập được
quyết định bởi cách encode dữ liệu — khác hẳn về bản chất so với hàm mà MLP
(tổ hợp affine + ELU) có thể biểu diễn, dù cùng bậc tham số. Câu hỏi khoa học
của Nhịp 2 vì vậy KHÔNG còn là "ít tham số có giúp không" (đã trả lời: không),
mà là: **"lớp hàm dạng Fourier riêng phần này có phù hợp hơn với cấu trúc thật
của bài toán ước lượng TE ở N nhỏ không — và nó có làm hẹp được domain gap đã
thấy ở Pha T.3b (mục 4 dưới) hay không?"**

---

## 2. Pha P′.0 — Bổ sung TRƯỚC khi vào lượng tử (MỚI, chưa có trong `ROADMAP.md` gốc)

Mục đích: làm câu chuyện lượng tử **mới và đứng vững** trước khi tốn công dựng
hạ tầng — đúng tinh thần "kiểm định giả thuyết rẻ trước khi đầu tư đắt" đã dùng
xuyên suốt Nhịp 1.

### 2.1. Ablation "dequantization" cổ điển (quan trọng nhất, nên làm đầu tiên)

Xây 1 mô hình cổ điển mô phỏng ĐÚNG tính chất Fourier của mạch lượng tử (không
cần mạch lượng tử thật) — ví dụ **Random Fourier Features** hoặc lớp đặc trưng
cố định `[sin(kx), cos(kx)]` với vài tần số `k`, rồi 1 lớp tuyến tính phía sau
— số tham số tương đương mạch lượng tử dự kiến (~vài chục đến ~1 trăm tham số).

- Nếu mô hình Fourier cổ điển này **cũng cải thiện** ở N nhỏ/domain gap → tín
  hiệu tốt đến từ "lớp hàm dạng Fourier", không nhất thiết cần lượng tử thật —
  vẫn đáng làm lượng tử (vì mạch lượng tử tiếp cận được không gian Fourier
  rộng hơn với ít tham số hơn nhờ entanglement — điểm mạnh thật của lượng tử),
  nhưng phải nói rõ trong bản thảo đây là điều đã kiểm chứng trước, không chỉ
  "chạy quantum simulator rồi khoe".
- Nếu mô hình Fourier cổ điển **không cải thiện gì** → tín hiệu (nếu có sau
  này từ mạch lượng tử) probably đến từ cấu trúc riêng của lượng tử (entanglement,
  Hilbert space lớn) — câu chuyện "vì sao cần lượng tử" mạnh hơn nhiều.

Cả 2 kết quả đều là câu chuyện tốt cho bản thảo — đây là 1 kiểm định rẻ
(không cần cài PennyLane, chạy classical, tái dùng đúng lưới N=10-200 đã có).

**Việc cần làm:** `src/pqrst/estimators/classical_fourier.py` (`FourierFeatureStatisticsNetwork`,
cùng interface `BaseTEEstimator`) + notebook `notebooks/phase_p2_00_dequantization_ablation.ipynb`
chạy lại đúng lưới `phase_r_grid_summary.csv`/`phase_r2_periodic_grid_summary.csv`.

**Đã làm + 1 bug thật đáng ghi lại cho bản thảo (bài học cho cả Nhịp 2):** bản
đầu tiên của `FourierFeatureStatisticsNetwork` encode RIÊNG từng chiều đầu vào
(`y_t`, `x_lag`, `y_lag`, `mask`) rồi mới ghép tuyến tính ở lớp đọc ra — khiến
hàm `T` chỉ có thể là dạng **TÁCH RỜI CỘNG TÍNH**
`f(y_t)+g(x_lag)+h(y_lag)+k(mask)`, không có số hạng tương tác X↔Y nào. Hậu
quả (chứng minh được bằng toán, bất đẳng thức Jensen áp trên phân phối
marginal của chính chặn dưới Donsker-Varadhan): với `T` tách rời cộng tính,
**giá trị tối ưu tuyệt đối của DV bound luôn là 0** — không cách huấn luyện
nào sửa được (đã thử tăng learning rate 50 lần, thêm hiệu chỉnh gradient EMA
chuẩn của MINE gốc, tăng `windows_per_batch` — vẫn sụp đổ về 0, vì đó là điểm
tối ưu THẬT của chính lớp hàm bị hạn chế, không phải bẫy tối ưu hoá). Sửa đúng:
chiếu cả 4 chiều đầu vào **cùng lúc** qua 1 ma trận ngẫu nhiên cố định trước
khi lấy hoà âm (Random Fourier Features nhiều chiều, Rahimi & Recht 2007) —
tạo số hạng tương tác X↔Y thật, giải quyết triệt để.

**Cảnh báo trực tiếp cho `T_theta` (Pha P′/Q′):** một mạch lượng tử data
re-uploading encode từng đặc trưng vào TỪNG QUBIT RIÊNG rồi chỉ đo (không có
lớp entangling nào, hoặc entangling quá yếu) sẽ mắc ĐÚNG lỗi tách rời cộng
tính này — DV bound cũng sẽ bị chặn ở 0 vì lý do toán học giống hệt, không
phải vì lượng tử "không đủ mạnh". Lớp **entangling layer** (CNOT ring, đã có
trong thiết kế `circuit.py` ở mục 3) chính là thứ tạo ra số hạng tương tác
giữa các qubit — **phải kiểm tra kỹ lớp này có thực sự trộn được thông tin
giữa qubit encode X và qubit encode Y hay không**, đây là điều kiện toán học
bắt buộc để `T_theta` có thể ước lượng TE khác 0, không chỉ là chi tiết thiết
kế phụ.

**Kết quả cuối (đã chạy đầy đủ, không còn sụp đổ) — trả lời câu hỏi mục 1:**

| N | KSG | Amortized (MLP, 25.473 tham số) | Fourier_classical (41 tham số) |
|---|---|---|---|
| **Linear VAR** (MSE) | 0.0170 → 0.0019 | 0.0181 → 0.0039 | 0.0303 → 0.0326 (gần như không đổi) |
| **Periodic Coupling** (MSE) | 0.0294 → 0.0139 | 0.0210 → 0.0059 | 0.1737 → 0.1962 (gần như không đổi) |

`Fourier_classical` **thua cả KSG và Amortized ở MỌI N, trên CẢ 2 loại dữ
liệu** — thua đậm nhất trên dữ liệu phi tuyến (periodic). Loss hội tụ ổn định
có nghĩa (không còn dính ở 0), variance giảm đúng theo N (0.0061→0.00036,
linear) — đây là số liệu THẬT, không phải sụp đổ, và câu trả lời là rõ ràng:
**mô hình Fourier cổ điển ít tham số KHÔNG đủ để cải thiện** so với 2 baseline.

**Đọc kết quả này theo đúng khung mục 1:** đây rơi vào nhánh "Fourier cổ điển
KHÔNG cải thiện gì" → ủng hộ câu chuyện cần lượng tử THẬT (entanglement) mạnh
hơn, không chỉ dạng hàm Fourier ít tham số. **Nhưng cần nêu 1 giới hạn thật khi
viết vào bản thảo:** kết quả này chỉ loại trừ được "41 tham số, đúng cấu trúc
Fourier nhiều chiều" — chưa loại trừ được khả năng đơn giản là CẦN NHIỀU HƠN
41 tham số cổ điển (vd tăng `n_directions`/`n_harmonics`) mà không cần gì
"lượng tử" cả. Nếu muốn kết luận chặt hơn nữa trước khi đầu tư PennyLane, có
thể thử quét `n_directions` lớn hơn (vd 10-20) xem MSE có tiến gần Amortized
không — đây là việc rẻ, làm thêm được nếu muốn chắc chắn hơn, không bắt buộc.

### 2.2. Hình thức hoá công cụ chẩn đoán PCA-theo-khối thành module tái dùng — ĐÃ XONG

Đã rút thành `src/pqrst/evaluation/feature_space.py`: `windows_to_tensors()`,
`collect_block_records()`, `fit_pca()`, `project_pca()` — model-agnostic qua
duck-typing (bất kỳ model có method `forward_blocks(y_t,x_lag,y_lag,mask) ->
dict[str, Tensor]` đều dùng được). Đã thêm `forward_blocks()` vào cả
`MaskedStatisticsNetwork` (amortized.py) và `FourierFeatureStatisticsNetwork`
(classical_fourier.py) — khi có `T_theta` ở Pha Q′/R′/S′, chỉ cần thêm đúng 1
method `forward_blocks()` tương tự vào class mạch lượng tử, KHÔNG cần sửa gì ở
`feature_space.py`.

8 test (`tests/test_feature_space.py`) + đã đối chiếu số **khớp 100%** với hàm
gốc trong `phase_t_04_pca_feature_analysis.ipynb` (không viết lại notebook đó -
kết quả 3 hình đã báo cáo ở Pha T.3b vẫn đúng, không cần chạy lại).

### 2.3. Chuẩn bị môi trường

Cài `pennylane` + `pennylane-lightning` (backend `lightning.qubit` — nhanh hơn
`default.qubit` cho mô phỏng cổ điển mạch nhỏ). Không cần Qiskit (PennyLane có
autograd/PyTorch interface mượt hơn cho use-case này, đã ghi trong roadmap gốc).

**Thời gian mục 2:** 3–5 ngày (chủ yếu là 2.1, ablation cổ điển — code nhẹ, chạy
nhanh vì không cần mạch lượng tử thật).

**Exit của P′.0:** có kết luận rõ ràng cho câu hỏi mục 1 (Fourier cổ điển có
giúp không) TRƯỚC khi viết dòng code lượng tử đầu tiên; công cụ PCA đã tổng
quát hoá, có test.

---

## 3. Pha P′ — Chuẩn bị mạch lượng tử — ĐÃ XONG

- [`src/pqrst/estimators/quantum/circuit.py`](../src/pqrst/estimators/quantum/circuit.py)
  — mạch data re-uploading qua PennyLane (`lightning.qubit`): 6 qubit, 4 lớp,
  mỗi lớp = encode (RY, `encode_scale` học được) + xoay có tham số (RY,RZ) +
  entangling (CNOT vòng tròn).
- [`src/pqrst/estimators/quantum/wrapper.py`](../src/pqrst/estimators/quantum/wrapper.py)
  — `QuantumStatisticsNetwork`, **cùng chữ ký `forward(y_t,x_lag,y_lag,mask)`**
  với `T_phi`/`FourierFeatureStatisticsNetwork`, cộng `forward_blocks()` để
  dùng ngay được với `feature_space.py` (mục 2.2) — không cắm qua class
  `BaseTEEstimator` riêng, cắm thẳng vào `AmortizedTEEstimator` có sẵn giống
  cách đã làm ở Pha P′.0. 61 tham số (6 qubit × 4 lớp).

**Đã áp dụng ĐÚNG bài học từ bug P′.0 (mục 2.1) ngay từ đầu:** có test riêng
(`test_circuit_is_not_additively_separable`) xác nhận entangling thực sự trộn
được thông tin giữa qubit encode X và Y — **pass ngay lần đầu**, không lặp lại
vòng debug đã tốn nhiều công ở bản Fourier cổ điển.

**Exit: ĐẠT** — mạch chạy được, output hợp lệ (không NaN), gradient tính được
qua mọi tham số (8 test, `tests/test_quantum_circuit.py`, xem mục "Test suite"
cuối file). Cài `pennylane`+`pennylane-lightning` (đã thêm vào `requirements.txt`).

**⚠️ Phát hiện quan trọng về hiệu suất (ảnh hưởng trực tiếp đến kế hoạch Q′
dưới đây):** đo trực tiếp trên máy không có GPU — 1 lần forward mạch 6 qubit/4
lớp trên batch N=200 mẫu tốn **~1.4 giây** (mô phỏng lượng tử cổ điển chậm hơn
nhiều bậc so với 1 lần forward MLP, vốn chỉ mất micro-giây). Nếu train trên
TOÀN BỘ corpus 27.000 cửa sổ như `T_phi` ở Pha R (Nhịp 1), ước tính **hàng chục
giờ MỖI EPOCH** — hoàn toàn không khả thi trên máy hiện tại. **Pha Q′ dưới đây
đã điều chỉnh lại quy mô cho thực tế** (không phải lỗi, là giới hạn vật lý của
việc mô phỏng lượng tử bằng máy cổ điển — bản thân đây cũng là 1 điểm đáng nói
trong bản thảo: động lực thật để dùng phần cứng lượng tử thật, không phải mô
phỏng).

---

## 4. Pha Q′ — Hoán đổi có kiểm soát (1.5–3 tuần, ĐÃ ĐIỀU CHỈNH QUY MÔ)

**⚠️ Đọc mục 3 (cảnh báo hiệu suất) trước khi làm mục này** — không train trên
toàn bộ 27.000 cửa sổ như Pha R. Quy mô mới, thực tế:

- Cắm `T_theta` vào `train_amortized()` qua **`model_factory=lambda: QuantumStatisticsNetwork(...)`
  — hạ tầng này ĐÃ CÓ SẴN** từ Pha P′.0 (thêm cho `FourierFeatureStatisticsNetwork`),
  không cần sửa gì thêm ở `train_amortized()`/`losses.py`.
- Gradient qua `diff_method="adjoint"` (đã dùng trong `circuit.py`, nhanh hơn
  parameter-shift trên simulator — parameter-shift cần 2 lần chạy mạch cho MỖI
  tham số MỖI mẫu, quá chậm với 61 tham số).
- **Train trên 1 tập con nhỏ** (gợi ý: 1 cấu hình VAR cố định, vài trăm–1.000
  cửa sổ, ưu tiên N nhỏ 10–50 để giảm cả số mẫu/cửa sổ) — coi đây là "proof of
  concept", KHÔNG so apples-to-apples toàn corpus với `T_phi` ở bước này (việc
  đó dành cho Pha R′, cũng phải thu nhỏ quy mô tương ứng).
- **Áp ngay công cụ PCA-theo-khối (mục 2.2) lên `T_theta`** — không chờ đến
  Pha T′. Nếu ngay từ Pha Q′ đã thấy `T_theta` học được biểu diễn tách theo
  coupling `c` tốt hơn/khác `T_phi`, đó là tín hiệu sớm rất mạnh.

**Exit:** loss giảm ổn định trên tập con; gradient không biến mất (theo dõi
norm gradient theo epoch).

**Kết quả lần chạy đầu (`phase_p2_01_quantum_training_poc.ipynb`, 128 cửa sổ
train, N=20, c=0.6, 12 epoch, ~34 phút):**
- ✅ Loss giảm đều, không sụp đổ về 0 (train 0.005 → -0.008, val 0.0016 → -0.0064) —
  không lặp lại lỗi tách rời cộng tính của bản Fourier.
- ✅ Gradient khoẻ ở cả 3 nhóm tham số (encode_scale 0.79, theta 1.42, readout 2.82).
- ⚠️ **Chưa hội tụ:** loss vẫn đang giảm gần tuyến tính ở epoch 12 (chưa chững).
  DV bound ước lượng ≈ 0.006 nats — còn rất xa mức bản Fourier đã hội tụ (~0.025)
  và xa TE thật (0.18) → mô hình mới chỉ "bắt đầu học".
- ⚠️ **PCA chưa thấy cấu trúc theo `c`** (các điểm trộn lẫn màu) — hợp lý vì mô hình
  còn non; KHÔNG kết luận gì từ hình này cho tới khi hội tụ.

**Kết luận Q′:** đạt điều kiện thoát tối thiểu (loss giảm, gradient sống, không
sụp đổ), nhưng **chưa đủ để đi tiếp R′**. Bước kế: train dài hơn (loss chưa chững)
và/hoặc tăng learning rate (đang 0.001, mô phỏng lượng tử ít bước nên có thể cần
~0.01), rồi mới đánh giá lại PCA + so với `T_phi`.

**🚦 CỔNG DỰ PHÒNG (giữ nguyên từ `ROADMAP.md`):** ≥2 cấu hình siêu tham số
không hội tụ (nghi barren plateau) → dừng Nhịp 2, đưa vào bản thảo Nhịp 1 như
phụ lục/hướng mở — Nhịp 1 đã là bản thảo hoàn chỉnh, không phải rủi ro chí mạng.

---

## 5. Pha R′ — Kiểm định & so sánh trên synthetic (2–3 tuần)

- Chạy `T_theta` trên **đúng** `phase_r_grid_summary.csv` +
  `phase_r2_periodic_grid_summary.csv` đã đóng băng (không sinh corpus mới).
- Thêm `T_theta` vào bảng so sánh: giờ là 5 phương pháp (KSG/Binning/Symbolic/
  `T_phi`/`T_theta`) × N × coupling × noise × 2 loại dữ liệu (linear/periodic).
- Đối chiếu trực tiếp với kết quả ablation Fourier cổ điển (mục 2.1) — đây là
  bảng trung tâm trả lời câu hỏi mục 1.

**Exit:** bảng số liệu đầy đủ trên tập test tách bạch (tái dùng đúng seed
`test_base_seed` đã có, không tính lại).

---

## 6. Pha S′ — Dữ liệu thật với lượng tử (1–2 tuần)

- Chạy `T_theta` qua đúng pipeline `data/processed/` (Fantasia + Apnea-ECG) đã
  có từ Pha S — không xây lại.
- Lặp lại `bidirectional_te_per_record` + `run_sanity_check_per_record` với
  `T_theta`.
- **Chạy lại hình domain-gap (mục 2.2) cho `T_theta`** — so trực tiếp với hình
  đã có của `T_phi` (`phase_t_pca_domain_gap.png`): domain gap có hẹp lại
  không? Đây là câu trả lời trực quan, mạnh nhất cho câu hỏi mục 1.

**Exit:** có kết quả `T_theta` trên dữ liệu thật, đối chiếu được với `T_phi`
(KSG vẫn là baseline chính — không kỳ vọng thắng ngay ở dữ liệu thật).

---

## 7. Pha T′ — Kết quả cuối & bản thảo hoàn chỉnh (2–3 tuần)

- Tổng hợp toàn bộ (giống cấu trúc `phase_t_01_main_results.ipynb` nhưng thêm
  cột `T_theta`).
- Ablation riêng cho lượng tử: độ sâu mạch, số qubit, cách encode — file riêng
  `results/tables/quantum_ablation.csv`.
- Đóng gói checkpoint: `phi_classical.pt` (đã có) + `theta_quantum.pkl` +
  script tái lập.
- Viết phần bổ sung cho bản thảo (Nhịp 2), nối vào bản thảo Nhịp 1 đã viết
  (không viết lại từ đầu).

**Exit (= hoàn thành dự án):** bản thảo đầy đủ so sánh cổ điển–lượng tử, có
câu trả lời rõ cho câu hỏi mục 1 (Fourier/entanglement có giúp hay không, dù
kết quả là có hoặc không), mã nguồn tái lập được từ đầu đến cuối.

---

## 8. Checklist thoát Nhịp 2

- [x] P′.0a: ablation Fourier cổ điển có kết luận rõ (Fourier_classical thua cả KSG và Amortized ở mọi N, 2 loại dữ liệu — ủng hộ hướng cần lượng tử thật, xem mục 2.1)
- [x] P′.0b: hình thức hoá công cụ PCA-theo-khối thành `src/pqrst/evaluation/feature_space.py`
      (model-agnostic qua `forward_blocks()`, đã thêm cho cả `MaskedStatisticsNetwork` và
      `FourierFeatureStatisticsNetwork`, 8 test, đã đối chiếu khớp 100% với hàm gốc trong
      `phase_t_04_pca_feature_analysis.ipynb`)
- [x] P′: `T_theta` chạy được, gradient hợp lệ (8 test pass, kể cả test entangling
      không tách rời cộng tính); phát hiện hiệu suất quan trọng cho Q′ (mục 3)
- [ ] Q′: loss hội tụ trên cấu hình đơn giản, không barren plateau (hoặc đã kích hoạt cổng dự phòng)
- [ ] R′: bảng so sánh 5 phương pháp đầy đủ trên synthetic
- [ ] S′: kết quả `T_theta` trên dữ liệu thật + hình domain-gap so sánh với `T_phi`
- [ ] T′: ablation lượng tử + bản thảo hoàn chỉnh Nhịp 1+2

## 9. Phân công (giữ đúng quy tắc đã dùng ở Pha T)

| Việc | Ai làm |
|---|---|
| Code (P′.0, P′, Q′, R′ hạ tầng) | Agent coding — chia nhỏ, không commit/push |
| Chạy notebook/train (có thể tốn máy ở Q′/R′) | Chủ dự án |
| Viết bản thảo phần Nhịp 2 | Chủ dự án (Claude hỗ trợ soát câu chữ) |
| Review code + số liệu + tính nhất quán với Nhịp 1 | Claude |
| Commit/push/nộp | Chủ dự án |
