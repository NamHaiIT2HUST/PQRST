# Pha R — Hướng dẫn chi tiết (Nhịp 1: Mở rộng & Kiểm định đầy đủ)

> **Đây là pha quan trọng nhất của Nhịp 1.** Pha P dựng hạ tầng, Pha Q chứng minh vòng lặp chạy được — nhưng cả hai đều chưa tạo ra *bằng chứng khoa học* nào. Pha R là nơi luận điểm cốt lõi của cả đề tài được xác lập hoặc bị bác bỏ.

**Input:** Pha Q xong — `train_mine`, `donsker_varadhan_loss`, `StatisticsNetwork` đã chạy đúng trên 1 cấu hình; 3 baseline từ Pha P đã kiểm chứng.

**Tiêu chí thoát (nguyên văn roadmap gốc):** trên tập kiểm định tổng hợp, `T_φ` có **phương sai thấp hơn** KSG/symbolic TE **ở vùng N nhỏ (N < 30)**.

---

## 1. Điều cần hiểu trước khi code: "amortized" nghĩa là gì

Đây là khái niệm cốt lõi, và **cũng là chỗ dễ hiểu sai nhất** — hiểu sai thì viết ra code chạy được nhưng vô nghĩa về mặt khoa học.

**Cách làm SAI (ngây thơ):** với mỗi cửa sổ ngắn cần ước lượng TE, train một mạng MINE riêng trên chính cửa sổ đó. → Với N=20 điểm, mạng không thể học được gì; kết quả sẽ tệ hơn cả KSG. Đây **không phải** amortized.

**Cách làm ĐÚNG (amortized):**
1. **Train một lần** trên kho ngữ liệu lớn gồm hàng chục nghìn cửa sổ, trải rộng nhiều mức coupling / noise / N.
2. **Đóng băng** mạng `T_φ`.
3. Khi gặp cửa sổ **mới, ngắn** (N=20), chỉ **forward pass** — không train lại gì cả.

**Vì sao cách này có thể thắng KSG ở N nhỏ** (đây chính là luận điểm của cả đề tài): KSG là phương pháp phi tham số, phải suy ra mọi thứ từ đúng 20 điểm nó nhìn thấy → phương sai rất lớn. `T_φ` thì đã học được *hình dạng chung* của hàm tỉ-số-mật-độ từ hàng chục nghìn cửa sổ trước đó → khi gặp 20 điểm mới, nó chỉ cần **áp dụng tri thức đã có**, không phải suy ra từ đầu. Đó là "learned prior" — thứ KSG về bản chất không thể có.

**Hệ quả cho thiết kế:** tập test **bắt buộc** phải dùng seed hoàn toàn khác tập train (`test_base_seed` trong config). Nếu `T_φ` được đánh giá trên chính dữ liệu nó đã thấy, toàn bộ kết quả vô giá trị.

---

## 2. Ba thay đổi kỹ thuật so với Pha Q

### 2.1. Conditional MI qua phân rã — ĐÃ VERIFY

Pha Q ước lượng MI không điều kiện. TE thì là **conditional** MI:

```
TE(X→Y) = I(Y[t]; X[t-1] | Y[t-1]) = I(Y[t]; X[t-1], Y[t-1]) − I(Y[t]; Y[t-1])
                                      \____ "mi_full" ____/   \_ "mi_reduced" _/
```

Tôi đã **kiểm chứng đồng nhất thức này bằng số**: so với công thức đóng TE độc lập từ Pha P, sai khác **1.4e-16** (độ chính xác máy) tại `a=0.5,b=0.5,c=0.6,noise_std=0.5`. Đây là nền móng của cả Pha R nên đã được xác nhận trước, không phải giả định.

Mỗi số hạng là một MI *không điều kiện* → dùng lại nguyên si cơ chế DV + shuffle-batch từ Pha Q, **không cần viết loss mới**.

### 2.2. Batch bây giờ là "batch các CỬA SỔ", không phải "batch các mẫu"

**Đây là lỗi dễ mắc nhất của cả Pha R.** Cơ chế shuffle-batch chỉ đúng khi các mẫu bị hoán vị đến **từ cùng một phân phối**. Nếu trộn mẫu từ nhiều cấu hình khác nhau (coupling khác nhau) rồi shuffle chung, số hạng marginal sẽ ước lượng sai một phân phối không tồn tại.

→ **Đúng:** một batch = nhiều cửa sổ; DV bound tính **riêng trong từng cửa sổ** (shuffle nội bộ cửa sổ đó), rồi lấy trung bình các cửa sổ làm loss.

### 2.3. Sửa lỗi phương sai đã phát hiện ở Pha Q

Review Pha Q ([`PHASE_Q_REPORT.md`](PHASE_Q_REPORT.md) mục 4) phát hiện 2 vấn đề, **cả hai bắt buộc sửa ở Pha R**:

| Vấn đề Pha Q | Sửa ở Pha R |
|---|---|
| Chỉ 1 lần shuffle mỗi lần đánh giá → phương sai rất cao | `eval_n_shuffles >= 20` (config), trung bình nhiều lần shuffle |
| Lấy "best-of-all-epochs" → selection bias, số đẹp giả tạo | `final_estimate_last_k_epochs = 10` — trung bình k epoch cuối sau hội tụ |

**Lý do bắt buộc:** Pha R so trực tiếp `T_φ` với baseline. Nếu `T_φ` được "chọn điểm may mắn nhất trong hàng chục lần đo" còn baseline thì không, kết luận "MINE thắng" là **giả tạo** — đúng loại lỗi khiến bài báo bị phản biện đánh sập.

**Cảnh báo thêm:** hiệu của 2 ước lượng nhiễu thì nhiễu hơn từng cái. Vì vậy notebook 04 **bắt buộc** báo cáo phương sai của `mi_full` và `mi_reduced` riêng, không chỉ của `te`. Dùng chung một mạng (có mask) cho cả 2 số hạng giúp sai số tương quan dương và triệt tiêu bớt — đó là lý do thiết kế `MaskedStatisticsNetwork`.

---

## 3. Bước 1 — Kho ngữ liệu (`notebooks/phase_r_01_generate_corpus.ipynb`)

**File code:** `src/pqrst/data/synthetic/corpus.py` · **Config:** `configs/synthetic/corpus.yaml`

Lưới quét: 5 mức coupling × 3 mức noise × 6 giá trị N × 300 cửa sổ = **27.000 cửa sổ** (roadmap yêu cầu ≥20.000), chia 85/15 → ~4.050 cửa sổ validation (yêu cầu ≥2.000). Tập test riêng: 18.000 cửa sổ, seed hoàn toàn khác.

**Điểm bắt buộc đúng:**
- Mỗi cửa sổ **một seed riêng biệt** — trùng seed = có bản sao trùng lặp, hỏng tính độc lập của tập đánh giá. Có test kiểm tra việc này.
- Chia train/val **theo cửa sổ nguyên vẹn**, không trộn mẫu của cùng một cửa sổ vào cả 2 tập (rò rỉ dữ liệu).
- Ground-truth chỉ phụ thuộc tham số, không phụ thuộc N → tính 1 lần mỗi ô lưới, đừng tính lại mỗi cửa sổ.

**Kết quả mong muốn:** file corpus nạp lại được nguyên vẹn; bảng ground-truth cho thấy TE tăng đơn điệu theo coupling và bằng 0 khi `c=0`.

## 4. Bước 2 — Huấn luyện amortized (`notebooks/phase_r_02_train_amortized.ipynb`)

**File code:** `src/pqrst/estimators/mine/amortized.py` · **Config:** `configs/mine/amortized.yaml`

`MaskedStatisticsNetwork`: một mạng dùng chung cho cả 2 số hạng, phân biệt bằng cờ mask (`mask=1` → chế độ full; `mask=0` → `x_lag` bị zero-out, chế độ reduced). Input dim = 4.

**Phương án dự phòng** nếu train không ổn định: 2 mạng riêng (input_dim 3 và 2). Nếu phải dùng, **ghi rõ trong báo cáo** và đo lại phương sai TE để so sánh.

**Kết quả mong muốn:** loss hội tụ; checkpoint `results/checkpoints/phi_amortized.pt` nạp lại cho ra đúng kết quả cũ.

> ⏱️ **Đây là bước tốn thời gian nhất Pha R.** Notebook có sẵn ô "chạy thử `max_epochs=2` để ước lượng thời gian" — **luôn chạy ô đó trước** khi chạy full.

## 5. Bước 3 — Đánh giá trên lưới (`notebooks/phase_r_03_evaluate_grid.ipynb`)

**File code:** `src/pqrst/evaluation/grid.py`

4 estimator (`Amortized`, `KSG`, `Symbolic`, `Binning`) chạy trên tập test tách bạch. Tất cả tuân thủ `BaseTEEstimator` — chính là lý do thiết kế interface đó từ Pha P: vòng lặp đánh giá không cần biết bên trong là gì, và Nhịp 2 chỉ cần thêm `T_θ` vào dict.

**Điểm bắt buộc đúng:** lưu kết quả **thô** (mỗi dòng = 1 cửa sổ × 1 estimator), không tổng hợp sẵn — để tầng phân tích tự do nhóm lại và audit từng giá trị. Estimator lỗi trên cửa sổ nào thì ghi `NaN` + đếm, không để vỡ vòng lặp; **tỉ lệ thất bại của baseline ở N nhỏ tự nó là một kết quả đáng báo cáo**.

## 6. Bước 4 — Phân tích & tiêu chí thoát (`notebooks/phase_r_04_analysis.ipynb`)

**Hình chính của cả Pha R:** phương sai theo N cho 4 estimator (log-log, đường dọc tại N=30). Đây là bằng chứng trực quan cho luận điểm cốt lõi.

**Chỉ số quan trọng nhất là PHƯƠNG SAI, không phải bias** — tiêu chí thoát nói về phương sai. Một estimator có bias lớn nhưng phương sai nhỏ vẫn thoả tiêu chí này (và bias có thể hiệu chỉnh được, phương sai thì không).

**`check_exit_criterion`** kiểm tra định lượng: `T_φ` thắng phương sai ở bao nhiêu % ô lưới có N<30, so với **cả** KSG **và** Symbolic. Ngưỡng đề xuất: thắng ≥70% số ô.

> ⚠️ **Nếu không đạt: báo cáo trung thực là không đạt.** Tuyệt đối không nới ngưỡng cho vừa số liệu — đó là p-hacking. Roadmap gốc đã tính sẵn tình huống này: "MLP không tách được tín hiệu ở N rất nhỏ" được xếp mức rủi ro **Trung bình**, và vẫn thành bài báo hợp lệ ("giới hạn của amortized estimation cổ điển"). Một kết quả âm tính trung thực có giá trị hơn một kết quả dương tính bị ép.

---

## 7. Checklist thoát Pha R

**Nhóm A — Code (agent làm):**
- [ ] `corpus.py`: `generate_corpus`, `split_corpus_by_seed`, `save_corpus`, `load_corpus`
- [ ] `var_linear_gaussian.py`: `compute_var_linear_ground_truths` (3 ground-truth)
- [ ] `conditional.py`: `estimate_mi_from_window`, `estimate_te_from_window`
- [ ] `amortized.py`: `MaskedStatisticsNetwork`, `AmortizedTEEstimator`, `train_amortized`
- [ ] `grid.py`: `evaluate_estimators_on_grid`, `summarize_grid`, `check_exit_criterion`
- [ ] 15 test trong `tests/test_corpus_and_conditional.py` — thay hết `pytest.skip`
- [ ] 4 notebook điền đầy đủ, chạy được tuần tự từ 01 → 04

**Nhóm B — Chạy thực nghiệm (chủ dự án chạy trên máy mình):**
- [ ] Notebook 01 chạy xong, corpus lưu ra đĩa
- [ ] Notebook 02 chạy xong, checkpoint lưu lại, loss hội tụ
- [ ] Notebook 03 chạy xong, `phase_r_grid_raw.csv` có dữ liệu
- [ ] Notebook 04 chạy xong, hình vẽ + verdict tiêu chí thoát

**Nhóm C — Kiểm định chất lượng:**
- [ ] `pytest -q` toàn bộ pass (20 test cũ + 15 test mới)
- [ ] Đồng nhất thức `te == mi_full − mi_reduced` đúng đến 1e-9
- [ ] Mọi seed trong corpus là duy nhất
- [ ] Tập test không giao với tập train
- [ ] `eval_n_shuffles >= 20` và **không** dùng best-of-all-epochs (2 lỗi Pha Q)
- [ ] Phương sai của `mi_full`, `mi_reduced`, `te` được báo cáo riêng

---

## 8. Phân công

| Việc | Ai làm | Ghi chú |
|---|---|---|
| Viết code (nhóm A) | Agent coding | Không commit, không push |
| Chạy 4 notebook (nhóm B) | **Chủ dự án** | Agent chỉ đưa lệnh, không tự chạy |
| Review code + kết quả | Claude (reviewer) | Như đã làm ở Pha P, Q |
| Commit / push | **Chủ dự án** | Claude chỉ in lệnh ra |

**Lệnh chạy notebook** (chủ dự án chạy trong PowerShell tại thư mục repo):

```bash
.\.venv\Scripts\Activate.ps1
```
```bash
$env:JAVA_HOME = "C:\Program Files\Java\jdk-22"
```
```bash
jupyter lab
```
Rồi mở lần lượt `notebooks/phase_r_01…` → `04` và chạy từng ô.

Nếu chưa có Jupyter:
```bash
pip install jupyterlab
```

## 9. Bước tiếp theo

Xong Pha R → **Pha S** (pipeline 4 bộ dữ liệu sinh lý thật + sanity check TE(tim→não) > TE(não→tim)). Đây là lần đầu dự án chạm dữ liệu thật, và cũng là nơi bắt đầu tốn thời gian debug tiền xử lý.
