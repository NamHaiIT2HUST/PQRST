# Báo cáo Pha R — Amortized MINE: Thiết kế, Lỗi phát hiện, và Kết quả đã sửa

> Bản báo cáo này thay thế bản gốc do agent coding viết — sau review phát hiện 1 lỗi
> chặn làm sai toàn bộ bảng kết quả trung tâm. Số liệu trong tài liệu này đã chạy lại
> với lỗi đã sửa, dùng đúng checkpoint đã train (không cần train lại).

---

## 1. Tóm tắt cho người bận

- **Pha R đạt về mặt hạ tầng** (corpus, phân rã conditional MI, mạng Masked, pipeline đánh giá) — tất cả đã verify đúng.
- **Phát hiện 1 lỗi chặn** trong bước đánh giá (`grid.py`) làm sai bảng kết quả trung tâm. Đã sửa, chạy lại toàn bộ 72.000 phép đo.
- **Tiêu chí thoát chính thức (thắng KSG ở N<30, ≥70% ô lưới): CHƯA ĐẠT** — cả trước và sau khi sửa lỗi (9/30 ô sau khi sửa, so với 6/30 báo cáo gốc — sửa lỗi làm Amortized có vẻ *khá hơn một chút*, không phải tệ hơn).
- **Nhưng kết quả sau khi sửa cho thấy 1 hiện tượng đáng chú ý mà báo cáo gốc không thấy được**: Amortized **thắng rõ và ngày càng cách biệt từ N≈30 trở lên**, chỉ thua ở đúng N=10 và N=20. Đây là phát hiện khoa học thật, đáng để tiếp tục khai thác — không phải một kết quả "thua toàn diện".

---

## 2. Rà soát các quyết định thiết kế (xác nhận lại độc lập)

| Quyết định | Đánh giá | Ghi chú |
|---|---|---|
| `MaskedStatisticsNetwork` dùng chung 1 mạng (mask=full/reduced) thay vì 2 mạng riêng | ✅ Đúng, code khớp thiết kế | Zero-out `x_lag` khi mask=0 thực hiện đúng trong `forward()` |
| Shuffle marginal dùng **chung 1 permutation** cho cả `mi_full` và `mi_reduced` để tạo tương quan dương, giảm phương sai hiệu | ✅ Đúng, đã đọc kỹ `conditional.py` | Đây chính là cơ chế giảm phương sai — xác nhận hoạt động đúng |
| `eval_n_shuffles=20` thay vì 1 lần (sửa lỗi Pha Q) | ✅ Đúng, đã dùng trong `train_amortized.evaluate()` | |
| Batch = nhiều cửa sổ, DV bound tính riêng trong từng cửa sổ | ✅ Đúng, không trộn mẫu giữa các cấu hình | Đúng lưu ý quan trọng nhất của guide |
| "Checkpoint cuối = trung bình 10 epoch cuối, không phải best-epoch" | ⚠️ **KHÔNG khớp code thật** | Xem mục 4 |

**Xác nhận độc lập số liệu phương sai từng số hạng** (mục 3 báo cáo gốc, không đi qua đường dẫn bị lỗi ở mục 3 dưới đây nên số liệu này **đáng tin**): `var(mi_full)≈0.012`, `var(mi_reduced)≈0.007`, `var(te)≈0.0036` — đúng là giảm mạnh so với tổng 2 số hạng (0.019), xác nhận cơ chế Masked Network hoạt động như thiết kế.

---

## 3. LỖI CHẶN đã phát hiện và sửa: dữ liệu đầu vào bị lệch chỉ số trong `grid.py`

**Vị trí:** `src/pqrst/evaluation/grid.py`, hàm `evaluate_estimators_on_grid`, đoạn dựng lại `(x, y)` từ `Window` để gọi `estimator.estimate(x, y)`:

```python
# TRƯỚC (sai):
x[:-1] = w.x_lag
x[1:] = w.x_lag   # <-- ghi đè lên dòng trên, làm x lệch 1 chỉ số
```

**Hậu quả:** mọi giá trị `x` đưa vào **cả 4 estimator** (Amortized, KSG, Symbolic, Binning) trong toàn bộ 72.000 phép đo bị lệch — tương đương như đo `TE(X[t-2]→Y[t])` chứ không phải `TE(X[t-1]→Y[t])` như thiết kế.

**Verify bằng số** (cùng 1 cấu hình `a=0.5,b=0.5,c=0.6,noise_std=0.5`):

| Cách tính | KSG (nats) |
|---|---|
| Gọi trực tiếp trên dữ liệu gốc | 0.1074 |
| Dựng lại đúng (đã sửa) | 0.1074 ✅ khớp |
| Dựng lại theo code lỗi | 0.1156 ❌ lệch |

**Bằng chứng gián tiếp củng cố thêm** — nhìn `bias vs N` của bản BUG (không đưa vào báo cáo này): cả KSG và Amortized đều có bias **âm gần như hằng số** xuyên suốt mọi N (không co về 0 khi N tăng) — bất thường với một estimator nhất quán (consistent) như KSG. Sau khi sửa, bias KSG co về 0 đúng như lý thuyết (xem mục 5).

**Đã sửa:** dùng `x[-1] = w.x_lag[-1]` (giá trị placeholder, đã verify thực nghiệm là không ảnh hưởng kết quả vì baseline chỉ đọc `x[:-1]`) thay cho dòng ghi đè sai.

**Đã thêm 2 test hồi quy** (`TestGridReconstruction` trong `tests/test_corpus_and_conditional.py`) để lỗi này không tái diễn — kiểm tra trực tiếp rằng dữ liệu dựng lại khớp dữ liệu gốc, và rằng KSG qua `evaluate_estimators_on_grid` cho đúng số như gọi trực tiếp.

**Tự nhận trách nhiệm:** `docs/PHASE_R_GUIDE.md` (do tôi viết) yêu cầu 15 test cho `corpus.py` và `conditional.py` nhưng **không yêu cầu test nào cho `grid.py`** — đây là khoảng trống trong chính guide gốc khiến lỗi này lọt qua 35/35 test pass. Đã bổ sung.

**Đã chạy lại toàn bộ đánh giá** (414 giây, dùng checkpoint đã train, không train lại) — mọi số liệu từ mục 5 trở đi trong báo cáo này là số liệu ĐÃ SỬA.

---

## 4. Vấn đề mức cao: mô tả "trung bình 10 epoch cuối" không khớp code thật

Báo cáo gốc viết: *"Checkpoint cuối cùng không phải là 'best epoch', mà là... trung bình cộng... trong 10 epoch có val loss phẳng nhất."*

**Thực tế trong `train_amortized()`:** model được deploy là `best_model_state` — trọng số của **đúng 1 epoch có val loss thấp nhất trong toàn bộ quá trình train** (cùng cơ chế early-stopping đã gây lỗi selection-bias ở Pha Q). Biến `final_val_loss` (trung bình 10 epoch cuối) **có được tính** nhưng chỉ nằm trong `history` để log, **không được dùng để chọn model deploy**.

**Vì sao chưa nghiêm trọng như ở Pha Q:** `evaluate()` dùng **cùng 1 seed cố định cho mọi epoch** (không random lại mỗi epoch) — nghĩa là nhiễu đánh giá gần như *giống nhau* qua các epoch, nên "chọn best epoch" ở đây gần với so sánh model thật hơn là "chọn nhiễu may mắn nhất" như ở Pha Q. Độ lệch chuẩn val loss 10 epoch cuối đo được là `0.00247` — thấp hơn nhiều so với `0.0055` ở Pha Q. Đường loss cong cũng hội tụ rất mượt (xem `phase_r_amortized_training_loss.png`), không giống hiện tượng chọn-điểm-nhiễu ở Pha Q.

**Khuyến nghị (không chặn, để dành lần train lại sau):** sửa `train_amortized` để thực sự lưu/trung bình trọng số (hoặc dự đoán) của k epoch cuối, thay vì chỉ log số liệu trung bình mà không dùng. Không retrain ngay bây giờ vì train 1 lần đã tốn ~5 giờ và ảnh hưởng thực tế ở lần train này có vẻ nhỏ (đường loss ổn định).

---

## 5. Kết quả đã sửa

### 5.1. Tiêu chí thoát chính thức

| | Báo cáo gốc (lỗi) | Sau khi sửa |
|---|---|---|
| Thắng KSG (N<30) | 6/30 (20%) | **9/30 (30%)** |
| Thắng Symbolic (N<30) | 29/30 | 28/30 |
| Thắng Binning (N<30) | (không tính riêng) | 30/30 |
| **Đạt tiêu chí (≥70% cả 2)** | ❌ Không | ❌ **Không** |

Tiêu chí thoát chính thức (thắng KSG **và** Symbolic ở ≥70% ô có N<30) **chưa đạt**, kể cả sau khi sửa lỗi.

### 5.2. Nhưng đây là bức tranh đầy đủ hơn — điểm giao cắt ở N≈30

![variance vs N](../results/figures/phase_r_variance_vs_n.png)

Phân theo N (trung bình qua mọi mức coupling/noise):

| N | Var(Amortized) | Var(KSG) | Ai thắng |
|---|---|---|---|
| 10 | 0.0117 | 0.0037 | KSG (rõ) |
| 20 | 0.0064 | 0.0054 | KSG (nhẹ) |
| **30** | **0.0042** | **0.0053** | **Amortized** |
| 50 | 0.0026 | 0.0044 | Amortized |
| 100 | 0.0014 | 0.0029 | Amortized (gấp 2) |
| 200 | 0.0007 | 0.0018 | Amortized (gấp 2.6) |

**Amortized thắng rõ và cách biệt ngày càng lớn từ N=30 trở lên — chỉ thua ở đúng N=10 và N=20.** Tiêu chí gốc chọn ngưỡng N<30 hoá ra rơi đúng vào vùng Amortized còn yếu nhất. Đây **không phải một kết quả "thua toàn diện"** như con số 9/30 nghe có vẻ vậy — mà là một điểm giao cắt (crossover) rõ ràng, ngay tại đúng ranh giới N=30 được chọn làm tiêu chí.

### 5.3. Bias: khác biệt bản chất giữa 2 loại estimator

![bias vs N](../results/figures/phase_r_bias_vs_n.png)

KSG là estimator **nhất quán (consistent)** — bias co về 0 đúng lý thuyết khi N tăng (`-0.078 → -0.008`). Amortized có bias co lại một phần nhưng **không về 0** (`-0.065 → -0.044`, giữ nguyên từ N=50 trở lên) — vì mạng đã đóng băng, không "học thêm" khi gặp cửa sổ dài hơn, bias của nó bị chặn bởi năng lực mô hình lúc train, không phải bởi N của cửa sổ đang xét. Đây là đặc trưng bản chất của phương pháp amortized, không phải lỗi.

**Hệ quả thực tiễn:** vì MSE = bias² + variance, bias không-về-0 của Amortized làm MSE của nó thua KSG ở N lớn dù variance thấp hơn hẳn. **Amortized đang "trả giá" thắng lợi về phương sai bằng một khoản bias cố định** — đây là hướng cải thiện rẻ nhất (mục 6).

---

## 6. Đề xuất cải thiện (trả lời câu hỏi "hướng nào để lật đổ KSG ở Nhịp 2")

**Trước khi nghĩ tới lượng tử — 3 việc cổ điển rẻ, nên làm trước:**

1. **Hiệu chỉnh bias (calibration) — rẻ nhất, không cần train lại.** Bias của Amortized khá ổn định về dấu và độ lớn theo N (mục 5.3). Fit 1 đường hiệu chỉnh đơn giản (vd hồi quy tuyến tính `bias(N)` trên tập validation) rồi trừ đi khi suy luận — có thể giúp Amortized thắng cả về MSE ở N≥30, không chỉ variance.
2. **Tập trung cải thiện đúng vùng N=10-20** — nơi variance của Amortized tăng vọt theo noise (`0.0037→0.0209` khi noise 0.3→1.0 ở N=10) trong khi KSG gần như không đổi (`0.0031→0.0029`). Amortized nhạy với nhiễu hơn nhiều ở N cực nhỏ — có thể do bài toán "hiệu 2 số hạng nhiễu" (đã cảnh báo trong `conditional.py`) trở nên nghiêm trọng hơn khi ít mẫu. Thử: tăng tỉ trọng cửa sổ N=10,20 trong corpus train (hiện tại đều 4.500 mỗi N, có thể tăng riêng N nhỏ), hoặc thêm 1 dạng regularization/shrinkage đặc thù cho N nhỏ.
3. **Kiểm định trên `var_nonlinear.py` / `periodic_coupling.py` (đã có sẵn từ Pha P), không chỉ VAR tuyến tính Gaussian.** KSG rất mạnh trên đúng bài toán Gaussian tuyến tính này — đây là "sân nhà" của phương pháp k-NN cho phân phối trơn, low-dimensional. Amortized có thể có lợi thế rõ hơn trên bài toán phi tuyến, nơi giả định hình học của KSG kém khớp hơn.

**Về câu hỏi lượng tử cụ thể:**

Ý nghĩa thật của việc thay MLP bằng mạch lượng tử **không phải là "lượng tử tính nhanh/giỏi hơn"** — mà là mạch data re-uploading có **rất ít tham số** (6-8 qubit × 3-5 lớp ≪ số tham số của MLP `[128,128,64]`). Nếu lý do Amortized thua ở N=10-20 là do **overfit vào corpus train** (mạng lớn học quá khớp các mẫu train, tổng quát hoá kém ở vùng dữ liệu cực ngắn), thì một kiến trúc tham số ít hơn — dù cổ điển hay lượng tử — có thể tổng quát tốt hơn ở đúng vùng N nhỏ.

**Kiểm định giả thuyết này TRƯỚC khi đầu tư vào lượng tử** (rẻ, làm được ngay bằng cổ điển): thử lại `MaskedStatisticsNetwork` với `hidden_dims` nhỏ hơn nhiều (vd `[16,16]` thay vì `[128,128,64]`) và train lại trên đúng corpus này.
- Nếu mạng nhỏ hơn thắng KSG rõ hơn ở N=10-20 → xác nhận giả thuyết "ít tham số = tổng quát tốt hơn ở N nhỏ", và đó chính là lý do chính đáng để kỳ vọng mạch lượng tử (vốn có rất ít tham số) giúp được ở Nhịp 2.
- Nếu mạng nhỏ hơn KHÔNG cải thiện (hoặc tệ hơn) → vấn đề không nằm ở số tham số, mà ở nơi khác (thiết kế loss, cách lấy mẫu, hay giới hạn lý thuyết của chính phân rã conditional MI) — và khi đó mạch lượng tử (cũng ít tham số) nhiều khả năng **không giải quyết được** vấn đề, cần nhìn hướng khác trước khi tốn công dựng Nhịp 2.

---

## 7. Đã triển khai tiếp — vòng cải thiện N nhỏ (theo quyết định đầu tư thêm)

Chủ dự án quyết định đầu tư thêm để cải thiện vùng N=10-20 trước khi qua Pha S. Đã làm:

**7.1. Sửa `train_amortized` — deploy đúng trung bình trọng số k epoch cuối**
Mục 4 chỉ ra model deploy thực tế vẫn là best-of-all-epochs (không khớp mô tả báo cáo). Đã sửa: model deploy giờ là **trung bình tham số (weight averaging)** của `final_estimate_last_k_epochs` epoch cuối cùng đã chạy — không còn chọn theo epoch có val loss thấp nhất. Đã verify bằng test (`TestTrainAmortized` trong `test_corpus_and_conditional.py`, đối chiếu trực tiếp bằng toán học rằng trọng số deploy = trung bình đúng k state_dict cuối). **Checkpoint chính hiện có (`phi_amortized.pt`) chưa được train lại với logic mới** — số liệu ở mục 5 vẫn dùng checkpoint cũ (tác động thực tế thấp, đã giải thích ở mục 4); logic mới sẽ áp dụng cho các lần train tiếp theo, bao gồm ablation dưới đây.

**7.2. Ablation mạng nhỏ — kiểm định giả thuyết trước khi đầu tư Nhịp 2 (mục 6)**
Đã tạo `configs/mine/amortized_small.yaml` (chỉ đổi `hidden_dims: [16,16]` so với bản chính `[128,128,64]`, mọi thứ khác giữ nguyên — ablation sạch, đổi đúng 1 biến) và `notebooks/phase_r_05_ablation_small_network.ipynb` (code đầy đủ, không phải stub). Notebook tái dùng corpus đã sinh từ notebook 01 (không cần sinh lại), train mạng nhỏ, chỉ đánh giá lại đúng estimator mới trên tập test (không chạy lại KSG/Binning/Symbolic vì kết quả các baseline đó không đổi), rồi so trực tiếp phương sai ở N<30 giữa mạng lớn/mạng nhỏ/3 baseline.

Đã đo thử 2 epoch trên đúng corpus thật: **~77s/epoch** (so với ~186s/epoch của mạng lớn) → ước tính **~2,1 giờ** cho đủ 100 epoch — nhanh hơn ~2.4 lần so với lần train chính.

**Cần chủ dự án chạy** (không tự chạy hộ theo yêu cầu — báo lại kết quả để review):
```bash
.\.venv\Scripts\Activate.ps1
```
```bash
$env:JAVA_HOME = "C:\Program Files\Java\jdk-22"
```
```bash
jupyter lab notebooks/phase_r_05_ablation_small_network.ipynb
```
Chạy tuần tự từng ô. Mục 2 của notebook sẽ tự in lại ước tính thời gian trên máy bạn trước khi chạy full ở mục 3 — kiểm tra con số đó hợp lý (không phải hàng chục giờ) rồi mới chạy tiếp.

## 8. Kết quả ablation mạng nhỏ — giả thuyết "ít tham số" bị bác bỏ

Đã chạy xong `notebooks/phase_r_05_ablation_small_network.ipynb` (100 epoch, ~116 phút, hội tụ tốt — val loss std=0.0005, thấp hơn cả mạng chính).

| | Mạng LỚN `[128,128,64]` | Mạng NHỎ `[16,16]` |
|---|---|---|
| Bias @ N=10 | -0.0651 | -0.0635 |
| Variance @ N=10 | 0.01168 | 0.01237 (nhích cao hơn) |
| Variance @ N=20 | 0.00639 | 0.00679 (nhích cao hơn) |
| Thắng KSG (N<30) | **9/30** | **9/30** |

![variance comparison](../results/figures/phase_r_ablation_variance_comparison.png)

**Kết luận rõ ràng: giảm ~70 lần số tham số không cải thiện gì** — 2 đường variance-vs-N gần như song song và sát nhau suốt từ N=10 đến N=200, cùng cắt KSG ở đúng N≈30, cùng thắng đúng 9/30 ô.

**Ý nghĩa:** giả thuyết "ít tham số = tổng quát tốt hơn ở N nhỏ" — lý do chính đáng nhất để kỳ vọng mạch lượng tử (cũng ít tham số) giúp được ở Nhịp 2 — **bị bác bỏ bởi bằng chứng thực nghiệm này**. Vấn đề ở N=10-20 nhiều khả năng không nằm ở dung lượng mô hình, mà ở **nhiễu Monte Carlo không thể giảm vốn có của công thức DV bound** khi số điểm dữ liệu quá ít (ước lượng `logsumexp` trên chỉ 10-20 điểm có phương sai lớn bất kể mạng "khéo" tới đâu) — đây là giới hạn của chính phương pháp đánh giá, không phải của kiến trúc mạng.

## 9. Khuyến nghị hướng tiếp theo

Với kết quả null này, 2 lựa chọn hợp lý:

**(a) Chấp nhận và dùng estimator lai (hybrid) — khuyến nghị, hiệu quả nhất về thời gian.** Điểm giao cắt N≈30 rất rõ và ổn định qua cả 2 lần thử nghiệm (mạng lớn, mạng nhỏ). Có thể dùng thẳng: **KSG cho N<30, Amortized cho N≥30** — tận dụng đúng điểm mạnh của từng phương pháp, không cần nghiên cứu thêm, và về khoa học vẫn là 1 kết quả tốt (amortized thắng rõ ở vùng N vừa/lớn — vẫn có giá trị công bố).

**(b) Đầu tư thêm — thử trên `var_nonlinear.py`/`periodic_coupling.py` (đã có sẵn từ Pha P) thay vì chỉ VAR tuyến tính Gaussian.** KSG rất mạnh trên đúng bài toán Gaussian tuyến tính (sân nhà của k-NN cho phân phối trơn). Có thể trên dữ liệu phi tuyến, khoảng cách sẽ khác hẳn. Đây là hướng còn chưa thử, nhưng tốn công hơn (phải mở rộng `corpus.py` sang 2 generator đó, sinh lại corpus, train lại).

**(c) Hiệu chỉnh bias hậu-nghiệm** (mục 6.1, rẻ, không train lại) — cải thiện MSE thực tế nhưng không đổi tiêu chí thoát dựa trên phương sai.

## 10. Việc còn lại

- [x] Sửa lỗi `grid.py`, thêm test hồi quy, chạy lại toàn bộ 72.000 phép đo.
- [x] Sửa `train_amortized` để deploy trung bình trọng số k epoch cuối, có test xác nhận.
- [x] Chạy ablation mạng nhỏ — kết quả null, đã phân tích ở mục 8.
- [ ] **Chờ quyết định chủ dự án:** chọn (a), (b), hoặc (c) ở mục 9 — hoặc dừng ở đây và qua Pha S với kết quả hiện tại (khuyến nghị nếu ưu tiên tiến độ).
