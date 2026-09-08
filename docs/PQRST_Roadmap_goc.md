# Lộ Trình PQRST — Hai Nhịp

**Kế hoạch triển khai Q-BHC / AQNE-TE — hướng tới công bố Q1–Q2**
**Phiên bản 2.0 — tách theo lời khuyên của mentor: làm bản DL cổ điển trước, tích hợp lượng tử sau**

> Một nhịp tim không chỉ có một chu kỳ PQRST — trái tim khoẻ mạnh đập lặp lại. Lộ trình này cũng vậy: **Nhịp 1** chạy trọn một chu kỳ P→T bằng mạng nơ-ron cổ điển (MINE gốc, đã có recipe đáng tin, không đụng đến lượng tử). **Nhịp 2** lặp lại đúng cấu trúc P→T đó, nhưng thay bộ ước lượng cổ điển bằng mạch lượng tử — một phép hoán đổi có kiểm soát, không phải làm lại từ đầu.

**Ngày lập:** 08/09/2026 · **Lý do sửa:** mentor khuyến nghị "chưa cần quantum vội, cứ DL đã rồi tích hợp" — tách rủi ro hội tụ mạch lượng tử ra khỏi rủi ro của toàn bộ phương pháp.

---

## Vì sao tách nhịp là đúng, không phải làm chậm dự án

Rủi ro lớn nhất của bản kế hoạch trước nằm ở Pha Q/R cũ: đặt cược ngay từ đầu vào việc một mạch lượng tử **chưa ai huấn luyện trước** có hội tụ hay không. Nếu không, mọi thứ phía sau (pipeline dữ liệu thật, sanity check, thí nghiệm chính) đều phải chờ.

Tách thành hai nhịp giải quyết đúng vấn đề này:

- **Nhịp 1 dùng MINE cổ điển** (Belghazi et al., 2018) — mạng nơ-ron MLP thường, huấn luyện bằng backprop qua bound Donsker–Varadhan. Đây là công thức **đã được cộng đồng dùng và kiểm chứng rộng rãi từ 2018** — rủi ro hội tụ gần như bằng không so với mạch lượng tử.
- Toàn bộ phần khó về *phương pháp luận* (mở rộng sang conditional MI/TE, kho dữ liệu tổng hợp, pipeline 4 bộ dữ liệu thật, sanity check, thiết kế thống kê) được **validate xong bằng bản cổ điển** trước khi lượng tử tham gia.
- **Nhịp 2 chỉ hoán đổi một khối** — thay mạng MLP bằng mạch lượng tử data re-uploading, giữ nguyên 100% phần còn lại (cùng kho dữ liệu tổng hợp, cùng hàm loss, cùng 4 bộ dữ liệu thật, cùng quy trình đánh giá). Đây là một **ablation có kiểm soát** (apples-to-apples), không phải xây lại — rủi ro thấp hơn hẳn, và về mặt khoa học còn *chặt hơn* bản kế hoạch cũ vì giờ có một đối chứng cổ điển thật sự công bằng.
- **Lợi ích phụ:** nếu Nhịp 2 gặp khó khăn hoặc hết thời gian, Nhịp 1 tự nó đã là một bản thảo hoàn chỉnh, khả dĩ nộp độc lập ("bộ ước lượng transfer entropy amortized cho ghép nối tim–não, huấn luyện bằng mạng nơ-ron cổ điển"). Không có kịch bản nào khiến công sức bị bỏ phí hoàn toàn.

---

## Bản đồ tổng quan

| Nhịp | Pha | Trọng tâm | Ước tính |
|---|---|---|---|
| **1 — Cổ điển** | P | Hạ tầng, baseline, dữ liệu tổng hợp | 1–1,5 tuần |
| | Q | MINE cổ điển (MLP + Donsker–Varadhan) chạy được | 1–2 tuần |
| | R | Mở rộng conditional MI/TE, kiểm định đầy đủ trên synthetic | 2–3 tuần |
| | S | Pipeline 4 bộ dữ liệu thật + sanity check | 2–3 tuần |
| | T | Thí nghiệm chính + bản thảo cổ điển (checkpoint) | 2–3 tuần |
| **2 — Lượng tử** | P′ | Dựng mạch data re-uploading, môi trường PennyLane/Qiskit | 3–5 ngày |
| | Q′ | Hoán đổi MLP → QNN, chạy lại trên synthetic đơn giản | 1,5–3 tuần |
| | R′ | Kiểm định QNN đầy đủ trên synthetic, so trực tiếp với MLP (Nhịp 1) | 2–3 tuần |
| | S′ | Áp QNN lên 4 bộ dữ liệu thật, lặp lại sanity check | 1–2 tuần |
| | T′ | Thí nghiệm cuối (cổ điển vs lượng tử) + bản thảo hoàn chỉnh | 2–3 tuần |

**Nhịp 1: ~8–12 tuần · Nhịp 2: ~5–8 tuần (nhờ tái dùng hạ tầng) · Tổng: ~13–20 tuần (≈3,25–5 tháng)**

---

# NHỊP 1 — Chu kỳ PQRST cổ điển

## Pha P — Khởi động

**Mục tiêu:** hạ tầng và dữ liệu sẵn sàng trước khi chạm vào mô hình.

- Cài môi trường: PyTorch, IDTxl/JIDT.
- Bộ sinh dữ liệu tổng hợp: VAR tuyến tính Gaussian, VAR phi tuyến, ghép nối tuần hoàn — có TE ground-truth biết trước hoặc hội tụ đáng tin ở N lớn.
- Cài ba baseline cổ điển (KSG, symbolic TE, binning) qua thư viện đã kiểm định.

**Tiêu chí thoát:** ba baseline chạy đúng trên dữ liệu tổng hợp, khớp ground-truth trong sai số chấp nhận được.

## Pha Q — MINE cổ điển

**Mục tiêu:** vòng lặp huấn luyện MLP qua bound Donsker–Varadhan chạy đúng — bước này **có rất nhiều tài liệu/mã nguồn tham khảo công khai** (MINE gốc 2018 có implementation phổ biến), nên là bước ít rủi ro nhất trong toàn bộ dự án.

- Mạng thống kê `T_φ`: MLP 2–3 lớp ẩn, nhận đầu vào là đặc trưng cửa sổ (X, Y).
- Loss Donsker–Varadhan + cơ chế shuffle-batch chuẩn MINE.
- Huấn luyện Adam, chạy trên một cấu hình tổng hợp đơn giản trước.

**Tiêu chí thoát:** loss giảm ổn định, `T_φ` ước lượng đúng MI trên bài toán tổng hợp có đáp án biết trước (sai số nhỏ so với ground-truth).

## Pha R — Mở rộng & kiểm định đầy đủ

**Mục tiêu:** `T_φ` tổng quát hoá tốt trên toàn bộ không gian cấu hình (nhiều mức ghép nối, nhiều mức nhiễu, N từ 10 đến 200), và mở rộng được sang conditional MI.

- Sinh kho ngữ liệu đầy đủ (≥20.000 cửa sổ train, ≥2.000 kiểm định tách bạch).
- Mở rộng loss sang conditional MI qua `I(A;B|C) = I(A;B,C) − I(A;C)`, dùng chung `T_φ`.
- Huấn luyện mini-batch trộn đa cấu hình; đánh giá bias/phương sai trên tập kiểm định tách bạch, so với ba baseline cổ điển ở Pha P.

**Tiêu chí thoát:** trên tập kiểm định tổng hợp, `T_φ` có phương sai thấp hơn KSG/symbolic TE **ở vùng N nhỏ (N < 30)** — đây là bằng chứng cốt lõi cho toàn bộ luận điểm của dự án, và giờ được xác lập bằng mô hình cổ điển an toàn trước khi lượng tử tham gia.

## Pha S — Dữ liệu thật

**Mục tiêu:** `T_φ` đã đóng băng chạy ổn định trên bốn bộ dữ liệu sinh lý thật.

- Pipeline tiền xử lý cho Fantasia, Apnea-ECG, MIT-BIH Polysomnographic, CAP Sleep Database: đồng bộ hoá kênh, trích đặc trưng cửa sổ, xử lý artifact.
- **Sanity check bắt buộc:** tái lập TE(tim→não) > TE(não→tim) trên MIT-BIH Polysomnographic và CAP Sleep Database.

**Tiêu chí thoát:** sanity check đạt trên ít nhất một bộ có nhãn rõ. Nếu không đạt, kiểm tra pipeline đồng bộ hoá trước khi nghi ngờ mô hình.

## Pha T — Kết quả & Bản thảo checkpoint

**Mục tiêu:** bộ kết quả cổ điển hoàn chỉnh — điểm dừng an toàn đầu tiên của cả dự án.

- Thí nghiệm chính: quét N trên bốn bộ dữ liệu, bốn phương pháp (`T_φ`, KSG, symbolic TE, binning).
- Bootstrap (≥500 lần) + permutation test.
- Viết bản thảo checkpoint — **có thể nộp độc lập nếu muốn dừng ở đây**, hoặc giữ làm nền cho Nhịp 2.

**🚦 GATE CHUYỂN NHỊP:**
- Kết quả tốt (MLP thắng rõ baseline cổ điển ở N nhỏ) → sang Nhịp 2 với câu chuyện mạnh: "cổ điển đã thắng, liệu lượng tử thắng hơn nữa không?"
- Kết quả ngang bằng → vẫn đáng sang Nhịp 2 (câu hỏi nghiên cứu chuyển thành "lượng tử có mang lại lợi thế mà cổ điển không có"), nhưng hạ kỳ vọng.
- Hết thời gian hoặc nguồn lực → dừng ở đây, nộp bản thảo checkpoint. Đây là kết quả đàng hoàng, không phải thất bại.

---

# NHỊP 2 — Chu kỳ PQRST lượng tử

*Toàn bộ hạ tầng, dữ liệu, baseline đã có từ Nhịp 1 — nhịp này chỉ hoán đổi bộ ước lượng.*

## Pha P′ — Chuẩn bị lượng tử

- Cài PennyLane/Qiskit Aer.
- Dựng mạch data re-uploading (6–8 qubit, 3–5 lớp) làm `T_θ`, thay thế vị trí của MLP `T_φ`.

**Tiêu chí thoát:** mạch chạy được, cho ra một giá trị vô hướng hợp lệ từ dữ liệu đầu vào mẫu.

## Pha Q′ — Hoán đổi có kiểm soát

- Cắm `T_θ` vào đúng vòng lặp huấn luyện Donsker–Varadhan đã có từ Pha Q (Nhịp 1) — chỉ đổi mạng, giữ nguyên loss, dữ liệu, cơ chế shuffle-batch.
- Gradient qua parameter-shift rule hoặc adjoint differentiation.
- Chạy trên cấu hình tổng hợp đơn giản nhất trước (giống hệt bước đầu của Pha Q, Nhịp 1).

**Tiêu chí thoát:** loss giảm ổn định; gradient không biến mất.

**🚦 CỔNG DỰ PHÒNG (kế thừa từ bản kế hoạch cũ):** nếu sau ≥2 cấu hình siêu tham số vẫn không hội tụ → dấu hiệu barren plateau. Vì Nhịp 1 đã là một bản thảo hoàn chỉnh, đây **không còn là rủi ro chí mạng** — có thể dừng Nhịp 2, báo cáo nỗ lực tích hợp lượng tử như một phụ lục/hướng mở trong bản thảo đã có.

## Pha R′ — Kiểm định & So sánh trực tiếp

- Kiểm định `T_θ` trên toàn bộ kho ngữ liệu tổng hợp (giống hệt phạm vi Pha R, Nhịp 1).
- **So sánh trực tiếp, cùng điều kiện:** `T_θ` (lượng tử) vs `T_φ` (MLP, Nhịp 1) vs ba baseline cổ điển — cùng dữ liệu, cùng thước đo. Đây là bảng số liệu trung tâm của cả bài báo.

**Tiêu chí thoát:** có bảng so sánh đầy đủ bốn phương pháp trên tập kiểm định tách bạch.

## Pha S′ — Dữ liệu thật với lượng tử

- Chạy lại `T_θ` trên đúng pipeline bốn bộ dữ liệu đã xây ở Pha S (Nhịp 1) — không cần xây lại.
- Lặp lại sanity check với `T_θ`.

**Tiêu chí thoát:** sanity check đạt; có kết quả `T_θ` trên cả bốn bộ, sẵn sàng đưa vào bảng so sánh cuối.

## Pha T′ — Kết quả cuối & Bản thảo hoàn chỉnh

- Thí nghiệm cuối: quét N, bootstrap, permutation test cho cả bốn phương pháp trên dữ liệu thật.
- Ablation riêng cho lượng tử: độ sâu mạch, số qubit, lựa chọn đặc trưng.
- Viết bản thảo hoàn chỉnh — nộp preprint arXiv song song nộp tạp chí.
- Đóng gói mã nguồn + hai bộ trọng số đã huấn luyện (`φ` cổ điển và `θ` lượng tử) để người khác tái lập cả hai.

**Tiêu chí thoát (= hoàn thành dự án):** bản thảo hoàn chỉnh có đầy đủ so sánh cổ điển–lượng tử, mã nguồn tái lập được từ đầu đến cuối.

---

## Tạp chí mục tiêu (không đổi)

| Tạp chí | Quartile |
|---|---|
| Chaos (AIP Publishing) | Q1 |
| IEEE J. Biomedical & Health Informatics | Q1 |
| Entropy (MDPI) | Q1 |
| Frontiers in Network Physiology | Q2 |
| Physiological Measurement | Q2 |

*Nếu dừng ở Nhịp 1 (bản cổ điển), Entropy và Frontiers in Network Physiology vẫn là điểm đến hợp lý — không nhất thiết cần yếu tố lượng tử để có giá trị công bố.*

---

## Bảng rủi ro theo nhịp

| Nhịp | Rủi ro lớn nhất | Mức độ nghiêm trọng nếu xảy ra |
|---|---|---|
| 1 — Pha Q/R | MLP không tách được tín hiệu ở N rất nhỏ | **Trung bình** — vẫn có bài báo về giới hạn của amortized estimation cổ điển |
| 1 — Pha S | Lỗi đồng bộ dữ liệu thật, sanity check thất bại | **Trung bình** — debug được, không phải giới hạn phương pháp |
| 2 — Pha Q′/R′ | QNN không hội tụ (barren plateau) | **Thấp** — đã có Nhịp 1 làm bản thảo an toàn, đây chỉ là mất cơ hội "thêm điểm", không mất trắng |
| 2 — Pha R′ | QNN hội tụ nhưng không thắng MLP | **Thấp** — đây tự nó là một phát hiện hợp lệ (định lượng khi nào lượng tử không giúp gì) |

---

*Lộ trình sửa ngày 08/09/2026 theo góp ý của mentor: tách rủi ro phương pháp luận (Nhịp 1, cổ điển, rủi ro thấp) khỏi rủi ro huấn luyện lượng tử (Nhịp 2, tích hợp có kiểm soát). Đây là kế hoạch làm việc, điều chỉnh theo tín hiệu thực tế ở từng gate.*
