# Kế hoạch nâng cấp hướng tới Q1 (gửi Thầy/Cô)

**Người viết:** [Tên sinh viên]

Kế hoạch này nhắm vào các điểm yếu đã nêu ở mục 6 của `BAO_CAO_MENTOR_TONG_HOP.md`. Không có gì bảo đảm bài lên Q1; mục tiêu là nâng chất lượng bằng chứng và có điểm dừng rõ ràng để không tốn công vô ích. Đặt tên phần việc mới là **Pha U**.

## Tổng quan

| Hướng | Mục tiêu | Thời gian | Tác động lên Q1 | Rủi ro |
|---|---|---|---|---|
| A. Sửa domain gap | Mạng học sẵn cho kết quả đúng trên dữ liệu thật | 2–3 tuần | **Rất cao** | Trung bình–cao |
| B. Chính thức hoá toán học | Mệnh đề + chứng minh cho 2 kết quả đã có | ~1 tuần (song song A) | Cao | Thấp |
| C. So sánh trực diện TREET | Đặt cạnh benchmark của họ ở N nhỏ | 1–2 tuần | Cao | Trung bình (phụ thuộc mã nguồn/công bố) |
| D. Mở rộng bằng chứng | Nhiều seed, thêm họ dữ liệu, trễ dài hơn, câu hỏi lâm sàng | ~2 tuần | Trung bình–cao | Thấp |
| E. Hoàn thiện kỹ thuật | Kiểm định thống kê, script tái lập, quản lý file kết quả | ~1 tuần | Trung bình | Thấp |
| Viết bản thảo | Sau khi có kết quả A–D | 3–4 tuần | — | — |

Tổng khoảng 8–10 tuần đến bản thảo hoàn chỉnh. Phần lượng tử **không** đầu tư thêm, giữ như phụ lục.

## A. Sửa domain gap (hướng quyết định)

**Vấn đề:** mạng học sẵn huấn luyện trên dữ liệu mô phỏng fail trên dữ liệu thật (TE hô hấp→tim âm, sai hướng); dữ liệu thật nằm ngoài vùng biểu diễn của mạng.

**Ý tưởng chính:** chặn dưới Donsker–Varadhan không cần nhãn TE thật, nên có thể tinh chỉnh mạng trên chính tín hiệu thật.

**Việc cụ thể:**
1. **A1 — Thăm dò nhanh (vài ngày):** tinh chỉnh `T_φ` bằng DV loss trên một phần bản ghi thật, đánh giá trên các bản ghi giữ lại (chia theo bản ghi để không rò rỉ). Xem hướng TE có đúng không, có tương quan với KSG theo từng bản ghi không.
2. **A2 — Domain randomization:** mở rộng corpus mô phỏng cho giống tín hiệu thật hơn (chuỗi dao động dải hẹp, nhiễu không Gaussian, nhiều hệ số tự hồi quy), huấn luyện lại, đánh giá như A1.
3. **A3 — Đo gap định lượng:** thêm một số đo khoảng cách trong không gian đặc trưng (trước/sau khi sửa) bên cạnh hình PCA hiện có.
4. **A4 — Kiểm chứng đầy đủ:** k-fold trên 23 bản ghi (mỗi fold giữ lại bản ghi để đánh giá), báo cáo hướng TE, khoảng tin cậy, Wilcoxon theo bản ghi, so với KSG.

**Không có TE thật trên dữ liệu thật**, nên tiêu chí thành công là: đúng hướng RSA, đồng thuận với KSG theo từng bản ghi, và gap giảm rõ.

**Điểm dừng G1 (sau A1, ~1 tuần):** nếu thích nghi không cải thiện gì rõ rệt, dừng hướng A, giữ định vị Q2 và chuyển sang B–E. Nếu cải thiện, đi tiếp A2–A4.

**Rủi ro:** thích nghi có thể làm mạng "học lại" KSG mà không thêm giá trị; hoặc cải thiện không ổn định giữa các bản ghi. Cách giảm: chia theo bản ghi nghiêm ngặt, báo cáo cả bản ghi kém.

## B. Chính thức hoá toán học

1. **Mệnh đề 1:** nếu hàm thống kê tách rời cộng tính giữa biến nguồn và biến đích thì chặn dưới DV tối ưu luôn ≤ 0. Đã có chứng minh Jensen và đã kiểm chứng thực nghiệm (mô hình Fourier sụp đổ đúng về 0); cần viết lại chặt chẽ, nêu điều kiện.
2. **Mệnh đề 2:** phương sai của TE = mi_full − mi_reduced bằng σ²_f + σ²_r − 2ρσ_fσ_r. Dùng chung mạng làm tăng ρ và giảm phương sai; đã đo ρ = 0.89 (MLP) và 0.79 (mạch lượng tử), kiểm tra công thức khớp số liệu đo.
3. Phân tích ngắn về điểm giao cắt N theo loại động lực học (giải thích vì sao ≈30 tuyến tính, ≈50 phi tuyến), nếu làm được ở mức lập luận hợp lý.

## C. So sánh trực diện với TREET/TENDE

1. **Kết quả kiểm tra mã nguồn (S1b — Đã hoàn thành 2026-09-30):**
   - Đã clone và kiểm tra repository chính thức của TREET (`https://github.com/omerlux/TREET`, MIT License). Repo chỉ chứa code cho AWGN, GMA1, GAR1 và dữ liệu Apnea Santa Fe; module benchmark tổng hợp (mục V-A) không được công khai.
   - TENDE (arXiv 2510.14096) không có mã nguồn mở đi kèm.
2. **Quyết định phương thức so sánh:**
   - Thực hiện so sánh đối chiếu gián tiếp với số liệu công bố của cả hai bài (đã tổng hợp đầy đủ trong `docs/so_sanh.md`).
   - Nhấn mạnh điểm cốt lõi của đề tài: TREET và TENDE chỉ tối ưu và kiểm định ở vùng mẫu lớn ($T \ge 500 - 50.000$), trong khi đề tài giải quyết trúng khoảng trống mẫu nhỏ ($N = 10 - 200$) và khả năng thích nghi miền không nhãn (Unsupervised Adaptation) trên tín hiệu y sinh thực tế.

## D. Mở rộng bằng chứng

1. Nhiều seed (≥5) và khoảng tin cậy cho mọi kết quả mô phỏng chính.
2. Thêm họ dữ liệu mô phỏng: phi tuyến khác, không Gaussian.
3. Trễ dài hơn (lịch sử k>1): đổi đầu vào mạng, kiểm tra kết luận có còn.
4. **Câu hỏi lâm sàng:** TE hô hấp→tim khác nhau giữa nhóm trẻ và già (Fantasia có hai nhóm), hoặc giữa người bình thường và nhóm có ngưng thở (Apnea-ECG). Đây là điểm cho bài có ý nghĩa vượt khỏi "bài kỹ thuật ước lượng". Cần kiểm tra cỡ mẫu có đủ không trước khi cam kết.

## E. Hoàn thiện kỹ thuật

1. Kiểm định ý nghĩa thống kê cho các so sánh sát nhau (Hybrid so với thành phần; lượng tử so với KSG).
2. Script tái lập một lệnh cho mọi bảng/hình của bài.
3. Quyết định quản lý `results/` (hiện bị `.gitignore`, hình/bảng chưa vào lịch sử git).
4. Rà lại test, tài liệu, cố định phiên bản thư viện.

## Tiến độ dự kiến

| Tuần | Việc |
|---|---|
| 1 | E (kiểm định thống kê), A1 (thăm dò) → **điểm dừng G1**; kiểm tra mã nguồn TREET |
| 2–3 | A2–A3 (nếu qua G1); B song song |
| 4–5 | A4; C |
| 6–7 | D |
| 8–10 | Viết bản thảo, rà soát số liệu |

## Điểm quyết định cần Thầy/Cô cho ý kiến

- Có đồng ý thử hướng A với điểm dừng G1 không (chi phí thử ~1 tuần).
- Ưu tiên hướng D.4 (câu hỏi lâm sàng) hay không, và nhóm nào Thầy/Cô thấy có ý nghĩa nhất.
- Chấp nhận phần lượng tử ở dạng phụ lục kết quả âm.
