# Báo cáo tổng thể gửi Thầy/Cô hướng dẫn

**Người viết:** [Tên sinh viên]

---

## 1. Bài toán đang giải

Các phương pháp ước lượng transfer entropy (TE) cổ điển (KSG, binning, symbolic)
cần số mẫu N đủ lớn mới cho kết quả ổn định. Tín hiệu sinh lý thực tế (nhịp tim,
hô hấp) thường chỉ có cửa sổ quan sát ngắn. Em xây một mạng nơ-ron học sẵn một
lần (amortized, theo hướng MINE) để thay thế các phương pháp cổ điển ở đúng vùng
N nhỏ này, và kiểm định xem điều đó có còn đúng trên dữ liệu sinh lý thật không.

## 2. Mô hình dùng

Mạng `T_φ` (statistics network): MLP **4 → 128 → 128 → 64 → 1**, hàm kích hoạt
ELU giữa các lớp ẩn — **25.473 tham số**. Huấn luyện bằng Adam (lr=0.001), tối
đa 100 epoch, trên 27.000 cửa sổ dữ liệu mô phỏng (quét 5 mức cường độ ghép nối
× 3 mức nhiễu × 6 kích thước cửa sổ N), đánh giá trên một tập test riêng 7.200
cửa sổ sinh từ seed hoàn toàn khác. Có làm thêm một bản thu nhỏ `[16,16]`
(369 tham số) để kiểm định giả thuyết "ít tham số giúp gì ở N nhỏ" trước khi
tính đến hướng lượng tử ở giai đoạn sau.

Nền tảng toán học: mạng được huấn luyện theo chặn dưới **Donsker-Varadhan** của
mutual information — nghĩa là mạng phải học phân biệt được cặp dữ liệu ghép đôi
thật (joint, từ phân phối `P(X,Y)`) với cặp dữ liệu ghép đôi ngẫu nhiên (marginal,
xáo trộn). Đây không phải một chi tiết kỹ thuật phụ — đây chính là điều kiện lý
thuyết quyết định chặn dưới có chặt hay không.

## 3. Kết quả trên dữ liệu thật

Gộp 23 bản ghi từ 2 nguồn PhysioNet độc lập (Fantasia, Apnea-ECG), dùng KSG:

| | TE(hô hấp→tim) | TE(tim→hô hấp) | CI 95% hiệu số | p (Wilcoxon) |
|---|---|---|---|---|
| Kết quả | 0.1175 nats | 0.0996 nats | [0.0057, 0.0311] | 0.0046 |

Chiều hô hấp→tim chiếm ưu thế có ý nghĩa thống kê — đúng hướng RSA đã biết trong
y văn. Kiểm định trên đơn vị mẫu là bản ghi (không phải cửa sổ) để tránh đánh
giá quá lạc quan do các cửa sổ trong cùng bản ghi không độc lập.

Tuy nhiên, mạng học sẵn (chưa hiệu chỉnh) khi áp lên đúng dữ liệu này lại cho
kết quả **sai cả hướng** — đây là phát hiện quan trọng em trình bày kỹ ở mục 5.

## 4. Mạng có học đúng cấu trúc dữ liệu không — bằng chứng từ không gian đặc trưng

Để trả lời câu hỏi này một cách trực quan (không chỉ qua vài chỉ số), em trích
xuất đặc trưng ở đầu ra của từng khối trong mạng (sau mỗi lớp Linear+ELU) và vẽ
PCA riêng cho từng khối.

**Bằng chứng 1 — mạng học đúng cấu trúc ghép nối của dữ liệu.** Tô màu các điểm
theo cường độ ghép nối thật (dữ liệu mô phỏng):

![PCA theo cường độ ghép nối](../results/figures/phase_t_pca_blocks_by_coupling.png)

Ở lớp Input các điểm gần trùng nhau (do em chuẩn hoá z-score theo từng cửa sổ
nên trung bình luôn xấp xỉ 0 — hệ quả tất nhiên, không phải bất thường). Nhưng
từ Block 1 trở đi, các điểm bắt đầu tách dần theo đúng giá trị cường độ ghép
nối, rõ nhất ở Block 3. Mạng không "học mù" — nó xây được một không gian đặc
trưng phản ánh đúng cấu trúc thật của dữ liệu.

**Bằng chứng 2 — mạng học đúng đối tượng toán học của bài toán.** Tô màu theo
joint (cặp thật) vs marginal (cặp xáo trộn) — đúng 2 loại mẫu mà lý thuyết
Donsker-Varadhan (mục 2) yêu cầu mạng phải phân biệt được:

![PCA joint vs marginal](../results/figures/phase_t_pca_blocks_joint_vs_marginal.png)

Hai nhóm tách biệt hoàn toàn rõ ràng ngay từ Block 1. Đây là bằng chứng hình học
trực tiếp rằng mạng học đúng điều lý thuyết yêu cầu, không chỉ "cho ra số đẹp"
một cách ngẫu nhiên.

**Bằng chứng 3 — nhìn thấy được "gap" ngay trong không gian đặc trưng, không chỉ
qua con số cuối.** Chiếu dữ liệu tim–hô hấp thật vào ĐÚNG không gian PCA đã học
từ dữ liệu mô phỏng (không train lại, không fit lại PCA):

![Domain gap](../results/figures/phase_t_pca_domain_gap.png)

Dữ liệu thật (tam giác đỏ) nằm **tách biệt hoàn toàn** khỏi vùng dữ liệu mô
phỏng ở mọi khối, càng rõ ở khối sâu. Đây chính là nguyên nhân hình học cho kết
quả sai hướng ở mục 3: dữ liệu thật rơi ra ngoài hẳn "vùng hiểu biết" mà mạng đã
học được, nên biểu diễn nội tại — và do đó kết quả ước lượng cuối cùng — không
còn đáng tin ở vùng đó.

## 5. Đóng góp của đề tài và vị trí so với các nghiên cứu khác

Trong lúc rà lại tài liệu, em thấy có 2 bài rất gần hướng của mình, cả hai đều
**đã được công bố có bình duyệt** (đã kiểm tra lại): **TREET** (Transfer Entropy
Estimation via Transformers — đã đăng tại IEEE Access, cũng thử nghiệm trên dữ
liệu Apnea) và **TENDE** (Transfer Entropy Neural Diffusion Estimation — đã
được nhận tại AISTATS 2026). Cả hai cũng dùng mạng nơ-ron ước lượng TE và test
trên dữ liệu thật, nên em không thể coi mình là "đầu tiên" theo hướng này.

Điểm khác của em: TREET và TENDE tập trung chứng minh estimator của họ chính
xác hơn các phương pháp khác (góc nhìn "ai thắng ai"). Em đi theo hướng khác —
**phân tích không gian đặc trưng để chẩn đoán khi nào mô hình còn tin được**
(mục 4), gắn trực tiếp với cơ sở lý thuyết Donsker-Varadhan mà phương pháp dựa
vào, và **chủ động phát hiện, lý giải tận gốc một trường hợp thất bại thật**
(domain generalization gap) thay vì chỉ báo cáo kết quả tốt. Em cho rằng đây là
một hướng đóng góp bổ sung cho, chứ không cạnh tranh trực tiếp với, 2 bài trên —
và sẽ trích dẫn/định vị rõ trong phần Related Work của bản thảo.

## 6. Việc còn lại

Toàn bộ phần thực nghiệm và phân tích ở trên đã hoàn tất và được kiểm tra lại
độc lập (chạy lại notebook, đối chiếu số liệu). Việc còn lại là viết bản thảo
đầy đủ (Abstract → Conclusion) và review đối chiếu số liệu lần cuối trước khi
nộp — em xin ý kiến thầy/cô về hướng bài (Entropy hay Frontiers in Network
Physiology) trước khi bắt đầu viết.
