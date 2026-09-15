# Minh chứng trực quan quá trình thực nghiệm

---

## Giai đoạn 1 — Kiểm chứng phương pháp MINE cổ điển

![Loss huấn luyện smoke test MINE](../results/figures/phase_q_mine_smoke_test_loss.png)

Đây là bước kiểm tra đầu tiên: huấn luyện một mạng nơ-ron nhỏ theo phương pháp
MINE (Mutual Information Neural Estimation) trên một bài toán đã biết trước
đáp án, để chắc chắn cách cài đặt (loss, tối ưu hoá) hoạt động đúng trước khi
mở rộng thành phiên bản "học sẵn" (amortized) quy mô lớn hơn.

---

## Giai đoạn 2 — Huấn luyện mạng ước lượng học sẵn (amortized)

![Loss huấn luyện Amortized (mạng chính)](../results/figures/phase_r_amortized_training_loss.png)

Đường cong hội tụ khi huấn luyện mạng chính trên một corpus lớn các cặp tín
hiệu tuyến tính có ghép nối khác nhau — mạng học hội tụ mượt, không có dấu hiệu
bất thường trong quá trình tối ưu.

![Phương sai theo N](../results/figures/phase_r_variance_vs_n.png)

So sánh phương sai của 4 phương pháp (KSG, Binning, Symbolic, mạng học sẵn)
theo kích thước mẫu N trên dữ liệu tuyến tính. Đây là hình cốt lõi trả lời câu
hỏi nghiên cứu chính: mạng học sẵn có phương sai thấp hơn rõ rệt từ một ngưỡng N
nhất định trở lên.

![Bias theo N](../results/figures/phase_r_bias_vs_n.png)

Cùng phép so sánh nhưng theo độ lệch (bias). Hình này cho thấy sự khác biệt bản
chất giữa hai nhóm phương pháp: KSG có độ lệch giảm dần về 0 khi N tăng (đúng lý
thuyết), còn mạng học sẵn có một độ lệch không đổi (do năng lực mô hình cố định
sau khi huấn luyện) — đây là cơ sở cho hướng cải thiện bằng hiệu chỉnh độ lệch ở
giai đoạn sau.

### Thí nghiệm đối chứng: mạng nhỏ hơn

![Loss huấn luyện mạng nhỏ (ablation)](../results/figures/phase_r_ablation_small_loss.png)
![So sánh phương sai mạng lớn vs nhỏ](../results/figures/phase_r_ablation_variance_comparison.png)

Trước khi cân nhắc hướng đi lượng tử (mạch tính toán có rất ít tham số), em đã
làm một thí nghiệm đối chứng cổ điển: huấn luyện lại mạng với số tham số nhỏ
hơn nhiều, để kiểm tra xem việc giảm tham số có tự giúp cải thiện vùng N nhỏ
hay không. Kết quả giúp em xác định đúng nguyên nhân cần giải quyết trước khi
đầu tư công sức vào phần lượng tử.

### Thí nghiệm đối chứng: dữ liệu phi tuyến

![Loss huấn luyện periodic coupling](../results/figures/phase_r2_periodic_training_loss.png)
![Phương sai theo N — periodic coupling](../results/figures/phase_r2_periodic_variance_vs_n.png)

Lặp lại toàn bộ phép so sánh trên một dạng dữ liệu khó hơn — ghép nối phi tuyến
theo chu kỳ, thay vì tuyến tính — để kiểm tra kết luận có còn đúng trên bài
toán không thuận lợi cho các phương pháp hình học như KSG hay không. Kết quả:
điểm chuyển giao (ngưỡng N mà mạng học sẵn bắt đầu thắng) dịch sang một giá trị
N lớn hơn so với dữ liệu tuyến tính — một phát hiện quan trọng dùng để thiết kế
quy tắc chọn phương pháp kết hợp ở giai đoạn sau.

---

## Giai đoạn 3 — Kiểm định trên dữ liệu sinh lý thật

![Tín hiệu đầu vào thật](../results/figures/phase_t_input_output_signals.png)

Ví dụ tín hiệu thật (bản ghi Fantasia) sau khi tiền xử lý: chuỗi khoảng cách
nhịp tim (RR) và chuỗi hô hấp đã lọc, cùng cách chia thành các cửa sổ 30 giây
dùng cho tính toán — minh họa cụ thể dạng dữ liệu đầu vào thực tế của mô hình.

![TE hai chiều, khoảng tin cậy 95%](../results/figures/phase_s_bidirectional_te.png)

Kết quả chính trên dữ liệu thật: so sánh TE theo hai chiều (hô hấp→tim và
tim→hô hấp) trên toàn bộ bản ghi, kèm khoảng tin cậy. Khoảng tin cậy của hiệu số
nằm hoàn toàn trên 0, cho thấy chiều hô hấp→tim chiếm ưu thế có ý nghĩa thống
kê — phù hợp với hiểu biết sinh lý học đã có.

---

## Giai đoạn 4 — Tổng hợp và cải tiến phương pháp

![Kết quả tổng hợp phương pháp học sẵn/kết hợp theo N](../results/figures/phase_t_main_variance_vs_n.png)

Hình tổng hợp cuối cùng, gộp cả dữ liệu tuyến tính và phi tuyến, thêm phương
pháp kết hợp (hybrid — tự chuyển giữa KSG và mạng học sẵn theo N). Phương pháp
kết hợp luôn bám theo đường có phương sai thấp nhất ở mọi kích thước mẫu.

![So sánh MSE trước/sau hiệu chỉnh độ lệch](../results/figures/phase_t_bias_calibration_mse.png)

Kết quả của một cải tiến thêm: sau khi hiệu chỉnh độ lệch cho mạng học sẵn (ước
lượng độ lệch từ các cấu hình dữ liệu khác, không dùng chính dữ liệu đang đánh
giá — tránh đánh giá quá lạc quan), phương pháp học sẵn thắng cả về sai số toàn
phần (MSE), không chỉ về phương sai, trên phần lớn miền N đã kiểm định.

---

## Tổng số hình: 12, trải đều qua toàn bộ quá trình thực nghiệm (từ kiểm chứng phương pháp cơ bản đến kết quả trên dữ liệu thật).
