# Báo cáo pha 1

---

## 1. Câu hỏi nghiên cứu

Các phương pháp ước lượng transfer entropy (TE) cổ điển như KSG (k-nearest
neighbor), binning, hay symbolic đều cần số mẫu N đủ lớn mới cho kết quả ổn
định, vì chúng phải ước lượng lại mật độ xác suất từ đầu cho mỗi cặp tín hiệu
mới. Trong khi đó, tín hiệu sinh lý thực tế (nhịp tim, hô hấp, điện não) thường
chỉ có cửa sổ quan sát ngắn — do bản chất không dừng của tín hiệu, hoặc do yêu
cầu theo dõi thời gian thực trên thiết bị đeo.

Câu hỏi đặt ra: **liệu một mạng nơ-ron học sẵn một lần (amortized estimator,
theo hướng MINE) có thể thay thế các phương pháp cổ điển ở đúng vùng N nhỏ, và
điều đó có còn đúng khi áp dụng lên dữ liệu sinh lý thật (không chỉ dữ liệu mô
phỏng)?**

---

## 2. Đầu vào của bài toán

Trình bày đầu vào theo 3 lớp, từ dữ liệu thô đến dạng thực sự đưa vào mô hình.

### 2.1. Dữ liệu thô

Sử dụng 2 nhóm dữ liệu:

- **Dữ liệu mô phỏng (synthetic):** sinh từ mô hình toán học đã biết trước giá
  trị TE thật (một chuỗi tuyến tính dạng VAR, và một chuỗi phi tuyến dạng ghép
  nối tuần hoàn). Mục đích: có "đáp án" để kiểm tra độ chính xác của từng
  phương pháp ước lượng.
- **Dữ liệu thật:** lấy từ 2 cơ sở dữ liệu công khai trên PhysioNet — **Fantasia**
  và **Apnea-ECG** — mỗi bản ghi gồm tín hiệu điện tim (ECG) và tín hiệu hô hấp
  đo đồng thời trên cùng một người.

### 2.2. Sau tiền xử lý

Từ ECG thô, tách ra chuỗi khoảng cách giữa các nhịp tim liên tiếp (RR
interval); từ tín hiệu hô hấp thô, lọc bằng-thông trong dải tần số hô hấp
người lớn bình thường (6–30 lần/phút). Cả hai chuỗi được đưa về cùng một lưới
thời gian rời rạc (4 mẫu/giây) để đảm bảo tại mỗi chỉ số mẫu, hai chuỗi tương
ứng đúng một thời điểm thực tế.

Hình dưới là ví dụ thật (bản ghi Fantasia `f1o01`, 3 phút đầu): chuỗi RR (đỏ) và
chuỗi hô hấp (xanh) sau khi lọc — có thể thấy cả hai dao động theo cùng nhịp hô
hấp, đây chính là hiện tượng sinh lý (rối loạn nhịp xoang theo hô hấp — RSA) mà
em muốn đo hướng ảnh hưởng. Panel dưới minh họa cách chia dữ liệu thành các cửa
sổ 30 giây (120 mẫu) không chồng lấn — đây là đơn vị tính toán thực sự.

![Tín hiệu đầu vào](../results/figures/phase_t_input_output_signals.png)

### 2.3. Dạng đưa vào mô hình ước lượng

Với mỗi cửa sổ dữ liệu, mô hình nhận vào một cặp chuỗi con: chuỗi "quá khứ" của
tín hiệu nguồn và chuỗi "quá khứ + hiện tại" của tín hiệu đích, để tính một
chiều TE. Đổi vai trò nguồn/đích cho nhau sẽ cho chiều ngược lại — vì vậy mỗi
cửa sổ luôn sinh ra hai giá trị, một cho mỗi chiều nhân quả.

---

## 3. Đầu ra của bài toán

Đầu ra cũng chia theo 3 mức tổng hợp:

1. **Một cửa sổ:** một số thực (đơn vị nats) — lượng thông tin về tương lai của
   tín hiệu đích được giải thích thêm nhờ biết quá khứ của tín hiệu nguồn.
2. **Một bản ghi:** trung bình các giá trị TE qua toàn bộ cửa sổ của bản ghi đó,
   cho ra 2 số — TE theo chiều hô hấp→tim và TE theo chiều tim→hô hấp.
3. **Toàn bộ tập dữ liệu:** khoảng tin cậy 95% (bootstrap) của hiệu số giữa hai
   chiều, cùng kiểm định Wilcoxon signed-rank, tính trên đơn vị mẫu là **bản
   ghi** (không phải cửa sổ, để tránh đánh giá quá lạc quan do các cửa sổ trong
   cùng một bản ghi không độc lập với nhau).

**Kết quả thực tế đã thu được** (gộp 23 bản ghi từ cả hai nguồn dữ liệu,
dùng phương pháp KSG):

| | TE(hô hấp→tim) | TE(tim→hô hấp) | Khoảng tin cậy 95% của hiệu số | p (Wilcoxon) |
|---|---|---|---|---|
| Kết quả | 0.1175 nats | 0.0996 nats | [0.0057, 0.0311] | 0.0046 |

Khoảng tin cậy hoàn toàn nằm trên 0 và p < 0.01, cho thấy chiều hô hấp→tim mạnh
hơn có ý nghĩa thống kê — phù hợp với hiểu biết sinh lý học đã được công bố về
RSA (Grossman & Taylor, 2007).

![Kết quả hai chiều TE trên dữ liệu thật](../results/figures/phase_s_bidirectional_te.png)

Với dữ liệu mô phỏng, vì đã biết trước giá trị TE thật, đầu ra ở đây không phải
kết luận sinh lý học mà là **độ lệch (bias) và phương sai (variance)** của mỗi
phương pháp so với giá trị đúng — dùng để so sánh chất lượng các phương pháp
theo kích thước mẫu N.

![So sánh chất lượng ước lượng theo N](../results/figures/phase_t_main_variance_vs_n.png)

---

## 4. Sơ đồ tổng quát

```
Dữ liệu thô (ECG + hô hấp, thật hoặc mô phỏng)
        │  tách nhịp tim, lọc tần số, đồng bộ về 1 lưới thời gian chung
        ▼
Hai chuỗi thời gian đã xử lý (RR, hô hấp)
        │  chia cửa sổ N mẫu
        ▼
Mô hình ước lượng TE (KSG / Binning / Symbolic / mạng học sẵn / kết hợp)
        │  1 giá trị TE cho mỗi cửa sổ, mỗi chiều
        ▼
Trung bình theo bản ghi  →  kiểm định thống kê trên toàn bộ mẫu
        ▼
Kết luận: chiều ghép nối nào chiếm ưu thế / phương pháp nào chính xác hơn ở N nào
```

---

## 5. Khoảng trống nghiên cứu

5 khoảng trống cụ thể:

**(1) Chưa có đánh giá định lượng rõ ràng về việc phương pháp học sẵn thắng
phương pháp cổ điển ở đúng vùng N nào.** Nhiều nghiên cứu chỉ nêu định hướng
chung ("mạng học sẵn phù hợp với mẫu nhỏ") mà không đo cụ thể ngưỡng N. Em đã đo
trực tiếp trên cả dữ liệu tuyến tính và phi tuyến để xác định điểm chuyển giao.

**(2) Điểm chuyển giao này không cố định mà phụ thuộc vào bản chất động lực học
của dữ liệu** (tuyến tính hay phi tuyến) — một ngưỡng chọn phương pháp cố định
áp dụng chung cho mọi loại dữ liệu có thể dẫn đến chọn sai phương pháp ở một số
vùng N cụ thể. Em đã thiết kế một quy tắc chọn phương pháp kết hợp (hybrid) với
ngưỡng an toàn chung cho cả hai loại dữ liệu đã kiểm định.

**(3) Vấn đề khái quát hóa (domain generalization) của mạng học sẵn — huấn
luyện trên dữ liệu mô phỏng có dùng được cho dữ liệu sinh lý thật không —
thường bị bỏ qua trong các nghiên cứu chỉ báo cáo kết quả trên dữ liệu mô
phỏng.** Khi em kiểm tra trực tiếp, mạng học sẵn (chưa hiệu chỉnh) cho kết quả
sai cả về dấu trên dữ liệu thật, trong khi KSG vẫn cho kết quả đúng hướng. Em
đã truy tìm nguyên nhân đến tận gốc (phân rã các thành phần thông tin trong hàm
mất mát) để hiểu rõ đây là giới hạn thật của phương pháp khi dữ liệu thật lệch
khỏi phân phối đã huấn luyện, chứ không phải lỗi cài đặt.

**(4) Vấn đề thống kê: nhiều nghiên cứu về nhịp tim–hô hấp gộp tất cả các cửa sổ
quan sát như các mẫu độc lập, dẫn đến đánh giá độ tin cậy quá lạc quan.** Em đã
sửa lại theo đúng đơn vị mẫu là bản ghi (không phải cửa sổ), cho khoảng tin cậy
rộng hơn nhưng đáng tin cậy hơn — và kết luận vẫn giữ được ý nghĩa thống kê.

**(5) Hướng mở cho giai đoạn tiếp theo của đề tài:** liệu một mạch tính toán
lượng tử với rất ít tham số có giúp cải thiện đúng vùng N nhỏ mà mạng cổ điển
còn yếu, hay hạn chế nằm ở bản chất thống kê của bài toán chứ không phải ở số
lượng tham số mô hình? Em đã làm một thực nghiệm đối chứng bằng mạng cổ điển
thu nhỏ để có cơ sở trả lời câu hỏi này trước khi đầu tư vào phần lượng tử.

---

## 6. Tóm tắt

| | Đầu vào | Đầu ra | Câu hỏi trả lời |
|---|---|---|---|
| Dữ liệu mô phỏng | Hai chuỗi tín hiệu mô phỏng, biết trước TE thật | Độ lệch/phương sai của từng phương pháp theo N | Phương pháp nào chính xác nhất ở kích thước mẫu nào? |
| Dữ liệu thật | RR + hô hấp đồng bộ, 23 bản ghi từ 2 nguồn độc lập | TE hai chiều mỗi bản ghi → kiểm định thống kê gộp | Chiều ghép nối tim–hô hấp nào chiếm ưu thế? |
| Khoảng trống chính | — | — | Mạng học sẵn thắng về phương sai ở N lớn nhưng chưa khái quát hóa được sang dữ liệu thật; KSG hiện vẫn là lựa chọn an toàn cho dữ liệu thật, còn cách kết hợp phương pháp và hiệu chỉnh độ lệch là hai hướng cải thiện em đã kiểm định trên dữ liệu mô phỏng |
