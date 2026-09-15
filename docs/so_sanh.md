# So sánh với các nghiên cứu khác & Kiểm tra rò rỉ dữ liệu
---

## 1. So sánh với các nghiên cứu khác (có benchmark số liệu thật)

### 1.1. Vùng kích thước mẫu N — điểm khác biệt rõ nhất

**cả TREET và TENDE đều không kiểm định ở vùng N nhỏ (10–200) mà đề tài của tập trung vào.**

| | Vùng N chính dùng để benchmark | Vùng N nhỏ nhất từng test |
|---|---|---|
| TREET | Không nêu rõ T cụ thể, nhưng kết luận chính là "đạt độ chính xác tốt nhất khi memory order l > 100" — nghĩa là cần chuỗi đủ dài | Không có benchmark cho N<100 |
| TENDE | T = 10.000 mẫu (thí nghiệm chính), thêm 50.000 mẫu (không gian nhiều chiều) | T = 500 (nhỏ nhất từng test) |
| **Đề tài đang làm** | **N = 10–200** | **N = 10** |

Nói cách khác: TREET/TENDE tối ưu cho vùng **mẫu lớn** (hàng nghìn đến hàng chục
nghìn điểm), còn đúng khoảng trống N=10-50 mà tín hiệu sinh lý thực tế thường
gặp (cửa sổ ngắn, không dừng) **chưa được 2 bài này kiểm định** — đây là vùng
đề tài đang làm đóng góp trực tiếp, không trùng lặp với 2 bài trên.

### 1.2. Kết quả trên dữ liệu tim–hô hấp thật — so được trực tiếp

Cả TREET và TENDE đều test trên **cùng 1 bộ dữ liệu** (Santa Fe Time Series
Competition, 1 bệnh nhân ngưng thở khi ngủ, nhịp tim + thể tích lồng ngực,
2 Hz) — không phải Fantasia/Apnea-ECG, nhưng cùng loại câu hỏi (hướng
ghép nối hô hấp–tim):

| | TE(hô hấp→tim) | TE(tim→hô hấp) | Tỷ lệ | Cỡ mẫu |
|---|---|---|---|---|
| TREET (k=4, l=2) | ≈0.012 nats | ≈0.002 nats | **≈6 lần** | 1 bệnh nhân |
| TENDE | — | — | **≈2–3 lần** (nêu trong bài) | 1 bệnh nhân |
| **Đề tài đang làm (KSG)** | 0.1175 nats | 0.0996 nats | **≈1.18 lần** | 23 bản ghi, 2 nguồn độc lập |

Cả 3 đều **cùng hướng** (hô hấp→tim chiếm ưu thế) — khớp nhau về mặt sinh lý
học. Nhưng tỷ lệ đang làm nhỏ hơn hẳn — cho rằng có 2 lý do hợp lý: (1) TREET/
TENDE đo trên **1 bệnh nhân ngưng thở khi ngủ** (bệnh lý, hiệu ứng ngưng thở có
thể phóng đại chênh lệch hướng), còn đang làm đo trên **23 bản ghi gộp từ người khỏe
mạnh (Fantasia) + người có nguy cơ ngưng thở (Apnea-ECG)** — quần thể khác, kết
luận tổng quát hơn nhưng hiệu ứng trung bình nhỏ hơn; (2) cách chọn trễ khác —
TREET/TENDE dò nhiều giá trị trễ `k`, em dùng trễ 1 bước cố định trên lưới 4 Hz.
Em sẽ nêu rõ điểm này trong Discussion, không so trực tiếp độ lớn mà chỉ so
hướng.

*Nguồn (đọc trực tiếp bản đầy đủ):* [TREET (arXiv 2402.06919)](https://arxiv.org/html/2402.06919v3), [TENDE (arXiv 2510.14096)](https://arxiv.org/html/2510.14096v3)

### 1.3. Kiến trúc và mục tiêu thiết kế — không cạnh tranh trực tiếp

TREET dùng Transformer, TENDE dùng diffusion model — cả hai là kiến trúc lớn,
tối ưu cho độ chính xác trên mẫu lớn. Mạng của em chỉ là MLP 3 lớp ẩn (25.473
tham số) — nhẹ hơn nhiều, thiết kế cho suy luận nhanh trên cửa sổ ngắn (amortized:
train 1 lần, áp dụng ngay không cần tối ưu lại mỗi lần). Đáng chú ý, chính TENDE
(khi so với TREET trên dữ liệu thật) báo cáo "TREET cho sai số lớn hơn hẳn" và
khi thêm chiều nhiễu dư thì "TREET có phương sai lớn, ra cả giá trị âm" — tức là
ngay cả 2 bài này cũng không thắng tuyệt đối lẫn nhau. Em coi đây là thêm 1 bằng
chứng rằng chưa có phương pháp nào chiếm ưu thế toàn diện, và vùng N nhỏ tuyệt
đối (điều em tập trung) vẫn là câu hỏi mở.

### 1.4. Về phương pháp ước lượng nói chung (khoảng trống domain generalization)

Đây là phần đóng góp phương pháp luận chính của đề tài, và đây là điểm em thấy
rõ nhất khoảng trống trong các nghiên cứu hiện có:

- Một bài báo gần đây (MIST, huấn luyện một mạng ước lượng thông tin tương hỗ
  trên một tập lớn các phân phối mô phỏng đã biết đáp án — cùng triết lý
  "amortized" với đề tài của em) **tự nhận xét**: "hầu hết các phương pháp ước
  lượng MI được kiểm định chủ yếu trên dữ liệu mô phỏng với cấu trúc phụ thuộc
  đơn giản và cỡ mẫu gần như vô hạn, đặt ra câu hỏi về khả năng áp dụng thực
  tế." Đây đúng là khoảng trống em đã kiểm định trực tiếp (mục 2 dưới cùng câu
  hỏi này) — và tìm ra amortized MINE của em thất bại khi gặp dữ liệu thật,
  đúng như lo ngại mà bài báo trên đặt ra nhưng chưa tự kiểm định.
- Một nghiên cứu khác về MINE ở không gian nhiều chiều ghi nhận: chặn dưới biến
  phân (variational lower bound) của MINE có thể tụt xuống dưới cả giá trị MI
  thật ở vùng MI thấp, cỡ mẫu nhỏ — khớp với hiện tượng em quan sát được: độ
  lệch (bias) không đổi của Amortized không co về 0 như KSG.
- Về KSG: được ghi nhận trong y văn là "không lệch (bias-free) ở số chiều
  thấp nhưng gặp giới hạn (curse of dimensionality) khi số chiều ≳20" — phù hợp
  với việc trong đề tài của em (số chiều thấp, chỉ 1 biến trễ), KSG vẫn là
  phương pháp nền tảng đáng tin cậy, đặc biệt trên dữ liệu thật.

*Nguồn:* [MIST: Mutual Information Estimation via Supervised Training](https://arxiv.org/pdf/2511.18945), [NMINE: Normalized Mutual Information Neural Estimation](https://arxiv.org/html/2607.27710v1), [Analysis of k-Nearest Neighbor Distances with Application to Entropy Estimation](https://arxiv.org/pdf/1603.08578)

Ngoài ra, hướng hô hấp→tim chiếm ưu thế cũng khớp với 2 nghiên cứu TE tim–hô hấp
khác không liên quan đến TREET/TENDE — [nghiên cứu trên trẻ sơ sinh](https://pmc.ncbi.nlm.nih.gov/articles/PMC7481456/) và [nghiên cứu trên vận động viên đạp xe](https://pmc.ncbi.nlm.nih.gov/articles/PMC7052290/) — nên đây là một kết luận đã được nhiều nhóm độc lập xác nhận, không riêng 2 bài em so sánh kỹ ở trên.

### 1.5. Nhận xét vị trí của đề tài

Đóng góp của em không nằm ở việc tạo ra một phương pháp hoàn toàn mới, mà ở việc
**định lượng chính xác** một khoảng trống mà chính các nghiên cứu amortized/MINE
gần đây tự nhận là chưa kiểm định kỹ (mục 1.4), đồng thời áp dụng lên đúng một
bài toán sinh lý học có ý nghĩa thực tế (RSA) và cho kết quả đúng hướng với y
văn đã có (mục 1.2). Khác với TREET/TENDE, đề tài của em tập trung đúng vào
vùng N nhỏ (10–200, mục 1.1) mà 2 bài này chưa kiểm định — đây là vùng đóng góp
không trùng lặp. Em xin đề xuất: nếu thầy/cô đồng ý, em sẽ đưa bảng so sánh này
vào phần Related Work của bản thảo.

---

## 2. Dùng dữ liệu mô phỏng có đảm bảo không rò rỉ dữ liệu không?

### 2.1. Rò rỉ dữ liệu (data leakage) là gì

Rò rỉ dữ liệu xảy ra khi thông tin từ **phần dữ liệu dùng để đánh giá** (test)
vô tình lọt vào **phần dữ liệu dùng để huấn luyện hoặc điều chỉnh** (train/tuning)
mô hình. Hậu quả: kết quả đánh giá trông có vẻ rất tốt, nhưng thực chất mô hình
chỉ đang "nhớ lại" một phần đáp án nó đã thấy, không phải khả năng tổng quát hoá
thật. Đây là một trong những lỗi phổ biến và nguy hiểm nhất khi công bố kết quả
học máy.

**Điểm quan trọng cần làm rõ:** dùng dữ liệu mô phỏng **không tự động đảm bảo
không có rò rỉ** — rò rỉ là vấn đề về CÁCH chia và sử dụng dữ liệu (train/val/test),
không phải về việc dữ liệu là thật hay mô phỏng. Em có thể vẫn để rò rỉ xảy ra
trên dữ liệu mô phỏng nếu chia tập không đúng cách.

### 2.2. Cách em đã kiểm soát rò rỉ trong đề tài

**(a) Huấn luyện mạng amortized:** em chia dữ liệu mô phỏng thành 3 tập độc lập
bằng seed sinh số ngẫu nhiên khác nhau — tập train, tập validation (dùng để
theo dõi lúc huấn luyện, chọn điểm dừng), và **một tập test hoàn toàn riêng,
sinh bằng seed gốc khác hẳn** (`test_base_seed`), được ghi rõ ngay từ lúc tạo dữ
liệu là "không được dùng trong bất kỳ quyết định nào ở bước huấn luyện". Toàn
bộ số liệu bias/variance/MSE báo cáo trong đề tài đều tính trên tập test này,
không phải tập đã huấn luyện.

**Điểm cần nói rõ để không phóng đại:** tập train và tập test dùng CÙNG một
lưới tham số (cùng các giá trị cường độ ghép nối, độ nhiễu, kích thước cửa sổ
N) — chỉ khác nhau ở lần lấy mẫu ngẫu nhiên cụ thể (seed khác). Nghĩa là: kết
quả đo được là khả năng tổng quát hoá lên **mẫu mới cùng một phân phối đã biết**,
chứ **chưa phải** khả năng tổng quát hoá lên một vùng tham số hoàn toàn chưa
từng gặp. Đây là một giới hạn hợp lý cần nêu trong phần Discussion, không phải
lỗi.

**(b) Hiệu chỉnh độ lệch (bias calibration, Pha T.3):** khi hiệu chỉnh độ lệch
cho một cấu hình dữ liệu cụ thể, em chỉ dùng thông tin độ lệch đo được từ CÁC
CẤU HÌNH KHÁC (kỹ thuật gọi là leave-one-config-out) — tuyệt đối không dùng
thông tin của chính cấu hình đang được đánh giá để tự hiệu chỉnh cho nó. Đây là
bước em chủ động thêm vào chính vì lo ngại rủi ro rò rỉ ở bước này.

**(c) Dữ liệu thật (Fantasia, Apnea-ECG):** đây là bằng chứng chống rò rỉ mạnh
nhất trong toàn bộ đề tài — dữ liệu thật **không được dùng để huấn luyện hay
điều chỉnh bất kỳ tham số nào của mạng amortized**, chỉ dùng để đánh giá cuối
cùng. Nếu có rò rỉ khiến kết quả "ảo" trên dữ liệu mô phỏng, thì khi gặp dữ liệu
thật (hoàn toàn ngoài phân phối train) mô hình vẫn sẽ thất bại — và thực tế
đúng là như vậy: mạng amortized thất bại rõ ràng trên dữ liệu thật (kết quả sai
cả hướng), trong khi KSG (không cần huấn luyện, không có khái niệm rò rỉ) vẫn
cho kết quả đúng. Nói cách khác, **chính sự thất bại này lại là minh chứng rằng
kết quả tốt trên dữ liệu mô phỏng không phải do rò rỉ/nhớ đáp án** — nếu là vậy,
nó cũng sẽ "nhớ" tốt trên dữ liệu thật, nhưng không phải như thế.

### 2.3. Tóm tắt trả lời thầy/cô

> Dữ liệu mô phỏng không tự động đảm bảo không rò rỉ — điều đảm bảo là cách em
> tách riêng tập test bằng seed độc lập ngay từ đầu, không đụng đến khi huấn
> luyện. Giới hạn cần nêu rõ: tập test dùng chung lưới tham số với tập train
> (chỉ khác mẫu ngẫu nhiên), nên đây là kiểm định khả năng tổng quát hoá lên
> **mẫu mới**, chưa phải lên **miền tham số mới**. Bằng chứng thuyết phục nhất
> chống lại nghi ngờ rò rỉ chính là kết quả trên dữ liệu thật — nơi mô hình
> không được train, và nó thất bại đúng như kỳ vọng nếu không có rò rỉ.

---

Sources:
- [TREET: TRansfer Entropy Estimation via Transformers](https://arxiv.org/html/2402.06919v3)
- [TENDE: Transfer Entropy Neural Diffusion Estimation](https://arxiv.org/html/2510.14096v3)
- [Transfer Entropy Modeling of Newborn Cardiorespiratory Regulation](https://pmc.ncbi.nlm.nih.gov/articles/PMC7481456/)
- [A Transfer Entropy Approach for the Assessment of the Impact of Inspiratory Muscle Training on the Cardiorespiratory Coupling of Amateur Cyclists](https://pmc.ncbi.nlm.nih.gov/articles/PMC7052290/)
- [MIST: Mutual Information Estimation via Supervised Training](https://arxiv.org/pdf/2511.18945)
- [NMINE: Normalized Mutual Information Neural Estimation](https://arxiv.org/html/2607.27710v1)
- [Analysis of k-Nearest Neighbor Distances with Application to Entropy Estimation](https://arxiv.org/pdf/1603.08578)
