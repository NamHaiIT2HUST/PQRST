# Ghi chú Toán học (Theory Notes) - Pha U

Tài liệu này chính thức hoá các phát hiện toán học từ quá trình thực nghiệm, đóng vai trò làm nền tảng lý thuyết (Mệnh đề) bảo vệ cho các quyết định thiết kế kiến trúc của mô hình ước lượng TE (Transfer Entropy) trong dự án.

## Mệnh đề 1: Sự sụp đổ của Donsker-Varadhan Bound đối với các hàm thống kê tách rời cộng tính

**Phát biểu:**
Giả sử hàm thống kê (statistics network) $T(x, y)$ dùng để xấp xỉ tỉ số mật độ trong chặn dưới Donsker-Varadhan (DV) có tính chất *tách rời cộng tính* (additively separable), tức là tồn tại hai hàm $f(x)$ và $g(y)$ sao cho:
$$T(x, y) = f(x) + g(y)$$

Khi đó, chặn dưới Donsker-Varadhan của Mutual Information (MI) luôn bị chặn trên bởi $0$, bất kể cấu trúc của phân phối thực tế $P_{XY}$:
$$I_{DV}(X; Y) \le 0$$

**Chứng minh:**
Chặn dưới DV được định nghĩa là:
$$I_{DV} = \mathbb{E}_{P_{XY}}[T(X, Y)] - \log \mathbb{E}_{P_X \otimes P_Y}\left[e^{T(X, Y)}\right]$$

Thay $T(x, y) = f(x) + g(y)$ vào biểu thức:
1. Số hạng thứ nhất (kỳ vọng đồng thời):
$$\mathbb{E}_{P_{XY}}[f(X) + g(Y)] = \mathbb{E}_{P_X}[f(X)] + \mathbb{E}_{P_Y}[g(Y)]$$
*(Do tính chất tuyến tính của kỳ vọng).*

2. Số hạng thứ hai (kỳ vọng biên):
$$\mathbb{E}_{P_X \otimes P_Y}\left[e^{f(X) + g(Y)}\right] = \mathbb{E}_{P_X \otimes P_Y}\left[e^{f(X)} e^{g(Y)}\right] = \mathbb{E}_{P_X}[e^{f(X)}] \mathbb{E}_{P_Y}[e^{g(Y)}]$$
*(Do $X$ và $Y$ độc lập trong phân phối biên $P_X \otimes P_Y$).*

Lấy logarit:
$$\log \left( \mathbb{E}_{P_X}[e^{f(X)}] \mathbb{E}_{P_Y}[e^{g(Y)}] \right) = \log \mathbb{E}_{P_X}[e^{f(X)}] + \log \mathbb{E}_{P_Y}[e^{g(Y)}]$$

Áp dụng bất đẳng thức Jensen cho hàm lồi $\phi(z) = e^z$, ta có $\mathbb{E}[e^Z] \ge e^{\mathbb{E}[Z]}$, suy ra $\log \mathbb{E}[e^Z] \ge \mathbb{E}[Z]$.
Áp dụng điều này cho từng số hạng:
$$\log \mathbb{E}_{P_X}[e^{f(X)}] \ge \mathbb{E}_{P_X}[f(X)]$$
$$\log \mathbb{E}_{P_Y}[e^{g(Y)}] \ge \mathbb{E}_{P_Y}[g(Y)]$$

Từ đó, biểu thức DV bound trở thành:
$$I_{DV} = \left( \mathbb{E}_{P_X}[f(X)] - \log \mathbb{E}_{P_X}[e^{f(X)}] \right) + \left( \mathbb{E}_{P_Y}[g(Y)] - \log \mathbb{E}_{P_Y}[e^{g(Y)}] \right) \le 0 + 0 = 0$$

**Ý nghĩa thiết kế:**
Mệnh đề này giải thích trực tiếp tại sao mô hình Cổ điển học bằng Fourier Features (cấu trúc P'.0a) luôn cho ước lượng MI hội tụ về 0. Nó bắt buộc các mô hình ước lượng MI/TE (dù là MLP cổ điển hay mạch lượng tử) phải chứa **kết nối tương tác phi tuyến (cross-terms/entanglement)** giữa các biến $X$ và $Y$ (ví dụ: các phép nhân ma trận chéo trong MLP hoặc cổng CNOT trong mạch lượng tử).


## Mệnh đề 2: Giảm phương sai ước lượng TE thông qua tham số chia sẻ (Shared Network) và hoán vị biên

**Phát biểu:**
Ước lượng Transfer Entropy (TE) được định nghĩa là hiệu của hai Mutual Information (MI):
$$\widehat{TE} = \widehat{I}_{full} - \widehat{I}_{reduced}$$
Trong đó $I_{full} = I(X_{lag}, Y_{lag}; Y_{t})$ và $I_{reduced} = I(Y_{lag}; Y_{t})$.

Phương sai của ước lượng này phụ thuộc vào hệ số tương quan $\rho$ giữa 2 phân đoạn đo:
$$\text{Var}(\widehat{TE}) = \text{Var}(\widehat{I}_{full}) + \text{Var}(\widehat{I}_{reduced}) - 2 \rho \sqrt{\text{Var}(\widehat{I}_{full}) \text{Var}(\widehat{I}_{reduced})}$$

Bằng cách thiết kế `MaskedStatisticsNetwork` chia sẻ chung tham số mạng và sử dụng **cùng một hoán vị ngẫu nhiên (shared permutation)** khi tính ước lượng Monte Carlo cho số hạng biên, ta tạo ra sự tương quan dương rất mạnh ($\rho > 0$) giữa $\widehat{I}_{full}$ và $\widehat{I}_{reduced}$. Kết quả là $\text{Var}(\widehat{TE})$ bị triệt tiêu đáng kể, nhỏ hơn nhiều so với tổng phương sai của hai phép đo độc lập.

**Ý nghĩa thiết kế:**
Đây là cơ sở toán học chứng minh kiến trúc `MaskedStatisticsNetwork` (Pha R) tối ưu hơn so với việc huấn luyện hai mạng MLP hoàn toàn tách biệt.


## Mệnh đề 3: Tác động của Nhiễu Gauss Cộng tính (AWGN) và Tính Bền vững của Mô hình Học sẵn

**Phát biểu:**
Xét tín hiệu đo quan sát được bị nhiễm nhiễu trắng Gauss độc lập:
$$\widetilde{X}[t] = X[t] + \xi_x[t], \quad \widetilde{Y}[t] = Y[t] + \xi_y[t]$$
với $\xi_x[t] \sim \mathcal{N}(0, \sigma_{\xi_x}^2)$ và $\xi_y[t] \sim \mathcal{N}(0, \sigma_{\xi_y}^2)$ độc lập với nhau và độc lập với tín hiệu gốc.

1. **Bất đẳng thức xử lý dữ liệu (Data Processing Inequality - DPI):**
   Mối quan hệ nhân quả thực chất suy giảm khi nhiễu tăng:
   $$TE(\widetilde{X} \to \widetilde{Y}) \le TE(X \to Y)$$
   Cụ thể đối với hệ tuyến tính Gaussian (VAR(1)), nếu phương sai nhiễu đo tăng theo mức $\text{SNR}$, tỉ số phương sai phần dư điều kiện co về 1, làm giá trị TE thực sự tiệm cận về 0 theo tốc độ $\mathcal{O}(\text{SNR})$.

2. **So sánh tính bền vững giữa KSG (k-NN) và Amortized Neural Estimator:**
   * **Phương pháp KSG (k-NN):** KSG dựa trên khoảng cách Euclidean cực tiểu $r = \min_{i} \|z - z_i\|$. Khi có nhiễu Gaussian đa chiều (không gian kết hợp 3 chiều $(y_t, x_{lag}, y_{lag})$), hiện tượng tập trung khoảng cách (distance concentration phenomenon) do nhiễu làm sai lệch nghiêm trọng thể tích hình cầu k-NN $\epsilon(i)$, dẫn đến độ chệch (Bias) tăng vọt tỉ lệ nghịch với SNR, đặc biệt ở mẫu nhỏ $N$.
   * **Amortized MINE:** Mạng nơ-ron thống kê $T_\phi$ với các hàm kích hoạt phi tuyến trơn (ELU) đóng vai trò như một bộ lọc không gian (spatial smooth regularizer). Do đã được huấn luyện trước trên manifold dữ liệu đa dạng, $T_\phi$ duy trì được xấp xỉ tỉ số mật độ logarit $\log \frac{dP_{XY}}{dP_X dP_Y}$ ổn định hơn trước các nhiễu vi mô cục bộ so với các phép đo khoảng cách lân cận rời rạc.

**Hệ quả thực nghiệm:**
Khi đưa mức nhiễu Gauss từ $\text{SNR} = 20\text{dB}$ xuống $0\text{dB}$, phương sai và sai số tương đối của KSG tăng nhanh hơn rõ rệt so với Amortized MINE và mô hình lai Hybrid, xác nhận tính ưu việt của việc có prior học sẵn trong môi trường đo nhiễu sinh lý (như ECG/PPG bị nhiễu chuyển động).


## Mệnh đề 4: Cơ sở Toán học của Điểm giao cắt (Crossover Point) giữa KSG và Amortized MINE theo Cỡ mẫu $N$

**Phát biểu:**
Sai số toàn phương trung bình (MSE) của một bộ ước lượng TE được phân rã thành:
$$\text{MSE}(\widehat{TE}) = \text{Bias}^2(\widehat{TE}) + \text{Var}(\widehat{TE})$$

1. **Hành vi tiệm cận của KSG (Phi tham số k-NN):**
   * $\text{Bias}_{KSG}(N) = \mathcal{O}(N^{-\gamma})$ với $\gamma > 0$ phụ thuộc vào số chiều. Khi $N$ tăng, bias co dần về 0.
   * $\text{Var}_{KSG}(N) = \mathcal{O}(N^{-1})$. Khi $N$ rất nhỏ ($N < 30$), mặc dù phương sai vẫn giảm theo $1/N$, nhưng tính thích nghi cục bộ của k-NN giúp nó không bị ràng buộc bởi năng lực mô hình cố định.

2. **Hành vi của Amortized MINE (Tham số hóa học sẵn):**
   * $\text{Bias}_{Amortized}(N) \approx \text{const} \ne 0$ (tiệm cận một hằng số phụ thuộc vào dung lượng biểu diễn của mạng $T_\phi$ tại thời điểm huấn luyện, không co về 0 theo $N$).
   * $\text{Var}_{Amortized}(N)$: Ở $N$ rất nhỏ ($N < 30$), hàm mất mát Donsker-Varadhan chịu nhiễu Monte Carlo cực đại từ số hạng $\log \left( \frac{1}{N} \sum_{i=1}^N e^{T(x_i, y_{\pi(i)})} \right)$ (hiện tượng hàm mũ của biến ngẫu nhiên ở mẫu hữu hạn nhỏ). Khi $N \ge 50$, luật số lớn phát huy tác dụng, phương sai Monte Carlo của MINE sụt giảm với tốc độ nhanh hơn KSG nhờ thông tin tiền nghiệm đã học trong trọng số $\phi$.

**Hệ quả:**
Đường cong $\text{Var}(N)$ của Amortized MINE dốc hơn đường cong của KSG và cắt KSG tại một ngưỡng quan sát được $N^* \in [30, 50]$. Đây là lý do toán học trực tiếp giải thích sự cần thiết của bộ ước lượng lai `HybridTEEstimator`:
$$\widehat{TE}_{Hybrid}(X \to Y) = \begin{cases} \widehat{TE}_{KSG}(X \to Y) & \text{khi } N < N^* \\ \widehat{TE}_{Amortized}(X \to Y) & \text{khi } N \ge N^* \end{cases}$$

