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
