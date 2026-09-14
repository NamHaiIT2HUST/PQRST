from pqrst.baselines.base import BaseTEEstimator

class HybridTEEstimator(BaseTEEstimator):
    """TE = KSG neu N < threshold_n, Amortized neu N >= threshold_n. Nguong mac dinh
    50 (KHONG phai 30). Ly do: voi du lieu THAT khong biet truoc dong luc hoc la
    tuyen tinh hay phi tuyen. Nguong 50 dam bao khong bao gio chon nham phuong an
    te hon o bat ky N da kiem chung nao, tren ca 2 loai du lieu."""

    def __init__(self, ksg_estimator, amortized_estimator, threshold_n: int = 50):
        self.ksg = ksg_estimator
        self.amortized = amortized_estimator
        self.threshold_n = threshold_n

    def estimate(self, x, y, **kwargs) -> float:
        n = len(x) - 1
        if n < self.threshold_n:
            return self.ksg.estimate(x, y, **kwargs)
        return self.amortized.estimate(x, y, **kwargs)
