import pandas as pd

class BiasCalibrator:
    """Hieu chinh bias theo N bang bang tra + noi tuyen tinh (khong ngoai suy
    ngoai khoang N da fit)."""
    def __init__(self):
        self._table = {}
        self._n_sorted = []

    def fit(self, n_values, bias_values) -> "BiasCalibrator":
        df = pd.DataFrame({'n': n_values, 'bias': bias_values})
        mean_bias = df.groupby('n')['bias'].mean().to_dict()
        self._table = mean_bias
        self._n_sorted = sorted(mean_bias.keys())
        return self

    def predict(self, n: int) -> float:
        if not self._n_sorted:
            return 0.0
        if n in self._table:
            return self._table[n]
        if n <= self._n_sorted[0]:
            return self._table[self._n_sorted[0]]
        if n >= self._n_sorted[-1]:
            return self._table[self._n_sorted[-1]]
            
        n_prev, n_next = None, None
        for i in range(len(self._n_sorted) - 1):
            if self._n_sorted[i] < n < self._n_sorted[i+1]:
                n_prev = self._n_sorted[i]
                n_next = self._n_sorted[i+1]
                break
                
        b_prev = self._table[n_prev]
        b_next = self._table[n_next]
        
        w = (n - n_prev) / (n_next - n_prev)
        return b_prev + w * (b_next - b_prev)

def calibrate_grid_leave_one_config_out(df: pd.DataFrame, estimator_name: str = "Amortized") -> pd.DataFrame:
    """Voi moi dong (config_name, n_samples) cua estimator_name trong df:
    - Fit BiasCalibrator.fit(n_values, bias_values) CHI tren cac dong co
      config_name KHAC dong dang xet (leave-one-config-out - tranh data leakage).
    - bias_hat = calibrator.predict(n_samples cua dong dang xet)
    - calibrated_bias = bias - bias_hat
    - calibrated_variance = variance (KHONG doi - vi tru hang so khong lam doi phuong sai)
    - calibrated_mse = calibrated_bias**2 + calibrated_variance
    Tra ve DataFrame moi, cung cau truc cot nhu df goc, voi estimator =
    f"{estimator_name}_calibrated", giu nguyen config_name/coupling_c/noise_std/n_samples.
    """
    df_est = df[df['estimator'] == estimator_name].copy()
    
    calibrated_rows = []
    for idx, row in df_est.iterrows():
        train_df = df_est[df_est['config_name'] != row['config_name']]
        
        calibrator = BiasCalibrator()
        if not train_df.empty:
            calibrator.fit(train_df['n_samples'], train_df['bias'])
            
        bias_hat = calibrator.predict(row['n_samples'])
        
        calibrated_bias = row['bias'] - bias_hat
        calibrated_variance = row['variance']
        calibrated_mse = calibrated_bias**2 + calibrated_variance
        
        new_row = row.copy()
        new_row['estimator'] = f"{estimator_name}_calibrated"
        new_row['bias'] = calibrated_bias
        new_row['variance'] = calibrated_variance
        new_row['mse'] = calibrated_mse
        calibrated_rows.append(new_row)
        
    return pd.DataFrame(calibrated_rows)
