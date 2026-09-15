import pytest
import pandas as pd
from pqrst.estimators.calibration import BiasCalibrator, calibrate_grid_leave_one_config_out

def test_bias_calibrator_fit_predict():
    calibrator = BiasCalibrator()
    calibrator.fit([10, 10, 20, 20], [-0.1, -0.1, -0.05, -0.05])
    
    assert calibrator.predict(10) == pytest.approx(-0.1)
    assert calibrator.predict(20) == pytest.approx(-0.05)
    
    # Noi tuyen tinh
    assert calibrator.predict(15) == pytest.approx(-0.075)
    
    # Clamp (khong ngoai suy)
    assert calibrator.predict(5) == pytest.approx(-0.1)
    assert calibrator.predict(100) == pytest.approx(-0.05)

def test_calibrate_leave_one_config_out_no_leakage():
    # Tao 3 config, n_samples = 10 cho tat ca
    # Khi xet A: train = B, C => mean_bias_hat = (-0.2 - 0.3)/2 = -0.25
    df = pd.DataFrame({
        'estimator': ['Amortized', 'Amortized', 'Amortized'],
        'config_name': ['A', 'B', 'C'],
        'n_samples': [10, 10, 10],
        'bias': [-0.1, -0.2, -0.3],
        'variance': [0.01, 0.01, 0.01]
    })
    
    res = calibrate_grid_leave_one_config_out(df, estimator_name='Amortized')
    
    row_a = res[res['config_name'] == 'A'].iloc[0]
    
    # bias_hat phai la trung binh cua B va C (tuc la -0.25)
    # cal_bias = row_bias - bias_hat = -0.1 - (-0.25) = 0.15
    assert row_a['bias'] == pytest.approx(0.15)
    assert row_a['estimator'] == 'Amortized_calibrated'

def test_calibrate_grid_math():
    df = pd.DataFrame({
        'estimator': ['Amortized', 'Amortized'],
        'config_name': ['A', 'B'],
        'n_samples': [10, 10],
        'bias': [0.5, 0.3],   # row A bias = 0.5. train tren B => bias_hat = 0.3
        'variance': [0.04, 0.02] # row A variance = 0.04
    })
    
    res = calibrate_grid_leave_one_config_out(df, estimator_name='Amortized')
    row_a = res[res['config_name'] == 'A'].iloc[0]
    
    # Calibrated bias for A = 0.5 - 0.3 = 0.2
    # Calibrated var for A = 0.04
    # Calibrated mse for A = 0.2^2 + 0.04 = 0.04 + 0.04 = 0.08
    assert row_a['bias'] == pytest.approx(0.2)
    assert row_a['variance'] == pytest.approx(0.04)
    assert row_a['mse'] == pytest.approx(0.08)
