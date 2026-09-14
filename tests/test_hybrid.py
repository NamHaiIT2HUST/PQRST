import pytest
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from pqrst.estimators.hybrid import HybridTEEstimator
from pqrst.baselines.ksg import KSGTEEstimator
from pqrst.estimators.mine.amortized import AmortizedTEEstimator

class DummyEstimator:
    def __init__(self, val):
        self.val = val
    def estimate(self, x, y, **kwargs):
        return self.val

def test_hybrid_threshold():
    ksg = DummyEstimator(1.0)
    amortized = DummyEstimator(2.0)
    hybrid = HybridTEEstimator(ksg, amortized, threshold_n=50)
    
    x49 = np.zeros(50)
    assert hybrid.estimate(x49, x49) == 1.0
    
    x50 = np.zeros(51)
    assert hybrid.estimate(x50, x50) == 2.0

def test_hybrid_does_not_regress_periodic_n30_bug():
    """Test hoi quy DUNG cho bug da tim thay: threshold=30 lam Hybrid chon
    Amortized tai N=30 tren du lieu periodic, nhung KSG moi la lua chon tot hon
    o do. Test nay phai fail neu threshold_n <= 30."""
    
    BASE = Path.cwd().parent if Path.cwd().name == 'tests' else Path.cwd()
    df = pd.read_csv(BASE / 'results' / 'tables' / 'phase_r2_periodic_grid_summary.csv')
    
    df_ksg = df[(df['estimator'] == 'KSG') & (df['n_samples'] == 30)]
    df_amo = df[(df['estimator'].str.startswith('Amortized')) & (df['n_samples'] == 30)]
    
    # Kiem tra trung binh tren tap nay xem variance nao nho hon
    mean_ksg = df_ksg['variance'].mean()
    mean_amo = df_amo['variance'].mean()
    
    assert mean_ksg < mean_amo, "Baseline assertion: KSG better than Amortized at N=30 on periodic"
    
    # Kiem tra xem threshold_n default cua Hybrid co khien no chon KSG tai N=30 khong
    # Gia lap viec estimate tren mang do dai 31 de n_samples = 30
    ksg_mock = DummyEstimator("KSG")
    amo_mock = DummyEstimator("AMO")
    hybrid = HybridTEEstimator(ksg_mock, amo_mock)
    
    x30 = np.zeros(31) # len(x)-1 = 30
    choice = hybrid.estimate(x30, x30)
    
    assert choice == "KSG", f"Hybrid picked {choice}, but KSG is better at N=30 on periodic! Threshold bug."

def test_hybrid_tradeoff_report(caplog):
    """Dem va log danh sach cac dong hybrid phai hy sinh (khong chon estimator tot nhat)."""
    caplog.set_level(logging.INFO)
    BASE = Path.cwd().parent if Path.cwd().name == 'tests' else Path.cwd()
    tables_dir = BASE / 'results' / 'tables'
    
    hybrid = HybridTEEstimator(DummyEstimator("KSG"), DummyEstimator("AMO"))
    threshold = hybrid.threshold_n
    
    sacrifices = []
    
    files = [('Linear', 'phase_r_grid_summary.csv'), ('Periodic', 'phase_r2_periodic_grid_summary.csv')]
    for ds_name, file in files:
        df = pd.read_csv(tables_dir / file)
        df_ksg = df[df['estimator'] == 'KSG']
        df_amo = df[df['estimator'].str.startswith('Amortized')]
        
        group_cols = ['config_name', 'noise_std', 'n_samples']
        if 'coupling_c' in df.columns:
            group_cols.append('coupling_c')
            
        merged = pd.merge(df_ksg, df_amo, on=group_cols, suffixes=('_ksg', '_amo'))
        for _, row in merged.iterrows():
            n = row['n_samples']
            v_ksg = row['variance_ksg']
            v_amo = row['variance_amo']
            
            picked_ksg = (n < threshold)
            v_picked = v_ksg if picked_ksg else v_amo
            v_best = min(v_ksg, v_amo)
            
            if v_picked > v_best + 1e-9:
                sacrifices.append({
                    'dataset': ds_name,
                    'config': row['config_name'],
                    'n': n,
                    'v_picked': v_picked,
                    'v_best': v_best,
                    'picked': 'KSG' if picked_ksg else 'AMO'
                })
                
    logging.info(f"\n--- TRADEOFF REPORT ---")
    logging.info(f"Total sacrifices: {len(sacrifices)}")
    for s in sacrifices:
        logging.info(f"{s['dataset']} N={s['n']} config={s['config']}: picked {s['picked']} ({s['v_picked']:.6f}) but best was {s['v_best']:.6f}")
    
    # We expect some sacrifices (specifically N=30 on linear data)
    assert True

def test_hybrid_integration():
    """Test tich hop that, KSG + Amortized(loaded) chay khong loi tra ve float."""
    BASE = Path.cwd().parent if Path.cwd().name == 'tests' else Path.cwd()
    ckpt = BASE / 'results' / 'checkpoints' / 'phi_amortized_standardized.pt'
    if not ckpt.exists():
        pytest.skip("Checkpoint not found for integration test")
        
    ksg = KSGTEEstimator()
    amo = AmortizedTEEstimator.load(str(ckpt))
    hybrid = HybridTEEstimator(ksg, amo)
    
    # N=10 => uses KSG
    x10 = np.random.randn(11)
    y10 = np.random.randn(11)
    res10 = hybrid.estimate(x10, y10)
    assert isinstance(res10, float) and np.isfinite(res10)
    
    # N=100 => uses Amortized
    x100 = np.random.randn(101)
    y100 = np.random.randn(101)
    res100 = hybrid.estimate(x100, y100)
    assert isinstance(res100, float) and np.isfinite(res100)
