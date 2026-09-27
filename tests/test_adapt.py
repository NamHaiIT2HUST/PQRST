import copy
import numpy as np
import torch
import pytest

from pqrst.data.synthetic.corpus import Window
from pqrst.estimators.mine.amortized import (
    MaskedStatisticsNetwork, AmortizedTEEstimator, AmortizedTrainConfig,
)
from pqrst.estimators.mine.adapt import make_record_folds, adapt_amortized


def _w(seed, n=20):
    r = np.random.default_rng(seed)
    return Window(y_t=r.standard_normal(n), x_lag=r.standard_normal(n), y_lag=r.standard_normal(n),
                  n_samples=n, config_name="t", params={"c": 0.0, "noise_std": 1.0},
                  te_ground_truth=float("nan"), mi_full_ground_truth=float("nan"),
                  mi_reduced_ground_truth=float("nan"), seed=seed)


def test_make_record_folds_partition_and_determinism():
    ids = [f"r{i}" for i in range(10)]
    f1 = make_record_folds(ids, 4, seed=1)
    f2 = make_record_folds(ids, 4, seed=1)
    assert f1 == f2
    flat = [x for f in f1 for x in f]
    assert sorted(flat) == sorted(ids) and len(flat) == len(set(flat))
    assert len(f1) == 4


def test_make_record_folds_bad_k():
    with pytest.raises(ValueError):
        make_record_folds(["a", "b"], 5)


def test_adapt_does_not_modify_base_and_changes_copy():
    torch.manual_seed(0)
    cfg = AmortizedTrainConfig(hidden_dims=[8, 8], standardize=True, eval_n_shuffles=2)
    base = AmortizedTEEstimator(MaskedStatisticsNetwork([8, 8]), cfg)
    before = copy.deepcopy(base.model.state_dict())
    adapted, hist = adapt_amortized(base, [_w(i) for i in range(6)], [_w(100), _w(101)],
                                    learning_rate=1e-2, max_epochs=2, windows_per_batch=3,
                                    final_estimate_last_k_epochs=1)
    for k, v in base.model.state_dict().items():
        assert torch.equal(v, before[k])          # base khong doi
    changed = any(not torch.equal(adapted.model.state_dict()[k], before[k]) for k in before)
    assert changed                                # ban sao da doi
    assert np.isfinite(hist["final_val_loss"])
    assert adapted.config.standardize is True     # giu nguyen chuan hoa
