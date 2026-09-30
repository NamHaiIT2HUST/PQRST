import numpy as np
import pandas as pd
from pqrst.evaluation.significance import paired_bootstrap_test, compare_paired_series


def test_paired_bootstrap_identical():
    x = np.ones(50)
    res = paired_bootstrap_test(x, x, n_bootstrap=200, seed=42)
    assert res["mean_diff"] == 0.0
    assert res["ci_low"] == 0.0
    assert res["ci_high"] == 0.0
    assert not res["significant_005"]


def test_paired_bootstrap_distinct():
    rng = np.random.default_rng(42)
    a = rng.normal(loc=1.0, scale=0.1, size=100)
    b = rng.normal(loc=0.0, scale=0.1, size=100)
    res = paired_bootstrap_test(a, b, n_bootstrap=500, seed=42)
    assert res["mean_diff"] > 0.8
    assert res["ci_low"] > 0
    assert res["p_value_bootstrap"] < 0.01
    assert res["significant_005"]


def test_compare_paired_series():
    df = pd.DataFrame({
        "group": ["g1"] * 30 + ["g2"] * 30,
        "model_a": np.concatenate([np.ones(30) * 2, np.ones(30) * 5]),
        "model_b": np.concatenate([np.ones(30) * 1, np.ones(30) * 5]),
    })
    res_df = compare_paired_series(df, "model_a", "model_b", group_col="group", n_bootstrap=200)
    assert len(res_df) == 2
    assert "group" in res_df.columns
    row_g1 = res_df[res_df["group"] == "g1"].iloc[0]
    assert row_g1["significant_005"]
    row_g2 = res_df[res_df["group"] == "g2"].iloc[0]
    assert not row_g2["significant_005"]
