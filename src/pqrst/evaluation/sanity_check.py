from __future__ import annotations
import numpy as np

def bidirectional_te(windows_forward: list, windows_backward: list, estimator) -> dict:
    te_f = []
    te_b = []
    for w_f, w_b in zip(windows_forward, windows_backward):
        x_f = np.zeros(w_f.n_samples + 1); y_f = np.zeros(w_f.n_samples + 1)
        x_f[:-1] = w_f.x_lag; y_f[:-1] = w_f.y_lag; y_f[1:] = w_f.y_t
        try: te_f.append(estimator.estimate(x_f, y_f))
        except: pass
        
        x_b = np.zeros(w_b.n_samples + 1); y_b = np.zeros(w_b.n_samples + 1)
        x_b[:-1] = w_b.x_lag; y_b[:-1] = w_b.y_lag; y_b[1:] = w_b.y_t
        try: te_b.append(estimator.estimate(x_b, y_b))
        except: pass
        
    return {
        "te_forward_mean": np.mean(te_f),
        "te_backward_mean": np.mean(te_b),
        "te_forward_values": te_f,
        "te_backward_values": te_b,
        "difference": np.mean(te_f) - np.mean(te_b),
        "n_windows": len(te_f)
    }

def permutation_test_te(windows: list, estimator, n_permutations: int = 500, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    
    te_obs = []
    x_list = []; y_list = []
    for w in windows:
        x = np.zeros(w.n_samples + 1); y = np.zeros(w.n_samples + 1)
        x[:-1] = w.x_lag; y[:-1] = w.y_lag; y[1:] = w.y_t
        te_obs.append(estimator.estimate(x, y))
        x_list.append(x)
        y_list.append(y)
        
    te_obs_mean = np.mean(te_obs)
    
    null_dist = []
    for _ in range(n_permutations):
        perm = rng.permutation(len(windows))
        te_null = []
        for i, j in enumerate(perm):
            if i == j: j = (j + 1) % len(windows)
            te_null.append(estimator.estimate(x_list[j], y_list[i]))
        null_dist.append(np.mean(te_null))
        
    p_value = np.mean(np.array(null_dist) >= te_obs_mean)
    
    return {
        "te_observed": te_obs_mean,
        "null_distribution": null_dist,
        "p_value": p_value,
        "significant": p_value < 0.05
    }

def bootstrap_ci(values: np.ndarray, n_bootstrap: int = 500, confidence: float = 0.95, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))
    
    alpha = 1.0 - confidence
    return np.percentile(means, alpha / 2 * 100), np.percentile(means, (1 - alpha / 2) * 100)

def run_sanity_check(windows_forward: list, windows_backward: list, estimator, n_bootstrap: int = 500, seed: int = 42) -> dict:
    res = bidirectional_te(windows_forward, windows_backward, estimator)
    ci_f = bootstrap_ci(res["te_forward_values"], n_bootstrap, seed=seed)
    ci_b = bootstrap_ci(res["te_backward_values"], n_bootstrap, seed=seed)
    
    diff = np.array(res["te_forward_values"]) - np.array(res["te_backward_values"])
    ci_diff = bootstrap_ci(diff, n_bootstrap, seed=seed)
    
    passed = (res["te_forward_mean"] > res["te_backward_mean"]) and (ci_diff[0] > 0)
    
    if passed:
        print("Sanity check PASSED: TE(tim->nao) lon hon ro ret so voi TE(nao->tim).")
    else:
        print("Sanity check FAILED. Vui long kiem tra:")
        print("1. Dong bo hoa thoi gian\n2. Loc ectopic\n3. Nhieu tim\n4. Do dai cua so\n5. Doi chieu KSG")
        
    res.update({
        "ci_forward": ci_f,
        "ci_backward": ci_b,
        "ci_difference": ci_diff,
        "passed": passed
    })
    return res
