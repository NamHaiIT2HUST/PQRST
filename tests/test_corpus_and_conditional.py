"""Test cho corpus (Pha R) va phan ra conditional MI. Xem checklist trong
docs/PHASE_R_GUIDE.md muc 7.

Cac test duoi day la KHUNG (skeleton) - than ham con thieu, ban tu dien logic assert.
"""

import numpy as np
import pytest
import torch
import tempfile
import os
import shutil
from pqrst.data.synthetic.var_linear_gaussian import generate_var_linear_gaussian, compute_var_linear_ground_truths
from pqrst.data.synthetic.corpus import (
    generate_corpus, generate_corpus_periodic, split_corpus_by_seed, save_corpus, load_corpus,
)
from pqrst.estimators.mine.amortized import MaskedStatisticsNetwork, AmortizedTrainConfig, AmortizedTEEstimator
from pqrst.estimators.mine.conditional import estimate_mi_from_window, estimate_te_from_window
from pqrst.estimators.mine.network import StatisticsNetwork
from pqrst.baselines.base import BaseTEEstimator

class TestGroundTruths:
    def test_decomposition_identity_holds(self):
        a, b, c, noise_std = 0.5, 0.5, 0.6, 0.5
        _, _, te_indep = generate_var_linear_gaussian(10, a, b, c, noise_std, seed=42)
        gt = compute_var_linear_ground_truths(a, b, c, noise_std)
        assert abs((gt["mi_full"] - gt["mi_reduced"]) - te_indep) < 1e-9

    def test_ground_truths_across_parameter_grid(self):
        for c in [0.0, 0.2, 0.5]:
            for noise_std in [0.1, 0.5, 1.0]:
                _, _, te_indep = generate_var_linear_gaussian(10, 0.5, 0.5, c, noise_std, seed=42)
                gt = compute_var_linear_ground_truths(0.5, 0.5, c, noise_std)
                assert abs((gt["mi_full"] - gt["mi_reduced"]) - te_indep) < 1e-9

    def test_te_zero_when_uncoupled(self):
        gt = compute_var_linear_ground_truths(0.5, 0.5, 0.0, 0.5)
        assert abs(gt["te"]) < 1e-12

    def test_te_increases_with_coupling(self):
        te_vals = []
        for c in [0.1, 0.3, 0.6]:
            gt = compute_var_linear_ground_truths(0.5, 0.5, c, 0.5)
            te_vals.append(gt["te"])
        assert te_vals[0] < te_vals[1] < te_vals[2]


class TestCorpus:
    def test_corpus_size_matches_grid(self):
        corpus = generate_corpus([0.1, 0.2], [0.1, 0.5], [10, 20], 3, 0.5, 0.5, 42)
        assert len(corpus) == 2 * 2 * 2 * 3

    def test_window_shapes_consistent(self):
        corpus = generate_corpus([0.1], [0.1], [10, 20], 1, 0.5, 0.5, 42)
        for w in corpus:
            assert len(w.y_t) == w.n_samples
            assert len(w.x_lag) == w.n_samples
            assert len(w.y_lag) == w.n_samples

    def test_all_seeds_unique(self):
        corpus = generate_corpus([0.1, 0.2], [0.1, 0.5], [10, 20], 3, 0.5, 0.5, 42)
        seeds = [w.seed for w in corpus]
        assert len(set(seeds)) == len(seeds)

    def test_corpus_reproducible(self):
        c1 = generate_corpus([0.1], [0.1], [10], 2, 0.5, 0.5, 42)
        c2 = generate_corpus([0.1], [0.1], [10], 2, 0.5, 0.5, 42)
        assert np.array_equal(c1[0].y_t, c2[0].y_t)

    def test_split_is_disjoint_by_window(self):
        corpus = generate_corpus([0.1], [0.1], [10], 10, 0.5, 0.5, 42)
        train_w, val_w = split_corpus_by_seed(corpus, 0.2, 42)
        
        train_seeds = set(w.seed for w in train_w)
        val_seeds = set(w.seed for w in val_w)
        
        assert len(train_seeds.intersection(val_seeds)) == 0
        assert len(train_seeds) + len(val_seeds) == len(corpus)

    def test_save_load_roundtrip(self):
        corpus = generate_corpus([0.1], [0.1], [10], 2, 0.5, 0.5, 42)
        tmpdir = tempfile.mkdtemp()
        try:
            p = os.path.join(tmpdir, "corpus.npz")
            save_corpus(corpus, p)
            loaded = load_corpus(p)
            
            assert len(corpus) == len(loaded)
            np.testing.assert_array_equal(corpus[0].y_t, loaded[0].y_t)
            assert corpus[0].seed == loaded[0].seed
        finally:
            shutil.rmtree(tmpdir)


class TestCorpusPeriodic:
    """generate_corpus_periodic (vong thu nghiem du lieu phi tuyen, xem
    docs/PHASE_R_REPORT.md muc 9)."""

    def test_corpus_size_matches_grid(self):
        pgt = {(0.0, 0.1): 0.0, (0.3, 0.1): 0.05}
        corpus = generate_corpus_periodic(
            coupling_values=[0.0, 0.3], noise_values=[0.1], n_values=[10, 20],
            n_windows_per_cell=3, omega_x=0.3, omega_y=0.31,
            pseudo_ground_truths=pgt, base_seed=1,
        )
        assert len(corpus) == 2 * 1 * 2 * 3

    def test_ground_truth_zero_when_uncoupled(self):
        pgt = {(0.0, 0.1): 0.0, (0.3, 0.1): 0.05}
        corpus = generate_corpus_periodic(
            coupling_values=[0.0, 0.3], noise_values=[0.1], n_values=[10],
            n_windows_per_cell=2, omega_x=0.3, omega_y=0.31,
            pseudo_ground_truths=pgt, base_seed=1,
        )
        uncoupled = [w for w in corpus if w.params["c"] == 0.0]
        assert all(w.te_ground_truth == 0.0 for w in uncoupled)

    def test_all_seeds_unique(self):
        pgt = {(0.0, 0.1): 0.0, (0.3, 0.1): 0.05, (0.6, 0.1): 0.1}
        corpus = generate_corpus_periodic(
            coupling_values=[0.0, 0.3, 0.6], noise_values=[0.1], n_values=[10, 20],
            n_windows_per_cell=3, omega_x=0.3, omega_y=0.31,
            pseudo_ground_truths=pgt, base_seed=1,
        )
        seeds = [w.seed for w in corpus]
        assert len(set(seeds)) == len(seeds)

    def test_mi_ground_truths_are_nan(self):
        """Khong co baseline MI da bien de tinh pseudo ground-truth cho mi_full/
        mi_reduced tren du lieu periodic - phai la NaN (khong duoc bay bia gia tri)."""
        pgt = {(0.3, 0.1): 0.05}
        corpus = generate_corpus_periodic(
            coupling_values=[0.3], noise_values=[0.1], n_values=[10],
            n_windows_per_cell=1, omega_x=0.3, omega_y=0.31,
            pseudo_ground_truths=pgt, base_seed=1,
        )
        assert np.isnan(corpus[0].mi_full_ground_truth)
        assert np.isnan(corpus[0].mi_reduced_ground_truth)

    def test_save_load_roundtrip_preserves_nan(self):
        pgt = {(0.3, 0.1): 0.05}
        corpus = generate_corpus_periodic(
            coupling_values=[0.3], noise_values=[0.1], n_values=[10],
            n_windows_per_cell=1, omega_x=0.3, omega_y=0.31,
            pseudo_ground_truths=pgt, base_seed=1,
        )
        tmpdir = tempfile.mkdtemp()
        try:
            p = os.path.join(tmpdir, "periodic_corpus.npz")
            save_corpus(corpus, p)
            loaded = load_corpus(p)
            assert np.isnan(loaded[0].mi_full_ground_truth)
            np.testing.assert_array_equal(corpus[0].y_t, loaded[0].y_t)
        finally:
            shutil.rmtree(tmpdir)


class TestConditionalEstimation:
    def test_estimate_mi_from_window_finite(self):
        model = StatisticsNetwork(input_dim=2, hidden_dims=[16])
        a_vals = np.random.randn(20)
        b_vals = np.random.randn(20)
        val = estimate_mi_from_window(model, a_vals, b_vals, n_shuffles=5, seed=42)
        assert np.isfinite(val)

    def test_estimate_te_returns_all_three_terms(self):
        model = MaskedStatisticsNetwork([16])
        y_t = np.random.randn(20)
        x_lag = np.random.randn(20)
        y_lag = np.random.randn(20)
        res = estimate_te_from_window(model, y_t, x_lag, y_lag, 5, 42)
        
        assert set(res.keys()) == {"mi_full", "mi_reduced", "te"}
        assert np.isclose(res["te"], res["mi_full"] - res["mi_reduced"])

    def test_more_shuffles_reduces_variance(self):
        model = StatisticsNetwork(input_dim=2, hidden_dims=[16])
        a_vals = np.random.randn(20)
        b_vals = np.random.randn(20)
        
        vals_1 = []
        vals_50 = []
        
        for i in range(10):
            vals_1.append(estimate_mi_from_window(model, a_vals, b_vals, n_shuffles=1, seed=i))
            vals_50.append(estimate_mi_from_window(model, a_vals, b_vals, n_shuffles=50, seed=i))
            
        assert np.var(vals_50) < np.var(vals_1)


class TestAmortizedEstimator:
    def test_conforms_to_base_interface(self):
        model = MaskedStatisticsNetwork([16])
        config = AmortizedTrainConfig()
        est = AmortizedTEEstimator(model, config)
        
        assert isinstance(est, BaseTEEstimator)
        x = np.random.randn(20)
        y = np.random.randn(20)
        te = est.estimate(x, y)
        assert isinstance(te, float)

    def test_save_load_preserves_predictions(self):
        model = MaskedStatisticsNetwork([16])
        config = AmortizedTrainConfig(hidden_dims=[16])
        est = AmortizedTEEstimator(model, config)
        
        tmpdir = tempfile.mkdtemp()
        try:
            p = os.path.join(tmpdir, "model.pt")
            est.save(p)
            est2 = AmortizedTEEstimator.load(p)
            
            x = np.random.randn(20)
            y = np.random.randn(20)
            
            te1 = est.estimate(x, y, seed=42)
            te2 = est2.estimate(x, y, seed=42)
            assert np.isclose(te1, te2)
        finally:
            shutil.rmtree(tmpdir)


class TestGridReconstruction:
    """Bao ve chong lai loi da phat hien khi review: evaluate_estimators_on_grid
    dung lai (x, y) tu Window de goi estimator.estimate(x, y) - phai dam bao dung lai
    CHINH XAC x_lag/y_lag/y_t goc, khong bi lech chi so."""

    def test_reconstructed_arrays_match_original_generator_output(self):
        from pqrst.evaluation.grid import evaluate_estimators_on_grid
        from pqrst.baselines.base import BaseTEEstimator

        a, b, c, noise_std, N, seed = 0.5, 0.5, 0.6, 0.5, 30, 999
        x_true, y_true, te_gt = generate_var_linear_gaussian(N + 1, a, b, c, noise_std, seed)

        from pqrst.data.synthetic.corpus import Window
        w = Window(
            y_t=y_true[1:], x_lag=x_true[:-1], y_lag=y_true[:-1],
            n_samples=N, config_name="test", params={"a": a, "b": b, "c": c, "noise_std": noise_std},
            te_ground_truth=te_gt, mi_full_ground_truth=0.0, mi_reduced_ground_truth=0.0, seed=seed,
        )

        captured = {}

        class CapturingEstimator(BaseTEEstimator):
            def estimate(self, x, y, **kwargs):
                captured["x"] = x.copy()
                captured["y"] = y.copy()
                return 0.0

        evaluate_estimators_on_grid({"probe": CapturingEstimator()}, [w], progress=False)

        np.testing.assert_allclose(captured["x"][:-1], w.x_lag)
        np.testing.assert_allclose(captured["y"][:-1], w.y_lag)
        np.testing.assert_allclose(captured["y"][1:], w.y_t)

    def test_ksg_on_reconstructed_grid_matches_direct_call(self):
        """So sanh truc tiep: KSG tren (x,y) dung lai qua evaluate_estimators_on_grid
        phai cho DUNG so voi goi KSG truc tiep tren x_true, y_true goc."""
        import warnings
        warnings.filterwarnings("ignore")
        from pqrst.evaluation.grid import evaluate_estimators_on_grid
        from pqrst.baselines.ksg import KSGTEEstimator
        from pqrst.data.synthetic.corpus import Window

        a, b, c, noise_std, N, seed = 0.5, 0.5, 0.6, 0.5, 50, 7
        x_true, y_true, te_gt = generate_var_linear_gaussian(N + 1, a, b, c, noise_std, seed)
        te_direct = KSGTEEstimator().estimate(x_true, y_true)

        w = Window(
            y_t=y_true[1:], x_lag=x_true[:-1], y_lag=y_true[:-1],
            n_samples=N, config_name="test", params={"a": a, "b": b, "c": c, "noise_std": noise_std},
            te_ground_truth=te_gt, mi_full_ground_truth=0.0, mi_reduced_ground_truth=0.0, seed=seed,
        )
        raw = evaluate_estimators_on_grid({"KSG": KSGTEEstimator()}, [w], progress=False)
        te_via_grid = raw.iloc[0]["te_estimate"]

        assert abs(te_via_grid - te_direct) < 1e-9


class TestTrainAmortized:
    """Bao ve hanh vi 'deploy = trung binh tham so k epoch cuoi' (sua loi
    selection-bias phat hien khi review - xem docs/PHASE_R_REPORT.md muc 4).
    Chua co test nao goi truc tiep train_amortized() truoc day - day la khoang trong
    coverage thu 2 phat hien khi review (khoang trong thu 1 la grid.py, da vasa)."""

    def _make_small_corpus(self, seed=1):
        from pqrst.data.synthetic.corpus import generate_corpus, split_corpus_by_seed
        corpus = generate_corpus([0.3, 0.6], [0.3, 0.5], [10, 20], 20, 0.5, 0.5, base_seed=seed)
        return split_corpus_by_seed(corpus, 0.2, split_seed=seed)

    def test_runs_and_returns_valid_estimator(self):
        from pqrst.estimators.mine.amortized import train_amortized, AmortizedTrainConfig
        from pqrst.baselines.base import BaseTEEstimator

        train_w, val_w = self._make_small_corpus()
        cfg = AmortizedTrainConfig(hidden_dims=[16], windows_per_batch=8, max_epochs=5,
                                    patience=5, eval_n_shuffles=3,
                                    final_estimate_last_k_epochs=3, seed=1)
        est, hist = train_amortized(train_w, val_w, cfg)

        assert isinstance(est, BaseTEEstimator)
        assert len(hist["val_loss_history"]) == 5
        assert hist["n_epochs_averaged"] == 3

        x = np.random.default_rng(0).normal(size=15)
        y = np.random.default_rng(1).normal(size=15)
        te = est.estimate(x, y, seed=42)
        assert np.isfinite(te)

    def test_deployed_weights_are_average_of_last_k_epochs(self):
        """Verify truc tiep TOAN HOC: trong so model deploy phai bang trung binh
        cong cua k state_dict cuoi cung - khong duoc la 1 epoch don le (best-of-all)."""
        import copy
        from pqrst.estimators.mine.amortized import (
            train_amortized, AmortizedTrainConfig, MaskedStatisticsNetwork,
        )
        from pqrst.estimators.mine.losses import donsker_varadhan_loss
        import torch

        train_w, val_w = self._make_small_corpus(seed=2)
        cfg = AmortizedTrainConfig(hidden_dims=[8], windows_per_batch=8, max_epochs=4,
                                    patience=10, eval_n_shuffles=2,
                                    final_estimate_last_k_epochs=4, seed=3)

        # Chay lai chinh xac cung logic train nhu train_amortized, tu thu thap state_dict
        # cua tung epoch, roi tu tinh trung binh doc lap de doi chieu.
        torch.manual_seed(cfg.seed)
        model = MaskedStatisticsNetwork(hidden_dims=cfg.hidden_dims)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

        def prep(w):
            return (torch.tensor(w.y_t, dtype=torch.float32).unsqueeze(-1),
                    torch.tensor(w.x_lag, dtype=torch.float32).unsqueeze(-1),
                    torch.tensor(w.y_lag, dtype=torch.float32).unsqueeze(-1))

        collected_states = []
        for epoch in range(cfg.max_epochs):
            rng = np.random.default_rng(cfg.seed + epoch)
            perm = rng.permutation(len(train_w))
            for i in range(0, len(train_w), cfg.windows_per_batch):
                idx = perm[i:i + cfg.windows_per_batch]
                optimizer.zero_grad()
                group_losses = []
                for j in idx:
                    ty_t, tx_lag, ty_lag = prep(train_w[j])
                    N = ty_t.shape[0]
                    for mask_val in (1.0, 0.0):
                        mask = torch.full((N, 1), mask_val)
                        t_joint = model(ty_t, tx_lag, ty_lag, mask)
                        p = torch.randperm(N)
                        t_marg = model(ty_t, tx_lag[p], ty_lag[p], mask)
                        group_losses.append(donsker_varadhan_loss(t_joint, t_marg))
                if group_losses:
                    torch.stack(group_losses).mean().backward()
                    optimizer.step()
            collected_states.append({k: v.cpu().clone() for k, v in model.state_dict().items()})

        expected_avg = {
            key: torch.stack([sd[key].float() for sd in collected_states], dim=0).mean(dim=0)
            for key in collected_states[0].keys()
        }

        # train_amortized() voi CUNG seed/config phai cho ra dung trong so trung binh nay.
        est, hist = train_amortized(train_w, val_w, cfg)
        deployed = est.model.state_dict()

        for key in expected_avg:
            assert torch.allclose(deployed[key], expected_avg[key], atol=1e-5), f"mismatch at {key}"
