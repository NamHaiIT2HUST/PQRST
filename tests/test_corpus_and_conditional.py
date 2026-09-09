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
from pqrst.data.synthetic.corpus import generate_corpus, split_corpus_by_seed, save_corpus, load_corpus
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
