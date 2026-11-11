"""Tests for the 2026-11 revision: NDCG fix, rank fusion, LightGBM bagging, inference, training population."""
import unittest

import numpy as np
import pandas as pd

import common
import fusion
import market_discovery
import ml_models
import statistics as st


class NdcgFix(unittest.TestCase):
    def test_miss_in_top_k_scores_zero_not_nan(self):
        # one success, ranked fifth: a top-2 list misses it -> 0, not NaN
        self.assertEqual(st.ndcg_at_k(np.array([0, 0, 0, 0, 1]), 2), 0.0)

    def test_undefined_only_without_any_relevant_item(self):
        self.assertTrue(np.isnan(st.ndcg_at_k(np.zeros(6), 3)))

    def test_hand_computed_value(self):
        # relevant at positions 2 and 3, ideal at 1 and 2 (binary gain)
        rel = np.array([0, 1, 1, 0])
        expected = (1 / np.log2(3) + 1 / np.log2(4)) / (1 + 1 / np.log2(3))
        self.assertAlmostEqual(st.ndcg_at_k(rel, 4), expected)


class Fusion(unittest.TestCase):
    def setUp(self):
        self.pool = pd.DataFrame({"destination_iso3": ["AAA", "BBB", "CCC", "DDD", "EEE"],
                                  "p": [5.0, 4.0, 3.0, 2.0, 1.0],      # ranks 1..5
                                  "l": [0.2, 0.9, 0.1, 0.8, 0.3]})     # ranks 3,1,4,2,5

    def test_scores_in_unit_interval(self):
        g = fusion.fuse_pool(self.pool, ["p", "l"])
        self.assertTrue(((g.fused_score >= 0) & (g.fused_score <= 1)).all())
        self.assertAlmostEqual(fusion.fuse_pool(self.pool, ["p", "p"]).fused_score.max(), 1.0)

    def test_same_order_as_archived_formula(self):
        g = fusion.fuse_pool(self.pool, ["p", "l"], tie_rule="iso3")
        n = len(g)
        old = .5 * ((n - g["_rank_p"] + 1) / (n - 1) + (n - g["_rank_l"] + 1) / (n - 1))
        old_order = g.assign(old=old).sort_values(["old", "destination_iso3"], ascending=[False, True]).destination_iso3.tolist()
        new_order = g.sort_values("fused_rank").destination_iso3.tolist()
        self.assertEqual(old_order, new_order)

    def test_tie_rules(self):
        # BBB (p rank 2, l rank 1) and AAA (p rank 1, l rank 3) -> AAA 0.75, BBB 0.875; make an exact tie instead:
        pool = pd.DataFrame({"destination_iso3": ["ZZZ", "AAA", "MMM"], "p": [3.0, 1.0, 2.0], "l": [1.0, 3.0, 2.0]})
        iso = fusion.fuse_pool(pool, ["p", "l"], tie_rule="iso3").sort_values("fused_rank").destination_iso3.tolist()
        first = fusion.fuse_pool(pool, ["p", "l"], tie_rule="first_component").sort_values("fused_rank").destination_iso3.tolist()
        self.assertEqual(iso[0], "AAA")          # all three tie at 0.5; ISO3 decides
        self.assertEqual(first[0], "ZZZ")        # PPML rank 1 decides
        with self.assertRaises(ValueError):
            fusion.fuse_pool(pool, ["p", "l"], tie_rule="alphabet")


class LightgbmBagging(unittest.TestCase):
    def _predict(self, seed):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(600, 6)); y = (x[:, 0] + rng.normal(size=600) > 0.5).astype(int)
        params = ml_models.model_configs("LightGBM", classification=True, n_trials=1)[0] | {"n_jobs": 1}
        model = ml_models.build_estimator("LightGBM", params, seed=seed, classification=True).fit(x, y)
        return model.predict_proba(x)[:, 1]

    def test_seeds_differ_when_bagging_is_on(self):
        self.assertTrue(ml_models.lightgbm_bagging_params())
        self.assertGreater(np.abs(self._predict(11) - self._predict(23)).max(), 0)

    def test_deterministic_without_bagging(self):
        saved = dict(common.CFG["compute"])
        try:
            common.CFG["compute"].update(lightgbm_subsample=1.0, lightgbm_subsample_freq=0, lightgbm_colsample_bytree=1.0)
            self.assertEqual(ml_models.lightgbm_bagging_params(), {})
            self.assertEqual(np.abs(self._predict(11) - self._predict(23)).max(), 0)
        finally:
            common.CFG["compute"].clear(); common.CFG["compute"].update(saved)

    def test_constant_feature_is_excluded(self):
        self.assertNotIn("common_official_language", ml_models.feature_columns()[0])


class Inference(unittest.TestCase):
    def test_year_signflip_floor(self):
        years = np.repeat(np.arange(2018, 2025), 3)
        out = st.year_signflip_p(np.full(len(years), 0.05), years)
        self.assertEqual(out["years_positive"], 7)
        self.assertAlmostEqual(out["p_year_signflip"], 2 / 2 ** 7)
        three = st.year_signflip_p(np.full(9, 0.05), np.repeat([2022, 2023, 2024], 3))
        self.assertAlmostEqual(three["p_year_signflip"], 0.25)

    def test_paired_contrast_inference_runs(self):
        rng = np.random.default_rng(1)
        blocks = np.repeat(np.arange(20), 7); years = np.tile(np.arange(2018, 2025), 20)
        out = st.paired_contrast_inference(0.03 + rng.normal(0, 0.05, len(blocks)), blocks, years, reps=500)
        self.assertTrue(out["ci_low"] < out["estimate"] < out["ci_high"])
        self.assertTrue(np.isfinite(out["t_twoway"]))


class TrainingPopulation(unittest.TestCase):
    def test_strict_is_subset(self):
        f = pd.DataFrame({"entry_eligible": [True, True, False], "entry_strict_inactive": [True, False, True]})
        saved = common.CFG["study"].get("entry_training_population")
        try:
            common.CFG["study"]["entry_training_population"] = "strict_inactive"
            self.assertEqual(market_discovery.entry_training_mask(f).tolist(), [True, False, False])
            common.CFG["study"]["entry_training_population"] = "threshold_inactive"
            self.assertEqual(market_discovery.entry_training_mask(f).tolist(), [True, True, False])
        finally:
            common.CFG["study"]["entry_training_population"] = saved


if __name__ == "__main__":
    unittest.main()
