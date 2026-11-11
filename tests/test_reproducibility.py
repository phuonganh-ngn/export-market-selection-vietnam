import unittest
import numpy as np
from ml_models import validation_mae_scale


class ReproducibilityTests(unittest.TestCase):
    def test_fixed_seed_random_scores(self):
        self.assertTrue(np.array_equal(np.random.default_rng(73).random(20),np.random.default_rng(73).random(20)))

    def test_original_usd_mae_scale(self):
        pred_log=np.log1p(np.array([5.0,10.0]))
        self.assertAlmostEqual(validation_mae_scale(np.array([10.0,20.0]),pred_log),2.0)


if __name__=="__main__": unittest.main()
