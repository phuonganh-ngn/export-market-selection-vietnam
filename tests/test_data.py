import unittest
import numpy as np
from statistics import forecast_metrics


class DataIntegrityTests(unittest.TestCase):
    def test_export_metric_nonnegative_and_zero_flow_split(self):
        metrics=forecast_metrics(np.array([0.0,100.0]),np.array([10.0,80.0]))
        self.assertGreaterEqual(metrics["mae_usd"],0)
        self.assertEqual(metrics["mae_zero_usd"],10.0)
        self.assertEqual(metrics["mae_positive_usd"],20.0)

    def test_wape_undefined_for_zero_denominator(self):
        metrics=forecast_metrics(np.array([0.0,0.0]),np.array([0.0,0.0]))
        self.assertTrue(np.isnan(metrics["wape"]))


if __name__=="__main__": unittest.main()
