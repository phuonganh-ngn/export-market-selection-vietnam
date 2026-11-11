import unittest
import numpy as np
from statistics import precision_at_k, ndcg_at_k


class RankingTests(unittest.TestCase):
    def test_top_k_and_ideal_ndcg(self):
        rel=np.array([1,1,0,0])
        self.assertEqual(precision_at_k(rel,2),1.0)
        self.assertAlmostEqual(ndcg_at_k(rel,4),1.0)

    def test_ndcg_undefined_without_relevant_cases(self):
        self.assertTrue(np.isnan(ndcg_at_k(np.zeros(5),5)))


if __name__=="__main__": unittest.main()
