import unittest
import pandas as pd
from temporal_validation import assert_no_leakage


class LeakageTests(unittest.TestCase):
    def test_declared_two_year_gap(self):
        train=pd.DataFrame({"outcome_year":[2018,2020],"feature_year":[2016,2018]})
        test=pd.DataFrame({"outcome_year":[2022],"feature_year":[2020]})
        assert_no_leakage(train,test,2022,2)

    def test_future_feature_is_rejected(self):
        train=pd.DataFrame({"outcome_year":[2020],"feature_year":[2018]})
        test=pd.DataFrame({"outcome_year":[2022],"feature_year":[2021]})
        with self.assertRaises(AssertionError): assert_no_leakage(train,test,2022,2)


if __name__=="__main__": unittest.main()
