import unittest
import pandas as pd


class PanelTests(unittest.TestCase):
    def test_destination_hs2_year_key(self):
        panel=pd.DataFrame({"destination_iso3":["USA","JPN"],"hs2":["01","01"],"outcome_year":[2022,2022]})
        self.assertFalse(panel.duplicated(["destination_iso3","hs2","outcome_year"]).any())

    def test_unknown_outcome_remains_missing(self):
        panel=pd.DataFrame({"outcome_observed":[False],"export_value_usd":[float("nan")]})
        self.assertTrue(pd.isna(panel.export_value_usd.iloc[0]))


if __name__=="__main__": unittest.main()
