import unittest
import pandas as pd


class AggregationTests(unittest.TestCase):
    def test_hs6_to_hs2_value_conservation(self):
        hs6=pd.DataFrame({"j":[1,1,1],"i":[2,3,2],"hs6":["010110","010190","020110"],"v":[1.0,2.0,3.0]})
        hs6["hs2"]=hs6.hs6.str[:2]
        hs2=hs6.groupby(["j","hs2"],as_index=False).v.sum()
        self.assertAlmostEqual(hs2.v.sum(),hs6.v.sum())
        self.assertEqual(hs2.loc[hs2.hs2.eq("01"),"v"].iloc[0],3.0)


if __name__=="__main__": unittest.main()
