from spreadsheet_wrangler import cluster
import unittest
import pandas as pd

class TestCluster(unittest.TestCase):
    # cluster unique by cola
    def test_single_column_cluster(self):
        df = pd.DataFrame({"cola":[1,2,3,4,5,6,6], "unique":[1,1,1,2,2,2,3], "additional":list(range(7))})
        clustered = cluster(df, ["cola"], "unique")

        # takes the last seen value of the none clustered values
        expected = pd.DataFrame({"cola":[1,2,3,4,5,6], "unique":[(1,),(1,),(1,),(2,),(2,),(2,3)], "additional":[0,1,2,3,4,6]})

        for column in df.columns:
            for l, r in zip(clustered[column], expected[column]):
                self.assertEqual(l, r)

    # cluster unique by cola and additional
    def test_multi_column_unique_cluster(self):
        df = pd.DataFrame({"cola":[1,2,3,4,5,6,6], "unique":[1,1,1,2,2,2,3], "additional":list(range(7))})
        clustered = cluster(df, ["cola", "additional"], "unique")

        # takes the last seen value of the none clustered values
        expected = pd.DataFrame({"cola":[1,2,3,4,5,6,6], "unique":[(1,),(1,),(1,),(2,),(2,),(2,),(3,)], "additional":[0,1,2,3,4,5,6]})

        for column in df.columns:
            for l, r in zip(clustered[column], expected[column]):
                self.assertEqual(l, r)

    # cluster unique by cola and additional with duplicate lines
    def test_multi_column_unique_cluster(self):
        df = pd.DataFrame({"cola":[1,2,3,4,5,6,6], "unique":[1,1,1,2,2,2,3], "additional":[0,1,2,3,4,5,5]})
        clustered = cluster(df, ["cola", "additional"], "unique")

        # takes the last seen value of the none clustered values
        expected = pd.DataFrame({"cola":[1,2,3,4,5,6], "unique":[(1,),(1,),(1,),(2,),(2,),(2,3)], "additional":[0,1,2,3,4,5]})

        for column in df.columns:
            for l, r in zip(clustered[column], expected[column]):
                self.assertEqual(l, r)

if __name__ == "__main__":
    unittest.main()
