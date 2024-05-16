import unittest

import pandas as pd

from spreadsheet_wrangler import cluster


class TestCluster(unittest.TestCase):
    # cluster unique by cola
    def test_single_column_cluster(self):
        df = pd.DataFrame(
            {
                "cola": [1, 2, 3, 4, 5, 6, 6],
                "unique": [1, 1, 1, 2, 2, 2, 3],
                "additional": list(range(7)),
            }
        )
        clustered = cluster(df, ["cola"], "unique")

        # takes the last seen value of the none clustered values
        expected = pd.DataFrame(
            {
                "cola": [1, 2, 3, 4, 5, 6],
                "unique": [(1,), (1,), (1,), (2,), (2,), (2, 3)],
                "additional": [0, 1, 2, 3, 4, 6],
            }
        )

        for column in df.columns:
            for left, right in zip(clustered[column], expected[column], strict=False):
                assert left == right

    # cluster unique by cola and additional
    def test_multi_column_unique_cluster(self):
        df = pd.DataFrame(
            {
                "cola": [1, 2, 3, 4, 5, 6, 6],
                "unique": [1, 1, 1, 2, 2, 2, 3],
                "additional": list(range(7)),
            }
        )
        clustered = cluster(df, ["cola", "additional"], "unique")

        # takes the last seen value of the none clustered values
        expected = pd.DataFrame(
            {
                "cola": [1, 2, 3, 4, 5, 6, 6],
                "unique": [(1,), (1,), (1,), (2,), (2,), (2,), (3,)],
                "additional": [0, 1, 2, 3, 4, 5, 6],
            }
        )

        for column in df.columns:
            for left, right in zip(clustered[column], expected[column], strict=False):
                assert left == right


if __name__ == "__main__":
    unittest.main()
