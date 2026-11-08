import unittest

import pandas as pd

from training.dataset import grouped_split_assignments


class DatasetSplitTests(unittest.TestCase):
    def test_host_groups_never_cross_splits_and_assignments_repeat(self):
        frame = pd.DataFrame(
            {
                "host_id": [f"KIC {host}" for host in (1, 1, 2, 3, 3, 4, 5, 6, 7, 8)],
                "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            }
        )
        first = grouped_split_assignments(frame, seed=17)
        second = grouped_split_assignments(frame, seed=17)
        self.assertEqual(first, second)
        for host, rows in frame.groupby("host_id"):
            self.assertEqual(len({first[host] for _ in rows.index}), 1)
        self.assertTrue(set(first.values()).issubset({"train", "validation", "test"}))


if __name__ == "__main__":
    unittest.main()
