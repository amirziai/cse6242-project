import unittest

import metrics
import numpy as np


class UnitTests(unittest.TestCase):
    def test_metrics_ground_truth(self):
        m = metrics.GroundTruthMetrics([1, 1, 1], [1, 1, 1])
        self.assertEqual(m.adjusted_rand_index(), 1)
        self.assertEqual(m.adjusted_mutual_info_score(), 1)
        self.assertEqual(m.homogeneity_completeness_v_measure(), (1, 1, 1))

    def test_metrics_no_ground_truth(self):
        x = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [10, 10, 10],
            [10, 10, 10]
        ])
        m = metrics.NoGroundTruthMetrics(x, [1, 1, 2, 2])
        self.assertEqual(m.silhouette_score(), 1)


if __name__ == '__main__':
    unittest.main()
