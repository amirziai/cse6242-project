import unittest

import metrics
import numpy as np
import pandas as pd

import preprocessing



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

    def test_similarity_metrocs(self):
        corpus = ["The dog ran over the cat","the movie is about a dog","the book was made into a movie","this is fun","The dog and the cat are over"]
        # ["The dog ran over the cat","the movie is about a dog","the book was made into a movie","this is fun","The dog and the cat are over"]
        # ['this is', 'it is']
        vec = preprocessing.NLPProcessor()
        out = vec.fit_transform(corpus)

        cosine_sim_matrix = pd.DataFrame(0., index=range(len(corpus)), columns=range(len(corpus)))

        for i in range(0,len(corpus)):
            for j in range(0,len(corpus)):
                cosine_sim_matrix.at[i, j] = metrics.DocumentSimilarity.get_cosine_similarity(out[i], out[j])[0]

        print(cosine_sim_matrix)


if __name__ == '__main__':
    unittest.main()
