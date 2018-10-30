from typing import List, Tuple

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, homogeneity_completeness_v_measure, \
    silhouette_score


# references
# https://datascienceplus.com/evaluation-of-topic-modeling-topic-coherence/
# http://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation

class GroundTruthMetrics:
    def __init__(self, labels: List[int], preds: List[int]) -> None:
        self.labels = labels
        self.preds = preds

    def adjusted_rand_index(self) -> float:
        """
        [-1, 1] score not not sensitive to change in label values.
        Perfect labeling gets 1.0,  independent labeling gets 0
        https://bit.ly/2yR5cOj
        :return: score
        """
        return adjusted_rand_score(self.labels, self.preds)

    def adjusted_mutual_info_score(self) -> float:
        """
        (-\infty, 1] score, normalized against chance, MI measures the agreement of 2 assignments.
        https://bit.ly/2S7GORg
        :return: score
        """
        return adjusted_mutual_info_score(self.labels, self.preds)

    def homogeneity_completeness_v_measure(self) -> Tuple[float]:
        """
        https://bit.ly/2yvI1JM
        :return: (homogeneity, completeness, v-measure)
        """
        return homogeneity_completeness_v_measure(self.labels, self.preds)


class NoGroundTruthMetrics:
    def __init__(self, x, labels):
        self.x = x
        self.labels = labels

    def silhouette_score(self, metric='euclidean') -> float:
        """
        [-1, 1]
        :param metric:
        :return: score
        """
        return silhouette_score(self.x, self.labels, metric)
