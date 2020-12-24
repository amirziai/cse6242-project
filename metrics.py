from typing import List, Tuple

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, homogeneity_completeness_v_measure, \
    silhouette_score
	
import numpy as np
import pandas as pd
import re, gensim
from gensim import corpora
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity as cs
from sklearn.feature_extraction.text import CountVectorizer



# references
# https://datascienceplus.com/evaluation-of-topic-modeling-topic-coherence/
# http://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
# https://www.kaggle.com/hengqujushi/cosine-similarity-nlp

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
		
class DocumentSimilarity:
    ##get cosine similarity
    def get_cosine_similarity(cos1, cos2):
        cosine_sim = []
        for i,j in zip(cos1,cos2):
            sim = cs(i,j)
            cosine_sim.append(sim[0][0])
        return cosine_sim
