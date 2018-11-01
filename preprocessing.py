from typing import Tuple, List

from scipy.sparse.csr import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class NLPProcessor:
    def __init__(self,
                 vectorizer: str='tf-idf',
                 ngram_range: Tuple[int]=(1, 1),
                 stop_words: str=None,
                 max_features: int=None
                 ) -> None:
        """
        Initialize processor.
        :param vectorizer: choice of vectorizer
        :param ngram_range: pass through for vectorizer, e.g. (1, 2) means uni-gram and bi-grams
        :param stop_words: pass through for vectorizer
        :param max_features: pass through for vectorizer, bounds the number of tokens
        """

        vectorizers = {'tf-idf': TfidfVectorizer, 'count': CountVectorizer}

        if vectorizer not in vectorizers:
            raise ValueError(f"{vectorizer} is not a valid choice of"
                             f"vectorizer. pick one of: {len(vectorizers.keys())}")

        self.vec = vectorizers[vectorizer](ngram_range=ngram_range, stop_words=stop_words, max_features=max_features)

    def fit_transform(self, corpus: List[str]) -> csr_matrix:
        """
        Take a list of strings (docs) and returns a vector representation.
        :param corpus: list of docs
        :return: sparse scipy csr_matrix
        """
        return self.vec.fit_transform(corpus)
