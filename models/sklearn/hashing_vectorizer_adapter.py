import logging

from sklearn.feature_extraction.text import HashingVectorizer


class HashingVectorizerAdapter:
    def __init__(self, **kwargs):
        logging.info('Hashing vectorizer')
        self.vectorizer = HashingVectorizer(**kwargs)

    def fit(self, data_reader, ids):
        pass

    def transform(self, X):
        return self.vectorizer.transform(X)