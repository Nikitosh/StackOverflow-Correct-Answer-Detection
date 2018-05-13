import logging

from sklearn.feature_extraction.text import TfidfVectorizer


class TfIdfVectorizerAdapter:
    def __init__(self, **kwargs):
        if 'ngram_range' in kwargs:
            logging.info('Tf-Idf with bigrams vectorizer')
        else:
            logging.info('Tf-Idf vectorizer')
        self.vectorizer = TfidfVectorizer(**kwargs)

    def fit(self, data_reader, ids):
        self.vectorizer.fit(data_reader.get_stemmed_texts(ids))

    def transform(self, X):
        return self.vectorizer.transform(X)
