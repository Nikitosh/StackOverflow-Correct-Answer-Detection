from sklearn.feature_extraction.text import TfidfVectorizer


class TfIdfVectorizerAdapter:
    def __init__(self, **kwargs):
        self.vectorizer = TfidfVectorizer(**kwargs)

    def fit(self, data_reader, ids):
        self.vectorizer.fit(data_reader.get_texts(ids))

    def transform(self, X):
        return self.vectorizer.transform(X)