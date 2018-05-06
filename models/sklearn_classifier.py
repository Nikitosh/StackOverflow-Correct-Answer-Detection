import numpy as np

from models.utils import stem_text


class SKLearnClassifier:
    def __init__(self, classifier, vectorizer):
        self.classifier = classifier
        self.vectorizer = vectorizer

    def pretrain(self, data_reader, ids):
        self.vectorizer.fit(data_reader, ids)

    def train_on_batch(self, X, y):
        all_classes = np.array([0, 1])
        X = self.vectorizer.transform(stem_text(text) for text in X['body'])
        self.classifier.partial_fit(X, y, classes=all_classes)

    def predict(self, X):
        X = self.vectorizer.transform(stem_text(text) for text in X['body'])
        return self.classifier.predict(X)
