import logging

import numpy as np


class MaxScoreAnswerClassifier:
    def __init__(self):
        logging.info('Max score answer classifier')

    def pretrain(self, data_reader, ids):
        pass

    def train_on_batch(self, X, y):
        return [0, 0]

    def evaluate(self, X, y):
        return [0, 0]

    def predict(self, X):
        result = []
        for relative_score in X['relative_score']:
            result.append(int(relative_score >= 1))
        return np.array(result)

    def save(self, epoch):
        pass