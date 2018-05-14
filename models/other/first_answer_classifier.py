import logging

import numpy as np


class FirstAnswerClassifier:
    def __init__(self):
        logging.info('First answer classifier')

    def pretrain(self, data_reader, ids):
        pass

    def train_on_batch(self, X, y):
        return [0, 0]

    def evaluate(self, X, y):
        return [0, 0]

    def predict(self, X):
        result = []
        for position in X['position']:
            result.append(int(position == 1))
        return np.array(result)

    def save(self, epoch):
        pass