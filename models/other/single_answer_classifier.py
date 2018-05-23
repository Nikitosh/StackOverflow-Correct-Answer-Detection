import logging

import numpy as np


class SingleAnswerClassifier:
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
        for answer_count, relative_position in zip(X.answer_count, X.age_pos):
            result.append(int(answer_count == 1 and relative_position == 0))
        return np.array(result)

    def save(self, epoch):
        pass