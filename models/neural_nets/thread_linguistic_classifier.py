import logging

import numpy as np
from keras import Input, Model
from keras.layers import Dense, concatenate

from models.neural_nets.neural_net_classifier import NeuralNetClassifier
from utils.utils import THREAD_FEATURES_COUNT


class ThreadLinguisticClassifier(NeuralNetClassifier):
    def __init__(self, hidden_layer_size):
        super().__init__()
        logging.info('Thread linguistic classifier')
        logging.info('hidden_layer_size = {}'.format(hidden_layer_size))

        linguistic_features = Input(shape=(self.linguistic_features_calculator.LINGUISTIC_FEATURES_COUNT,),
                                    name='linguistic_features')
        thread_features = Input(shape=(THREAD_FEATURES_COUNT,), name='thread_features')

        features = concatenate([linguistic_features, thread_features], name='features')

        fc = Dense(hidden_layer_size, activation='relu')(features)
        output = Dense(1, activation='sigmoid', name='output')(fc)

        self.model = Model(inputs=[linguistic_features, thread_features], outputs=[output])

        self.compile_model()

    def transform_input(self, data):
        linguistic_features = np.array([self.linguistic_features_calculator.get_normalized_linguistic_features(id, X)
                                        for id, X in zip(data.id, data.body)])
        thread_features = data[['position', 'relative_position']].as_matrix()
        return [linguistic_features, thread_features]
