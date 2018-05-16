import logging

from keras import Input, Model
from keras.layers import Dense

from models.neural_nets.neural_net_classifier import NeuralNetClassifier
from utils.utils import THREAD_FEATURES_COUNT


class ThreadClassifier(NeuralNetClassifier):
    def __init__(self, hidden_layer_size):
        super().__init__()
        logging.info('Thread classifier')
        logging.info('hidden_layer_size = {}'.format(hidden_layer_size))

        thread_features = Input(shape=(THREAD_FEATURES_COUNT,), name='thread_features')

        fc = Dense(hidden_layer_size, activation='relu')(thread_features)
        output = Dense(1, activation='sigmoid', name='output')(fc)

        self.model = Model(inputs=[thread_features], outputs=[output])
        self.compile_model()

    def pretrain(self, data_reader, ids):
        pass

    def transform_input(self, data):
        thread_features = data[['position', 'relative_position']].as_matrix()
        return [thread_features]
