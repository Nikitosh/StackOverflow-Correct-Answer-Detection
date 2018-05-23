import logging

from keras import Input, Model
from keras.layers import Dense

from models.neural_nets.neural_net_classifier import NeuralNetClassifier


class ThreadLinguisticClassifier(NeuralNetClassifier):
    def __init__(self, hidden_layer_size):
        super().__init__()
        logging.info('Thread linguistic classifier')
        logging.info('hidden_layer_size = {}'.format(hidden_layer_size))

        self.hidden_layer_size = hidden_layer_size

    def create_model(self):
        other_features = Input(shape=(NeuralNetClassifier.OTHER_FEATURES_COUNT,), name='other_features')

        fc = Dense(self.hidden_layer_size, activation='relu')(other_features)
        output = Dense(1, activation='sigmoid', name='output')(fc)

        self.model = Model(inputs=[other_features], outputs=[output])

        self.compile_model()

    def pretrain(self, data_reader, ids):
        self.create_model()

    def transform_input(self, data):
        other_features = self.get_other_features(data)
        return [other_features]
