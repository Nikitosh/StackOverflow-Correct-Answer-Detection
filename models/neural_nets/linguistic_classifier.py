import logging

from keras import Input, Model
from keras.layers import Dense

from models.neural_nets.neural_net_classifier import NeuralNetClassifier


class LinguisticClassifier(NeuralNetClassifier):
    def __init__(self, hidden_layer_size):
        super().__init__()
        logging.info('Linguistic classifier')
        logging.info('hidden_layer_size = {}'.format(hidden_layer_size))

        self.hidden_layer_size = hidden_layer_size

    def create_model(self):
        linguistic_features = Input(shape=(NeuralNetClassifier.LINGUISTIC_FEATURES_COUNT,), name='linguistic_features')

        fc = Dense(self.hidden_layer_size, activation='relu')(linguistic_features)
        output = Dense(1, activation='sigmoid', name='output')(fc)

        self.model = Model(inputs=[linguistic_features], outputs=[output])
        self.compile_model()

    def pretrain(self, data_reader, ids):
        self.create_model()

    def transform_input(self, data):
        linguistic_features = self.get_linguistic_features(data)
        return [linguistic_features]
