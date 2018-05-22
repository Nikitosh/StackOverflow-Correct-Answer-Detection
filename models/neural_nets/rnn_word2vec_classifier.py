import logging

import numpy as np
from keras import Input, Model
from keras.layers import Dropout, Dense, concatenate

from models.neural_nets.neural_net_classifier import NeuralNetClassifier
from utils.utils import get_lstm, lower_text, process_html


class RnnWord2VecClassifier(NeuralNetClassifier):
    def __init__(self, answer_body_words_count, lstm_embed_size, hidden_layer_size, bidirectional, dropout):
        super().__init__()
        logging.info('RNN Word2Vec classifier')
        logging.info('answer_body_words_count = {}, lstm_embed_size = {}, bidirectional = {}, dropout = {}'.format(
            answer_body_words_count, lstm_embed_size, bidirectional, dropout))

        self.answer_body_words_count = answer_body_words_count
        self.lstm_embed_size = lstm_embed_size
        self.hidden_layer_size = hidden_layer_size
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.lstm_layer = None
        self.dropout_layer = None

    def create_model(self):
        self.lstm_layer = get_lstm(self.lstm_embed_size, self.bidirectional, self.dropout)
        self.dropout_layer = Dropout(self.dropout)

        answer_body_input = Input(shape=(self.answer_body_words_count,), name='answer_body_input')
        linguistic_features = Input(shape=(self.linguistic_features_calculator.LINGUISTIC_FEATURES_COUNT,),
                                    name='linguistic_features')

        answer_body_features = self.embedding(answer_body_input)
        rnn_features = self.lstm_layer(answer_body_features)
        rnn_features = self.dropout_layer(rnn_features)
        features = concatenate([rnn_features, linguistic_features], name='features')
        fc = Dense(self.hidden_layer_size, activation='relu')(features)
        output = Dense(1, activation='sigmoid', name='output')(fc)

        self.model = Model(inputs=[answer_body_input, linguistic_features], outputs=[output])
        self.compile_model()

    def transform_input(self, data):
        answer_body = self.transform_sentence_batch_to_vector([lower_text(process_html(X)) for X in data['body']],
                                                              self.answer_body_words_count)
        linguistic_features = np.array([self.linguistic_features_calculator.get_normalized_linguistic_features(id, X)
                                        for id, X in zip(data.id, data.body)])
        return [answer_body, linguistic_features]
