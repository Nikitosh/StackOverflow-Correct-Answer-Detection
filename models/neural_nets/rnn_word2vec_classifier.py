import logging

import numpy as np
from keras import Input, Model
from keras.layers import Dropout, Dense, concatenate

from models.neural_nets.neural_net_classifier import NeuralNetClassifier
from utils.utils import transform_sentence_batch_to_vector, lower_text, process_html, get_lstm, THREAD_FEATURES_COUNT


class RnnWord2VecClassifier(NeuralNetClassifier):
    def __init__(self, answer_body_words_count, lstm_embed_size, hidden_layer_size, bidirectional, dropout):
        super().__init__()
        logging.info('RNN Word2Vec classifier')
        logging.info('answer_body_words_count = {}, lstm_embed_size = {}, bidirectional = {}, dropout = {}'.format(
            answer_body_words_count, lstm_embed_size, bidirectional, dropout))

        self.answer_body_words_count = answer_body_words_count

        text_input = Input(shape=(self.answer_body_words_count, self.num_features), name='text_input')
        linguistic_features = Input(shape=(self.linguistic_features_calculator.LINGUISTIC_FEATURES_COUNT,),
                                    name='linguistic_features')
        thread_features = Input(shape=(THREAD_FEATURES_COUNT,), name='thread_features')

        rnn_features = get_lstm(lstm_embed_size, bidirectional, dropout)(text_input)
        rnn_features = Dropout(dropout, name='rnn_features')(rnn_features)
        features = concatenate([rnn_features, linguistic_features, thread_features], name='features')
        fc = Dense(hidden_layer_size, activation='relu')(features)
        output = Dense(1, activation='sigmoid', name='output')(fc)

        self.model = Model(inputs=[text_input, linguistic_features, thread_features], outputs=[output])
        self.compile_model()

    def transform_input(self, data):
        answer_body = transform_sentence_batch_to_vector(self.word_vectors,
                                               [lower_text(process_html(X)) for X in data['body']],
                                               self.answer_body_words_count, self.num_features)
        linguistic_features = np.array([self.linguistic_features_calculator.get_normalized_linguistic_features(id, X)
                                        for id, X in zip(data.id, data.body)])
        thread_features = data[['position', 'relative_position']].as_matrix()
        return [answer_body, linguistic_features, thread_features]