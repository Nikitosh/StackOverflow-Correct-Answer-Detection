import logging

import numpy as np
from keras import Input, Model
from keras.layers import Dropout, Dense, merge, concatenate

from models.neural_nets.neural_net_classifier import NeuralNetClassifier
from utils.utils import transform_sentence_batch_to_vector, lower_text, process_html, \
    THREAD_FEATURES_COUNT, get_lstm


class RnnWord2VecWithQuestionClassifier(NeuralNetClassifier):
    def __init__(self, question_title_words_count, question_body_words_count, answer_body_words_count, lstm_embed_size,
                 hidden_layer_size, bidirectional, dropout, mode):
        super().__init__()
        logging.info('RNN Word2Vec with question classifier')
        logging.info(
            'question_title_words_count = {}, question_body_words_count = {}, answer_body_words_count = {}, lstm_embed_size = {}, hidden_layer_size = {}, bidirectional = {}, dropout = {}, mode = {}'.format(
                question_title_words_count, question_body_words_count, answer_body_words_count, lstm_embed_size,
                hidden_layer_size, bidirectional, dropout, mode))

        self.question_title_words_count = question_title_words_count
        self.question_body_words_count = question_body_words_count
        self.answer_body_words_count = answer_body_words_count

        question_title_input = Input(shape=(self.question_title_words_count, self.num_features),
                                     name='question_title_input')
        question_body_input = Input(shape=(self.question_body_words_count, self.num_features),
                                    name='question_body_input')
        answer_body_input = Input(shape=(self.answer_body_words_count, self.num_features),
                                  name='answer_body_input')
        linguistic_features = Input(shape=(self.linguistic_features_calculator.LINGUISTIC_FEATURES_COUNT,),
                                    name='linguistic_features')
        thread_features = Input(shape=(THREAD_FEATURES_COUNT,), name='thread_features')

        question_title_rnn_features = get_lstm(lstm_embed_size, bidirectional, dropout)(question_title_input)
        question_title_rnn_features = Dropout(dropout, name='question_title_rnn_features')(question_title_rnn_features)
        question_body_rnn_features = get_lstm(lstm_embed_size, bidirectional, dropout)(question_body_input)
        question_body_rnn_features = Dropout(dropout, name='question_body_rnn_features')(question_body_rnn_features)
        answer_body_rnn_features = get_lstm(lstm_embed_size, bidirectional, dropout)(answer_body_input)
        answer_body_rnn_features = Dropout(dropout, name='answer_body_rnn_features')(answer_body_rnn_features)
        rnn_features = merge([question_title_rnn_features, question_body_rnn_features, answer_body_rnn_features],
                             mode=mode)
        rnn_features = Dropout(dropout, name='rnn_features')(rnn_features)
        features = concatenate([rnn_features, linguistic_features, thread_features], name='features')
        fc = Dense(hidden_layer_size, activation='relu')(features)
        output = Dense(1, activation='sigmoid', name='output')(fc)

        self.model = Model(
            inputs=[question_title_input, question_body_input, answer_body_input, linguistic_features, thread_features],
            outputs=[output])
        self.compile_model()

    def transform_input(self, data):
        question_title = transform_sentence_batch_to_vector(self.word_vectors,
                                                            [lower_text(process_html(X)) for X in
                                                             data['question_title']],
                                                            self.question_title_words_count, self.num_features)
        question_body = transform_sentence_batch_to_vector(self.word_vectors,
                                                           [lower_text(process_html(X)) for X in data['question_body']],
                                                           self.question_body_words_count, self.num_features)
        answer_body = transform_sentence_batch_to_vector(self.word_vectors,
                                                         [lower_text(process_html(X)) for X in data['body']],
                                                         self.answer_body_words_count, self.num_features)
        linguistic_features = np.array([self.linguistic_features_calculator.get_normalized_linguistic_features(id, X)
                                        for id, X in zip(data.id, data.body)])
        thread_features = data[['position', 'relative_position']].as_matrix()
        return [question_title, question_body, answer_body, linguistic_features, thread_features]
