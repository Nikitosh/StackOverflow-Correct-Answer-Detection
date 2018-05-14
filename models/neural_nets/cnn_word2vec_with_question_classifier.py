import logging

import numpy as np
from keras import Input, Model
from keras.layers import Dropout, Dense, concatenate, Conv1D, GlobalMaxPooling1D

from models.neural_nets.neural_net_classifier import NeuralNetClassifier
from utils.utils import transform_sentence_batch_to_vector, lower_text, process_html, THREAD_FEATURES_COUNT


class CnnWord2VecWithQuestionClassifier(NeuralNetClassifier):
    def __init__(self, question_body_words_count, answer_body_words_count, filters_count, kernel_size,
                 hidden_layer_size, dropout):
        super().__init__()
        logging.info('CNN Word2Vec with question classifier')
        logging.info(
            'question_body_words_count = {}, answer_body_words_count = {}, filters_count = {}, kernel_size = {}, hidden_layer_size = {}, dropout = {}'.format(
                question_body_words_count, answer_body_words_count, filters_count, kernel_size, hidden_layer_size,
                dropout))

        self.question_body_words_count = question_body_words_count
        self.answer_body_words_count = answer_body_words_count

        question_body_input = Input(shape=(self.question_body_words_count, self.num_features),
                                    name='question_body_input')
        answer_body_input = Input(shape=(self.answer_body_words_count, self.num_features),
                                  name='answer_body_input')
        linguistic_features = Input(shape=(self.linguistic_features_calculator.LINGUISTIC_FEATURES_COUNT,),
                                    name='linguistic_features')
        thread_features = Input(shape=(THREAD_FEATURES_COUNT,), name='thread_features')

        question_body_features = Conv1D(filters_count, kernel_size, padding='valid', strides=1, activation='relu')(
            question_body_input)
        question_body_features = GlobalMaxPooling1D()(question_body_features)
        question_body_features = Dropout(dropout, name='question_body_features')(question_body_features)
        answer_body_features = Conv1D(filters_count, kernel_size, padding='valid', strides=1, activation='relu')(
            answer_body_input)
        answer_body_features = GlobalMaxPooling1D()(answer_body_features)
        answer_body_features = Dropout(dropout, name='answer_body_features')(answer_body_features)

        features = concatenate([question_body_features, answer_body_features, linguistic_features, thread_features],
                               name='features')
        fc = Dense(hidden_layer_size, activation='relu')(features)
        output = Dense(1, activation='sigmoid', name='output')(fc)

        self.model = Model(inputs=[question_body_input, answer_body_input, linguistic_features, thread_features],
                           outputs=[output])
        self.compile_model()

    def transform_input(self, data):
        question_body = transform_sentence_batch_to_vector(self.word_vectors,
                                                           [lower_text(process_html(X)) for X in data['question_body']],
                                                           self.question_body_words_count, self.num_features)
        answer_body = transform_sentence_batch_to_vector(self.word_vectors,
                                                         [lower_text(process_html(X)) for X in data['body']],
                                                         self.answer_body_words_count, self.num_features)
        linguistic_features = np.array([self.linguistic_features_calculator.get_normalized_linguistic_features(id, X)
                                        for id, X in zip(data.id, data.body)])
        thread_features = data[['position', 'relative_position']].as_matrix()
        return [question_body, answer_body, linguistic_features, thread_features]
