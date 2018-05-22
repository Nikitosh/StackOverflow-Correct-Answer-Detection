import logging

import numpy as np
from keras import Input, Model
from keras.layers import Dropout, Dense, concatenate, Conv1D, GlobalMaxPooling1D, Concatenate

from models.neural_nets.neural_net_classifier import NeuralNetClassifier
from utils.utils import lower_text, process_html, THREAD_FEATURES_COUNT


class CnnSeveralKernelsClassifier(NeuralNetClassifier):
    def __init__(self, question_body_words_count, answer_body_words_count, filters_count, kernel_sizes,
                 hidden_layer_size, dropout):
        super().__init__()
        logging.info('CNN Word2Vec with question classifier')
        logging.info(
            'question_body_words_count = {}, answer_body_words_count = {}, filters_count = {}, kernel_sizes = {}, hidden_layer_size = {}, dropout = {}'.format(
                question_body_words_count, answer_body_words_count, filters_count, kernel_sizes, hidden_layer_size,
                dropout))

        self.question_body_words_count = question_body_words_count
        self.answer_body_words_count = answer_body_words_count
        self.filters_count = filters_count
        self.kernel_sizes = kernel_sizes
        self.hidden_layer_size = hidden_layer_size
        self.dropout = dropout

        self.convolution_layers = None
        self.maxpool_layers = None
        self.concatenate_layer = None
        self.dropout_layer = None

    def create_model(self):
        self.convolution_layers = []
        self.maxpool_layers = []
        for kernel_size in self.kernel_sizes:
            self.convolution_layers.append(
                Conv1D(self.filters_count, kernel_size, padding='valid', strides=1, activation='relu'))
            self.maxpool_layers.append(GlobalMaxPooling1D())
        self.concatenate_layer = Concatenate()
        self.dropout_layer = Dropout(self.dropout)

        question_body_input = Input(shape=(self.question_body_words_count,), name='question_body_input')
        answer_body_input = Input(shape=(self.answer_body_words_count,), name='answer_body_input')
        linguistic_features = Input(shape=(self.linguistic_features_calculator.LINGUISTIC_FEATURES_COUNT,),
                                    name='linguistic_features')
        thread_features = Input(shape=(THREAD_FEATURES_COUNT,), name='thread_features')

        question_body_features = self.embedding(question_body_input)
        question_body_features = self.transform_text_features(question_body_features)
        answer_body_features = self.embedding(answer_body_input)
        answer_body_features = self.transform_text_features(answer_body_features)
        features = concatenate([question_body_features, answer_body_features, linguistic_features, thread_features],
                               name='features')
        fc = Dense(self.hidden_layer_size, activation='relu')(features)
        output = Dense(1, activation='sigmoid', name='output')(fc)

        self.model = Model(inputs=[question_body_input, answer_body_input, linguistic_features, thread_features],
                           outputs=[output])
        self.compile_model()

    def transform_text_features(self, text_input):
        convolutions = []
        for i in range(len(self.convolution_layers)):
            text_features = self.convolution_layers[i](text_input)
            text_features = self.maxpool_layers[i](text_features)
            convolutions.append(text_features)
        features = convolutions[0] if len(self.convolution_layers) == 1 else self.concatenate_layer(convolutions)
        return self.dropout_layer(features)

    def transform_input(self, data):
        question_body = self.transform_sentence_batch_to_vector(
            [lower_text(process_html(X)) for X in data['question_body']],
            self.question_body_words_count)
        answer_body = self.transform_sentence_batch_to_vector([lower_text(process_html(X)) for X in data['body']],
                                                              self.answer_body_words_count)
        linguistic_features = np.array([self.linguistic_features_calculator.get_normalized_linguistic_features(id, X)
                                        for id, X in zip(data.id, data.body)])
        thread_features = data[['position', 'relative_position']].as_matrix()

        return [question_body, answer_body, linguistic_features, thread_features]
