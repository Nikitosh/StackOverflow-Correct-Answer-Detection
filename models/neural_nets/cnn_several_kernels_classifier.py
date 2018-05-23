import logging

from keras import Input, Model
from keras.layers import Dropout, Dense, concatenate, Conv1D, GlobalMaxPooling1D, Concatenate, merge

from models.neural_nets.neural_net_classifier import NeuralNetClassifier
from utils.html_utils import process_html
from utils.word_utils import lower_text


class CnnSeveralKernelsClassifier(NeuralNetClassifier):
    def __init__(self, question_body_words_count, answer_body_words_count, filters_count, kernel_sizes,
                 hidden_layer_size, dropout, mode):
        super().__init__()
        logging.info('CNN Word2Vec with question classifier')
        logging.info(
            'question_body_words_count = {}, answer_body_words_count = {}, filters_count = {}, kernel_sizes = {}, hidden_layer_size = {}, dropout = {}, mode = {}'.format(
                question_body_words_count, answer_body_words_count, filters_count, kernel_sizes, hidden_layer_size,
                dropout, mode))

        self.question_body_words_count = question_body_words_count
        self.answer_body_words_count = answer_body_words_count
        self.filters_count = filters_count
        self.kernel_sizes = kernel_sizes
        self.hidden_layer_size = hidden_layer_size
        self.dropout = dropout
        self.mode = mode

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
        other_features = Input(shape=(NeuralNetClassifier.OTHER_FEATURES_COUNT,), name='other_features')

        question_body_features = self.embedding(question_body_input)
        question_body_features = self.transform_text_features(question_body_features)
        answer_body_features = self.embedding(answer_body_input)
        answer_body_features = self.transform_text_features(answer_body_features)
        features = merge([question_body_features, answer_body_features], mode=self.mode)
        features = concatenate([features, other_features], name='features')
        fc = Dense(self.hidden_layer_size, activation='relu')(features)
        output = Dense(1, activation='sigmoid', name='output')(fc)

        self.model = Model(inputs=[question_body_input, answer_body_input, other_features],
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
        other_features = self.get_other_features(data)

        return [question_body, answer_body, other_features]
