import logging

from keras import Input, Model
from keras.layers import Dropout, Dense, merge, concatenate

from models.neural_nets.neural_net_classifier import NeuralNetClassifier
from utils.html_utils import process_html
from utils.nn_utils import get_lstm
from utils.word_utils import lower_text


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
        self.lstm_embed_size = lstm_embed_size
        self.hidden_layer_size = hidden_layer_size
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.mode = mode

        self.lstm_layer = None
        self.dropout_layer = None

    def create_model(self):
        self.lstm_layer = get_lstm(self.lstm_embed_size, self.bidirectional, self.dropout)
        self.dropout_layer = Dropout(self.dropout)

        question_title_input = Input(shape=(self.question_title_words_count,), name='question_title_input')
        question_body_input = Input(shape=(self.question_body_words_count,), name='question_body_input')
        answer_body_input = Input(shape=(self.answer_body_words_count,), name='answer_body_input')
        other_features = Input(shape=(NeuralNetClassifier.OTHER_FEATURES_COUNT,), name='other_features')

        question_title_features = self.embedding(question_title_input)
        question_title_features = self.transform_text_features(question_title_features)
        question_body_features = self.embedding(question_body_input)
        question_body_features = self.transform_text_features(question_body_features)
        answer_body_features = self.embedding(answer_body_input)
        answer_body_features = self.transform_text_features(answer_body_features)
        rnn_features = merge([question_title_features, question_body_features, answer_body_features], mode=self.mode)
        rnn_features = Dropout(self.dropout, name='rnn_features')(rnn_features)
        features = concatenate([rnn_features, other_features], name='features')
        fc = Dense(self.hidden_layer_size, activation='relu')(features)
        output = Dense(1, activation='sigmoid', name='output')(fc)

        self.model = Model(
            inputs=[question_title_input, question_body_input, answer_body_input, other_features],
            outputs=[output])
        self.compile_model()

    def transform_text_features(self, text_input):
        text_features = self.lstm_layer(text_input)
        return self.dropout_layer(text_features)

    def transform_input(self, data):
        question_title = self.transform_sentence_batch_to_vector(
            [lower_text(process_html(X)) for X in data['question_title']],
            self.question_title_words_count)
        question_body = self.transform_sentence_batch_to_vector(
            [lower_text(process_html(X)) for X in data['question_body']],
            self.question_body_words_count)
        answer_body = self.transform_sentence_batch_to_vector([lower_text(process_html(X)) for X in data['body']],
                                                              self.answer_body_words_count)
        other_features = self.get_other_features(data)

        return [question_title, question_body, answer_body, other_features]
